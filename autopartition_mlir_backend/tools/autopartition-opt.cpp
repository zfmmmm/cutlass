#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;

namespace {

class AutoPartitionDialect final : public Dialect {
public:
  static StringRef getDialectNamespace() { return "autopartition"; }
  explicit AutoPartitionDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<AutoPartitionDialect>()) {
    // The first version intentionally uses generic MLIR operations instead of
    // TableGen, while the dialect still makes the operation a real MLIR op.
    allowUnknownOperations();
  }
};

static bool isTensor(Value value, unsigned rank, StringRef element = {}) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != static_cast<int64_t>(rank))
    return false;
  return element.empty() || type.getElementType().isa<Float16Type>();
}

static std::optional<int64_t> staticDim(Value value, unsigned index) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || index >= static_cast<unsigned>(type.getRank()) ||
      type.isDynamicDim(index))
    return std::nullopt;
  return type.getDimSize(index);
}

static bool hasGeluEvidence(Operation *op) {
  SmallVector<Operation *, 32> worklist{op};
  SmallPtrSet<Operation *, 32> seen;
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    StringRef name = current->getName().getStringRef();
    if (name.contains("erf") || name.contains("tanh") || name.contains("rsqrt") ||
        current->getLoc().isa<NameLoc>()) {
      if (name.contains("erf") || name.contains("tanh") || name.contains("rsqrt"))
        return true;
      if (auto fused = current->getLoc().dyn_cast<FusedLoc>()) {
        for (Location loc : fused.getLocations())
          if (auto nameLoc = loc.dyn_cast<NameLoc>(); nameLoc &&
              nameLoc.getName().getValue().contains("gelu"))
            return true;
      }
    }
    if (auto nameLoc = current->getLoc().dyn_cast<NameLoc>())
      if (nameLoc.getName().getValue().contains("gelu"))
        return true;
    for (Value result : current->getResults())
      for (Operation *user : result.getUsers())
        if (!seen.contains(user))
          worklist.push_back(user);
  }
  return false;
}

static Value findRankOneSource(Value value, unsigned depth = 0) {
  if (depth > 16)
    return Value();
  if (isTensor(value, 1))
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return Value();
  for (Value operand : producer->getOperands()) {
    if (Value source = findRankOneSource(operand, depth + 1))
      return source;
  }
  return Value();
}

static Operation *followEpilogue(Operation *matmul, Value &bias, Value &output,
                                 bool &geluFound) {
  if (matmul->getNumResults() != 1)
    return nullptr;
  Operation *current = matmul;
  SmallPtrSet<Operation *, 32> seen;
  while (current->getNumResults() == 1) {
    Value result = current->getResult(0);
    SmallVector<Operation *, 4> users(result.getUsers());
    if (users.empty())
      break;
    if (users.size() != 1)
      return nullptr;
    Operation *user = users.front();
    if (user->hasTrait<OpTrait::IsTerminator>())
      break;
    if (!seen.insert(user).second)
      return nullptr;
    StringRef name = user->getName().getStringRef();
    if (!name.starts_with("linalg.") && !name.starts_with("tensor.") &&
        !name.starts_with("arith.") && !name.starts_with("math.") &&
        !name.starts_with("chlo."))
      break;
    for (Value operand : user->getOperands()) {
      if (operand == result)
        continue;
      if (Value source = findRankOneSource(operand))
        bias = source;
    }
    if (name.contains("erf") || name.contains("tanh") ||
        hasGeluEvidence(user))
      geluFound = true;
    current = user;
  }
  output = current->getNumResults() ? current->getResult(0) : Value();
  return output ? current : nullptr;
}

static bool isAllowedEpilogueOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.starts_with("linalg.") || name.starts_with("tensor.") ||
         name.starts_with("arith.") || name.starts_with("math.") ||
         name.starts_with("chlo.");
}

static bool hasNestedGeluEvidence(Operation *op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    StringRef name = nested->getName().getStringRef();
    if (name.contains("erf") || name.contains("tanh") || name.contains("rsqrt"))
      found = true;
  });
  return found || hasGeluEvidence(op);
}

static Operation *findEpilogueDag(Operation *matmul, Value &bias, Value &output,
                                  bool &geluFound) {
  auto function = matmul->getParentOfType<func::FuncOp>();
  if (!function || function.getBody().empty())
    return nullptr;
  auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().back().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1)
    return nullptr;
  output = returnOp.getOperand(0);
  SmallVector<Value, 64> worklist{output};
  SmallPtrSet<Operation *, 32> dag;
  bool foundMatmul = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (isTensor(value, 1) && value != matmul->getOperand(0) &&
        value != matmul->getOperand(1))
      bias = value;
    Operation *producer = value.getDefiningOp();
    if (!producer || !dag.insert(producer).second)
      continue;
    if (producer == matmul) {
      foundMatmul = true;
      continue;
    }
    if (!isAllowedEpilogueOperation(producer))
      return nullptr;
    geluFound = geluFound || hasNestedGeluEvidence(producer);
    for (Value operand : producer->getOperands())
      worklist.push_back(operand);
  }
  if (!foundMatmul || !bias || !geluFound)
    return nullptr;
  for (Operation *user : matmul->getResult(0).getUsers())
    if (!dag.contains(user))
      return nullptr;
  return output.getDefiningOp();
}

static Operation *findMatmul(ModuleOp module) {
  Operation *found = nullptr;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "linalg.matmul" && !found)
      found = op;
  });
  return found;
}

static void eraseDeadPureOperations(func::FuncOp function, Operation *keep) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *, 64> dead;
    function.walk([&](Operation *candidate) {
      if (candidate == keep || candidate->hasTrait<OpTrait::IsTerminator>() ||
          candidate->getNumResults() == 0 || !isMemoryEffectFree(candidate))
        return;
      if (llvm::all_of(candidate->getResults(), [](Value result) { return result.use_empty(); }))
        dead.push_back(candidate);
    });
    for (Operation *candidate : llvm::reverse(dead)) {
      candidate->erase();
      changed = true;
    }
  }
}

static StringRef getStringAttr(Operation *op, StringRef name,
                               StringRef fallback = {}) {
  if (auto attr = op->getAttrOfType<StringAttr>(name))
    return attr.getValue();
  return fallback;
}

static int64_t getIntAttr(Operation *op, StringRef name, int64_t fallback = -1) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

static void setStringAttr(Operation *op, StringRef name, StringRef value) {
  op->setAttr(name, StringAttr::get(op->getContext(), value));
}

struct FuseGemmEpiloguePass
    : public PassWrapper<FuseGemmEpiloguePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseGemmEpiloguePass)
  StringRef getArgument() const final { return "autopartition-fuse-gemm-epilogue"; }
  StringRef getDescription() const final {
    return "Fuse a verified Linalg matmul plus bias and GELU use-def chain";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.getContext()->getOrLoadDialect<AutoPartitionDialect>();
    Operation *matmul = findMatmul(module);
    if (!matmul || matmul->getNumOperands() < 2 || matmul->getNumResults() != 1)
      return;
    auto aType = dyn_cast<RankedTensorType>(matmul->getOperand(0).getType());
    auto bType = dyn_cast<RankedTensorType>(matmul->getOperand(1).getType());
    auto outType = dyn_cast<RankedTensorType>(matmul->getResult(0).getType());
    if (!aType || !bType || !outType || aType.getRank() != 2 ||
        bType.getRank() != 2 || outType.getRank() != 2)
      return;

    Value bias, output;
    bool geluFound = false;
    Operation *epilogue = findEpilogueDag(matmul, bias, output, geluFound);
    if (!epilogue || !bias || !output || !isTensor(bias, 1) || !geluFound)
      return;
    auto m = staticDim(output, 0);
    auto n = staticDim(output, 1);
    auto k = staticDim(matmul->getOperand(0), 1);
    auto biasN = staticDim(bias, 0);
    if (!m || !n || !k || !biasN || *biasN != *n)
      return;
    if (matmul->getResult(0).getUses().empty())
      return;

    OpBuilder builder(epilogue);
    OperationState state(epilogue->getLoc(), "autopartition.gemm_bias_gelu");
    state.addOperands({matmul->getOperand(0), matmul->getOperand(1), bias});
    state.addTypes(output.getType());
    state.addAttribute("M", IntegerAttr::get(builder.getI64Type(), *m));
    state.addAttribute("N", IntegerAttr::get(builder.getI64Type(), *n));
    state.addAttribute("K", IntegerAttr::get(builder.getI64Type(), *k));
    state.addAttribute("dtype", builder.getStringAttr("f16"));
    state.addAttribute("target_sm", builder.getStringAttr("sm80"));
    state.addAttribute("tile_m", IntegerAttr::get(builder.getI64Type(), 64));
    state.addAttribute("tile_n", IntegerAttr::get(builder.getI64Type(), 64));
    state.addAttribute("tile_k", IntegerAttr::get(builder.getI64Type(), 64));
    state.addAttribute("thread_count", IntegerAttr::get(builder.getI64Type(), 128));
    state.addAttribute("alignment_bytes", IntegerAttr::get(builder.getI64Type(), 16));
    state.addAttribute("backend", builder.getStringAttr("unknown"));
    state.addAttribute("fallback_reason", builder.getStringAttr(""));
    Operation *fused = builder.create(state);
    output.replaceAllUsesWith(fused->getResult(0));
    eraseDeadPureOperations(matmul->getParentOfType<func::FuncOp>(), fused);
  }
};

struct LegalizeBackendPass
    : public PassWrapper<LegalizeBackendPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeBackendPass)
  StringRef getArgument() const final { return "autopartition-legalize-backend"; }
  StringRef getDescription() const final {
    return "Check AutoPartition backend dtype, shape, layout and target contract";
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() != "autopartition.gemm_bias_gelu")
        return;
      SmallVector<std::string, 8> reasons;
      auto dtype = getStringAttr(op, "dtype");
      auto target = getStringAttr(op, "target_sm");
      int64_t m = getIntAttr(op, "M"), n = getIntAttr(op, "N"), k = getIntAttr(op, "K");
      if (dtype != "f16") reasons.push_back("dtype must be f16");
      if (m < 0 || n < 0 || k < 0) reasons.push_back("M/N/K must be static");
      if (m >= 0 && m % 64) reasons.push_back("M must be a multiple of 64");
      if (n >= 0 && n % 64) reasons.push_back("N must be a multiple of 64");
      if (k >= 0 && k % 64) reasons.push_back("K must be a multiple of 64");
      if (op->getNumOperands() != 3 || !isTensor(op->getOperand(0), 2) ||
          !isTensor(op->getOperand(1), 2) || !isTensor(op->getOperand(2), 1))
        reasons.push_back("expected contiguous rank-2 A/B and rank-1 bias");
      if (getIntAttr(op, "alignment_bytes") < 16)
        reasons.push_back("alignment_bytes must be at least 16");
      if (target != "sm80") reasons.push_back("target_sm must be sm80");
      if (reasons.empty()) {
        setStringAttr(op, "backend", "AutoPartitionBackend");
        setStringAttr(op, "fallback_reason", "");
      } else {
        std::string reason;
        for (size_t i = 0; i < reasons.size(); ++i) {
          if (i) reason += "; ";
          reason += reasons[i];
        }
        setStringAttr(op, "backend", "TorchFallbackBackend");
        setStringAttr(op, "fallback_reason", reason);
      }
    });
  }
};

struct LowerToRuntimePass
    : public PassWrapper<LowerToRuntimePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToRuntimePass)
  StringRef getArgument() const final { return "autopartition-lower-to-runtime"; }
  StringRef getDescription() const final {
    return "Lower AutoPartition operation to an explicit runtime call";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 8> candidates;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "autopartition.gemm_bias_gelu")
        candidates.push_back(op);
    });
    for (Operation *op : candidates) {
      StringRef backend = getStringAttr(op, "backend", "TorchFallbackBackend");
      bool legal = backend == "AutoPartitionBackend";
      StringRef callee = legal ? "autopartition_fused_gemm_bias_gelu"
                               : "torch_fallback_gemm_bias_gelu";
      auto parent = op->getParentOfType<func::FuncOp>();
      if (!parent) continue;
      SmallVector<Type> inputTypes, resultTypes;
      for (Value operand : op->getOperands()) inputTypes.push_back(operand.getType());
      for (Value result : op->getResults()) resultTypes.push_back(result.getType());
      auto functionType = FunctionType::get(module.getContext(), inputTypes, resultTypes);
      if (!module.lookupSymbol<func::FuncOp>(callee)) {
        OpBuilder moduleBuilder(module.getBody(), module.getBody()->end());
        auto declaration = moduleBuilder.create<func::FuncOp>(op->getLoc(), callee, functionType);
        declaration.setPrivate();
        declaration->setAttr("autopartition.backend", StringAttr::get(module.getContext(), backend));
        declaration->setAttr("autopartition.fallback_reason",
                             StringAttr::get(module.getContext(), getStringAttr(op, "fallback_reason")));
      }
      OpBuilder builder(op);
      auto call = builder.create<func::CallOp>(op->getLoc(), callee, resultTypes, op->getOperands());
      call->setAttr("autopartition.backend", StringAttr::get(module.getContext(), backend));
      call->setAttr("autopartition.fallback_reason",
                    StringAttr::get(module.getContext(), getStringAttr(op, "fallback_reason")));
      if (op->getNumResults() == 1)
        op->getResult(0).replaceAllUsesWith(call.getResult(0));
      op->erase();
    }
  }
};

} // namespace

int main(int argc, char **argv) {
  PassRegistration<FuseGemmEpiloguePass>();
  PassRegistration<LegalizeBackendPass>();
  PassRegistration<LowerToRuntimePass>();

  DialectRegistry registry;
  registry.insert<AutoPartitionDialect, arith::ArithDialect, func::FuncDialect,
                  linalg::LinalgDialect, math::MathDialect, tensor::TensorDialect>();
  return asMainReturnCode(MlirOptMain(argc, argv, "AutoPartition MLIR optimizer\n", registry));
}
