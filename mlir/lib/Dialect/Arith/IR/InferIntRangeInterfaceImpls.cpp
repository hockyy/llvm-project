//===- InferIntRangeInterfaceImpls.cpp - Integer range impls for arith -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

#include <optional>
#include <utility>

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::intrange;

static intrange::OverflowFlags
convertArithOverflowFlags(arith::IntegerOverflowFlags flags) {
  intrange::OverflowFlags retFlags = intrange::OverflowFlags::None;
  if (bitEnumContainsAny(flags, arith::IntegerOverflowFlags::nsw))
    retFlags |= intrange::OverflowFlags::Nsw;
  if (bitEnumContainsAny(flags, arith::IntegerOverflowFlags::nuw))
    retFlags |= intrange::OverflowFlags::Nuw;
  return retFlags;
}

template <typename Op>
static bool proveNoOverflow(const APInt &lhs, const APInt &rhs, Op op) {
  bool overflow = false;
  (void)op(lhs, rhs, overflow);
  return !overflow;
}

template <typename Op>
static bool
proveNoOverflowForPairs(ArrayRef<std::pair<const APInt *, const APInt *>> pairs,
                        Op op) {
  for (const auto &[lhs, rhs] : pairs) {
    if (!proveNoOverflow(*lhs, *rhs, op))
      return false;
  }
  return true;
}

static OverflowFlags proveNoOverflowFlags(
    ArrayRef<ConstantIntRanges> args,
    function_ref<bool(ArrayRef<ConstantIntRanges>)> proveSigned,
    function_ref<bool(ArrayRef<ConstantIntRanges>)> proveUnsigned) {
  OverflowFlags flags = OverflowFlags::None;
  if (proveSigned(args))
    flags |= OverflowFlags::Nsw;
  if (proveUnsigned(args))
    flags |= OverflowFlags::Nuw;
  return flags;
}

static bool proveNoSignedAddOverflow(ArrayRef<ConstantIntRanges> argRanges) {
  const APInt &lhsMin = argRanges[0].smin();
  const APInt &lhsMax = argRanges[0].smax();
  const APInt &rhsMin = argRanges[1].smin();
  const APInt &rhsMax = argRanges[1].smax();
  // Signed add is monotone in both operands, so it is enough to check
  // the interval endpoints to prove no signed wrap for the whole range.
  return proveNoOverflowForPairs(
      {{&lhsMin, &rhsMin}, {&lhsMax, &rhsMax}},
      [](const APInt &lhs, const APInt &rhs, bool &overflow) {
        return lhs.sadd_ov(rhs, overflow);
      });
}

static bool proveNoUnsignedAddOverflow(ArrayRef<ConstantIntRanges> argRanges) {
  return proveNoOverflow(
      argRanges[0].umax(), argRanges[1].umax(),
      [](const APInt &lhs, const APInt &rhs, bool &overflow) {
        return lhs.uadd_ov(rhs, overflow);
      });
}

static bool proveNoSignedSubOverflow(ArrayRef<ConstantIntRanges> argRanges) {
  const APInt &lhsMin = argRanges[0].smin();
  const APInt &lhsMax = argRanges[0].smax();
  const APInt &rhsMin = argRanges[1].smin();
  const APInt &rhsMax = argRanges[1].smax();
  // For lhs - rhs, the extrema occur at (lhsMin - rhsMax) and
  // (lhsMax - rhsMin). If both are no-wrap, the full interval is no-wrap.
  return proveNoOverflowForPairs(
      {{&lhsMin, &rhsMax}, {&lhsMax, &rhsMin}},
      [](const APInt &lhs, const APInt &rhs, bool &overflow) {
        return lhs.ssub_ov(rhs, overflow);
      });
}

static bool proveNoUnsignedSubOverflow(ArrayRef<ConstantIntRanges> argRanges) {
  return argRanges[0].umin().uge(argRanges[1].umax());
}

static bool proveNoSignedMulOverflow(ArrayRef<ConstantIntRanges> argRanges) {
  const APInt &lhsMin = argRanges[0].smin();
  const APInt &lhsMax = argRanges[0].smax();
  const APInt &rhsMin = argRanges[1].smin();
  const APInt &rhsMax = argRanges[1].smax();
  // Signed multiply is not monotone across sign changes, so conservatively
  // require all four corner products to be no-wrap.
  return proveNoOverflowForPairs(
      {{&lhsMin, &rhsMin},
       {&lhsMin, &rhsMax},
       {&lhsMax, &rhsMin},
       {&lhsMax, &rhsMax}},
      [](const APInt &lhs, const APInt &rhs, bool &overflow) {
        return lhs.smul_ov(rhs, overflow);
      });
}

static bool proveNoUnsignedMulOverflow(ArrayRef<ConstantIntRanges> argRanges) {
  return proveNoOverflow(
      argRanges[0].umax(), argRanges[1].umax(),
      [](const APInt &lhs, const APInt &rhs, bool &overflow) {
        return lhs.umul_ov(rhs, overflow);
      });
}

static OverflowFlags proveNoOverflowForAdd(ArrayRef<ConstantIntRanges> args) {
  return proveNoOverflowFlags(args, proveNoSignedAddOverflow,
                              proveNoUnsignedAddOverflow);
}

static OverflowFlags proveNoOverflowForSub(ArrayRef<ConstantIntRanges> args) {
  return proveNoOverflowFlags(args, proveNoSignedSubOverflow,
                              proveNoUnsignedSubOverflow);
}

static OverflowFlags proveNoOverflowForMul(ArrayRef<ConstantIntRanges> args) {
  return proveNoOverflowFlags(args, proveNoSignedMulOverflow,
                              proveNoUnsignedMulOverflow);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void arith::ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                          SetIntRangeFn setResultRange) {
  if (auto scalarCstAttr = llvm::dyn_cast_or_null<IntegerAttr>(getValue())) {
    const APInt &value = scalarCstAttr.getValue();
    setResultRange(getResult(), ConstantIntRanges::constant(value));
    return;
  }
  if (auto arrayCstAttr =
          llvm::dyn_cast_or_null<DenseIntElementsAttr>(getValue())) {
    if (arrayCstAttr.isSplat()) {
      setResultRange(getResult(), ConstantIntRanges::constant(
                                      arrayCstAttr.getSplatValue<APInt>()));
      return;
    }

    std::optional<ConstantIntRanges> result;
    for (const APInt &val : arrayCstAttr) {
      auto range = ConstantIntRanges::constant(val);
      result = (result ? result->rangeUnion(range) : range);
    }

    assert(result && "Zero-sized vectors are not allowed");
    setResultRange(getResult(), *result);
    return;
  }
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

void arith::AddIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  OverflowFlags declaredFlags = convertArithOverflowFlags(getOverflowFlags());
  ConstantIntRanges range = inferAdd(argRanges, declaredFlags);
  OverflowFlags overflowFlags =
      proveNoOverflowForAdd(argRanges) | declaredFlags;
  setResultRange(getResult(), range.withOverflowFlags(overflowFlags));
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

void arith::SubIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  OverflowFlags declaredFlags = convertArithOverflowFlags(getOverflowFlags());
  ConstantIntRanges range = inferSub(argRanges, declaredFlags);
  OverflowFlags overflowFlags =
      proveNoOverflowForSub(argRanges) | declaredFlags;
  setResultRange(getResult(), range.withOverflowFlags(overflowFlags));
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

void arith::MulIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  OverflowFlags declaredFlags = convertArithOverflowFlags(getOverflowFlags());
  ConstantIntRanges range = inferMul(argRanges, declaredFlags);
  OverflowFlags overflowFlags =
      proveNoOverflowForMul(argRanges) | declaredFlags;
  setResultRange(getResult(), range.withOverflowFlags(overflowFlags));
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

void arith::DivUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivU(argRanges));
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

void arith::DivSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivS(argRanges));
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivUIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferCeilDivU(argRanges));
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

void arith::CeilDivSIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferCeilDivS(argRanges));
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

void arith::FloorDivSIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  return setResultRange(getResult(), inferFloorDivS(argRanges));
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

void arith::RemUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemU(argRanges));
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

void arith::RemSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemS(argRanges));
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

void arith::AndIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferAnd(argRanges));
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

void arith::OrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferOr(argRanges));
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

void arith::XOrIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferXor(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

void arith::MaxSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxS(argRanges));
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

void arith::MaxUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxU(argRanges));
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

void arith::MinSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinS(argRanges));
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

void arith::MinUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinU(argRanges));
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

void arith::ExtUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), extUIRange(argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

void arith::ExtSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), extSIRange(argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

void arith::TruncIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                        SetIntRangeFn setResultRange) {
  unsigned destWidth =
      ConstantIntRanges::getStorageBitwidth(getResult().getType());
  setResultRange(getResult(), truncRange(argRanges[0], destWidth));
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

void arith::IndexCastOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);

  if (srcWidth < destWidth)
    setResultRange(getResult(), extSIRange(argRanges[0], destWidth));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncRange(argRanges[0], destWidth));
  else
    setResultRange(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

void arith::IndexCastUIOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  Type sourceType = getOperand().getType();
  Type destType = getResult().getType();
  unsigned srcWidth = ConstantIntRanges::getStorageBitwidth(sourceType);
  unsigned destWidth = ConstantIntRanges::getStorageBitwidth(destType);

  if (srcWidth < destWidth)
    setResultRange(getResult(), extUIRange(argRanges[0], destWidth));
  else if (srcWidth > destWidth)
    setResultRange(getResult(), truncRange(argRanges[0], destWidth));
  else
    setResultRange(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

void arith::CmpIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  arith::CmpIPredicate arithPred = getPredicate();
  intrange::CmpPredicate pred = static_cast<intrange::CmpPredicate>(arithPred);
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnes(1);

  std::optional<bool> truthValue = intrange::evaluatePred(pred, lhs, rhs);
  if (truthValue.has_value() && *truthValue)
    min = max;
  else if (truthValue.has_value() && !(*truthValue))
    max = min;

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

void arith::SelectOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> argRanges, SetIntLatticeFn setResultRange) {
  std::optional<APInt> mbCondVal =
      argRanges[0].isUninitialized()
          ? std::nullopt
          : argRanges[0].getValue().getConstantValue();

  const IntegerValueRange &trueCase = argRanges[1];
  const IntegerValueRange &falseCase = argRanges[2];

  if (mbCondVal) {
    if (mbCondVal->isZero())
      setResultRange(getResult(), falseCase);
    else
      setResultRange(getResult(), trueCase);
    return;
  }

  // When one of the ranges is uninitialized, set the whole range to max
  // otherwise the result will ignore the uninitialized range.
  if (trueCase.isUninitialized() || falseCase.isUninitialized())
    setResultRange(getResult(), IntegerValueRange::getMaxRange(getResult()));
  else
    setResultRange(getResult(), IntegerValueRange::join(trueCase, falseCase));
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

void arith::ShLIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShl(argRanges, convertArithOverflowFlags(
                                                      getOverflowFlags())));
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

void arith::ShRUIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrU(argRanges));
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

void arith::ShRSIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                       SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrS(argRanges));
}
