//===- InferIntRangeInterfaceTest.cpp - Unit Tests for InferIntRange... --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

#include <gtest/gtest.h>

using namespace mlir;

TEST(IntRangeAttrs, BasicConstructors) {
  APInt zero = APInt::getZero(64);
  APInt two(64, 2);
  APInt three(64, 3);
  ConstantIntRanges boundedAbove(zero, two, zero, three);
  EXPECT_EQ(boundedAbove.umin(), zero);
  EXPECT_EQ(boundedAbove.umax(), two);
  EXPECT_EQ(boundedAbove.smin(), zero);
  EXPECT_EQ(boundedAbove.smax(), three);
}

TEST(IntRangeAttrs, FromUnsigned) {
  APInt zero = APInt::getZero(64);
  APInt maxInt = APInt::getSignedMaxValue(64);
  APInt minInt = APInt::getSignedMinValue(64);
  APInt minIntPlusOne = minInt + 1;

  ConstantIntRanges canPortToSigned =
      ConstantIntRanges::fromUnsigned(zero, maxInt);
  EXPECT_EQ(canPortToSigned.smin(), zero);
  EXPECT_EQ(canPortToSigned.smax(), maxInt);

  ConstantIntRanges cantPortToSigned =
      ConstantIntRanges::fromUnsigned(zero, minInt);
  EXPECT_EQ(cantPortToSigned.smin(), minInt);
  EXPECT_EQ(cantPortToSigned.smax(), maxInt);

  ConstantIntRanges signedNegative =
      ConstantIntRanges::fromUnsigned(minInt, minIntPlusOne);
  EXPECT_EQ(signedNegative.smin(), minInt);
  EXPECT_EQ(signedNegative.smax(), minIntPlusOne);
}

TEST(IntRangeAttrs, FromSigned) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;
  APInt negOne = zero - 1;
  APInt intMax = APInt::getSignedMaxValue(64);
  APInt intMin = APInt::getSignedMinValue(64);
  APInt uintMax = APInt::getMaxValue(64);

  ConstantIntRanges noUnsignedBound =
      ConstantIntRanges::fromSigned(negOne, one);
  EXPECT_EQ(noUnsignedBound.umin(), zero);
  EXPECT_EQ(noUnsignedBound.umax(), uintMax);

  ConstantIntRanges positive = ConstantIntRanges::fromSigned(one, intMax);
  EXPECT_EQ(positive.umin(), one);
  EXPECT_EQ(positive.umax(), intMax);

  ConstantIntRanges negative = ConstantIntRanges::fromSigned(intMin, negOne);
  EXPECT_EQ(negative.umin(), intMin);
  EXPECT_EQ(negative.umax(), negOne);

  ConstantIntRanges preserved = ConstantIntRanges::fromSigned(zero, one);
  EXPECT_EQ(preserved.umin(), zero);
  EXPECT_EQ(preserved.umax(), one);
}

TEST(IntRangeAttrs, Join) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;
  APInt two = zero + 2;
  APInt intMin = APInt::getSignedMinValue(64);
  APInt intMax = APInt::getSignedMaxValue(64);
  APInt uintMax = APInt::getMaxValue(64);

  ConstantIntRanges maximal(zero, uintMax, intMin, intMax);
  ConstantIntRanges zeroOne(zero, one, zero, one);

  EXPECT_EQ(zeroOne.rangeUnion(maximal), maximal);
  EXPECT_EQ(maximal.rangeUnion(zeroOne), maximal);

  EXPECT_EQ(zeroOne.rangeUnion(zeroOne), zeroOne);

  ConstantIntRanges oneTwo(one, two, one, two);
  ConstantIntRanges zeroTwo(zero, two, zero, two);
  EXPECT_EQ(zeroOne.rangeUnion(oneTwo), zeroTwo);

  ConstantIntRanges zeroOneUnsignedOnly(zero, one, intMin, intMax);
  ConstantIntRanges zeroOneSignedOnly(zero, uintMax, zero, one);
  EXPECT_EQ(zeroOneUnsignedOnly.rangeUnion(zeroOneSignedOnly), maximal);
}

TEST(IntRangeAttrs, OverflowFlags) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;
  APInt two = zero + 2;

  ConstantIntRanges nswOnly(zero, one, zero, one, OverflowFlags::Nsw);
  ConstantIntRanges nuwOnly(one, two, one, two, OverflowFlags::Nuw);

  EXPECT_NE(nswOnly.getOverflowFlags() & OverflowFlags::Nsw,
            OverflowFlags::None);
  EXPECT_EQ(nswOnly.getOverflowFlags() & OverflowFlags::Nuw,
            OverflowFlags::None);

  ConstantIntRanges both =
      nswOnly.withOverflowFlags(OverflowFlags::Nsw | OverflowFlags::Nuw);
  EXPECT_NE(both.getOverflowFlags() & OverflowFlags::Nsw, OverflowFlags::None);
  EXPECT_NE(both.getOverflowFlags() & OverflowFlags::Nuw, OverflowFlags::None);

  // rangeUnion conservatively preserves only proofs present in both inputs.
  EXPECT_EQ(nswOnly.rangeUnion(nuwOnly).getOverflowFlags(),
            OverflowFlags::None);
  EXPECT_EQ(both.rangeUnion(nswOnly).getOverflowFlags(), OverflowFlags::Nsw);

  // intersection preserves proofs from either input.
  EXPECT_EQ(nswOnly.intersection(nuwOnly).getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);
  EXPECT_EQ(both.intersection(nswOnly).getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);
  ConstantIntRanges none(zero, two, zero, two, OverflowFlags::None);
  EXPECT_EQ(nswOnly.intersection(none).getOverflowFlags(), OverflowFlags::Nsw);
}

TEST(IntRangeAttrs, OverflowFlagsPrinting) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;

  auto toString = [](const ConstantIntRanges &r) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << r;
    return buf;
  };

  ConstantIntRanges noFlags(zero, one, zero, one);
  EXPECT_EQ(toString(noFlags), "unsigned : [0, 1] signed : [0, 1]");

  ConstantIntRanges nsw(zero, one, zero, one, OverflowFlags::Nsw);
  EXPECT_EQ(toString(nsw), "unsigned : [0, 1] signed : [0, 1] overflow<nsw>");

  ConstantIntRanges nuw(zero, one, zero, one, OverflowFlags::Nuw);
  EXPECT_EQ(toString(nuw), "unsigned : [0, 1] signed : [0, 1] overflow<nuw>");

  ConstantIntRanges both(zero, one, zero, one,
                         OverflowFlags::Nsw | OverflowFlags::Nuw);
  EXPECT_EQ(toString(both),
            "unsigned : [0, 1] signed : [0, 1] overflow<nsw, nuw>");
}
