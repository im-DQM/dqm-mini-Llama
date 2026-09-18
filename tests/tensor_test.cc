#include <gtest/gtest.h>

#include <climits>
#include <stdexcept>

#include "dqm_llama/tensor.h"

// 测试里反复用到，简化书写（你的项目约定是 dqm 命名空间）
using dqm::Tensor;

// ===========================================================================
// 构造与属性
// ===========================================================================

TEST(TensorConstruction, ShapeSizeAndNumDims) {
    Tensor t({3, 4, 5}, 0.0f);
    EXPECT_EQ(t.num_dims(), 3);
    EXPECT_EQ(t.size(), 60u);
    EXPECT_EQ(t.num_elements(), 60u);
    EXPECT_EQ(t.shape[0], 3);
    EXPECT_EQ(t.shape[1], 4);
    EXPECT_EQ(t.shape[2], 5);
}

TEST(TensorConstruction, DefaultConstructedIsEmpty) {
    Tensor t;
    EXPECT_EQ(t.num_dims(), 0);
    EXPECT_EQ(t.size(), 0u);
}

TEST(TensorConstruction, FillValueAppliedToEveryElement) {
    Tensor t({10}, 3.14f);
    ASSERT_EQ(t.size(), 10u);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 3.14f);
    }
}

TEST(TensorConstruction, ZeroFillByDefault) {
    Tensor t({2, 2});
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

// ===========================================================================
// 构造：非法输入
// ===========================================================================

TEST(TensorConstructionErrors, RejectsZeroDimension) {
    EXPECT_THROW(Tensor({2, 0}, 0.0f), std::runtime_error);
}

TEST(TensorConstructionErrors, RejectsNegativeDimension) {
    EXPECT_THROW(Tensor({2, -3}, 0.0f), std::runtime_error);
}

TEST(TensorConstructionErrors, RejectsOverflowingElementCount) {
    // numel 溢出 size_t —— 必须在分配前检查，不能走到 std::vector 抛 length_error
    EXPECT_THROW(Tensor({INT_MAX, INT_MAX, INT_MAX}, 0.0f), std::runtime_error);
}

// ===========================================================================
// 工厂函数 MakeTensor1D ~ 4D
// ===========================================================================

TEST(TensorFactory, MakeTensor1D) {
    auto t = dqm::MakeTensor1D(10, 1.0f);
    EXPECT_EQ(t.num_dims(), 1);
    EXPECT_EQ(t.shape[0], 10);
    EXPECT_FLOAT_EQ(t[9], 1.0f);
}

TEST(TensorFactory, MakeTensor2D) {
    auto t = dqm::MakeTensor2D(3, 4, 2.0f);
    EXPECT_EQ(t.num_dims(), 2);
    EXPECT_EQ(t.size(), 12u);
    EXPECT_FLOAT_EQ(t[11], 2.0f);
}

TEST(TensorFactory, MakeTensor3D) {
    auto t = dqm::MakeTensor3D(2, 3, 4, 3.0f);
    EXPECT_EQ(t.num_dims(), 3);
    EXPECT_EQ(t.size(), 24u);
}

TEST(TensorFactory, MakeTensor4D) {
    auto t = dqm::MakeTensor4D(2, 3, 4, 5, 4.0f);
    EXPECT_EQ(t.num_dims(), 4);
    EXPECT_EQ(t.size(), 120u);
}

// ===========================================================================
// 索引：行主序（核心不变量）
// ===========================================================================

TEST(TensorIndexing, FlatIndexIsRowMajor) {
    Tensor t({3, 4, 5}, 0.0f);
    EXPECT_EQ(t.FlatIndex({0, 0, 0}), 0u);
    EXPECT_EQ(t.FlatIndex({0, 1, 0}), 5u);
    EXPECT_EQ(t.FlatIndex({0, 0, 1}), 1u);
    EXPECT_EQ(t.FlatIndex({1, 0, 0}), 20u);
    EXPECT_EQ(t.FlatIndex({2, 3, 4}), 59u);
}

TEST(TensorIndexing, AtVectorWritesToRowMajorPosition) {
    Tensor t({2, 3}, 0.0f);
    t.At({1, 2}) = 6.0f;
    EXPECT_FLOAT_EQ(t.data[5], 6.0f);  // 1*3 + 2
    EXPECT_FLOAT_EQ(t.At2(1, 2), 6.0f);
}

TEST(TensorIndexing, OperatorBracketIsRawAccess) {
    Tensor t({5}, 0.0f);
    t[2] = 42.0f;
    EXPECT_FLOAT_EQ(t.data[2], 42.0f);
    EXPECT_FLOAT_EQ(t[2], 42.0f);
}

// ===========================================================================
// 便捷访问器 At1 ~ At4
// ===========================================================================

TEST(TensorAccessors, At1WritesAndReads) {
    Tensor t({5}, 0.0f);
    t.At1(2) = 7.0f;
    EXPECT_FLOAT_EQ(t.At1(2), 7.0f);
}

TEST(TensorAccessors, At2RowMajorPosition) {
    Tensor t({2, 3}, 0.0f);
    t.At2(1, 2) = 9.0f;
    EXPECT_FLOAT_EQ(t.At2(1, 2), 9.0f);
    EXPECT_FLOAT_EQ(t.data[5], 9.0f);
}

TEST(TensorAccessors, At3RowMajorPosition) {
    Tensor t({2, 2, 2}, 0.0f);
    t.At3(1, 1, 1) = 8.0f;
    EXPECT_FLOAT_EQ(t.At3(1, 1, 1), 8.0f);
    EXPECT_FLOAT_EQ(t.data[7], 8.0f);
}

TEST(TensorAccessors, At4RowMajorPosition) {
    Tensor t({2, 2, 2, 2}, 0.0f);
    t.At4(1, 1, 1, 1) = 4.0f;
    EXPECT_FLOAT_EQ(t.At4(1, 1, 1, 1), 4.0f);
    EXPECT_FLOAT_EQ(t.data[15], 4.0f);
}

// ===========================================================================
// 边界值：合法边界必须通过，越界必须抛 out_of_range
// ===========================================================================

TEST(TensorBounds, AcceptsFirstIndex) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_NO_THROW(t.At2(0, 0));
}

TEST(TensorBounds, AcceptsLastValidIndex) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_NO_THROW(t.At2(1, 2));  // 最后一行最后一列
}

TEST(TensorBounds, RejectsIndexEqualToDim) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.At2(0, 3), std::out_of_range);  // 列 == 3 即越界
}

TEST(TensorBounds, RejectsNegativeIndexInVectorForm) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.At({-1, 0}), std::out_of_range);
}

TEST(TensorBounds, RejectsNegativeIndexInConvenienceForm) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.At2(-1, 0), std::out_of_range);
}

TEST(TensorBounds, At3RejectsOutOfRange) {
    Tensor t({2, 2, 2}, 0.0f);
    EXPECT_THROW(t.At3(0, 0, 2), std::out_of_range);
}

// ===========================================================================
// 秩不符：必须抛 runtime_error（不是 out_of_range）
// ===========================================================================

TEST(TensorRankErrors, AtVectorRejectsWrongRank) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.At({1}), std::runtime_error);  // 2D 张量只给 1 个下标
}

TEST(TensorRankErrors, At1On2DTensorThrows) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.At1(0), std::runtime_error);
}

TEST(TensorRankErrors, At2On1DTensorThrows) {
    Tensor t({5}, 0.0f);
    EXPECT_THROW(t.At2(0, 0), std::runtime_error);
}

// ===========================================================================
// RowPtr
// ===========================================================================

TEST(TensorRowPtr, PointsAtRowStart) {
    Tensor t({3, 4}, 0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            t.At2(i, j) = static_cast<float>(i * 10 + j);
        }
    }
    const float* row1 = t.RowPtr(1);
    EXPECT_FLOAT_EQ(row1[0], 10.0f);
    EXPECT_FLOAT_EQ(row1[3], 13.0f);
}

TEST(TensorRowPtr, ThrowsOnNon2DTensor) {
    Tensor t({5}, 0.0f);
    EXPECT_THROW(t.RowPtr(0), std::runtime_error);
}

TEST(TensorRowPtr, ThrowsWhenRowOutOfRange) {
    Tensor t({3, 4}, 0.0f);
    EXPECT_THROW(t.RowPtr(3), std::out_of_range);
}

// ===========================================================================
// AssertShape
// ===========================================================================

TEST(TensorAssertShape, PassesWhenShapeMatches) {
    Tensor t({2, 3, 4}, 0.0f);
    EXPECT_NO_THROW(t.AssertShape({2, 3, 4}, "TestPass"));
}

TEST(TensorAssertShape, ThrowsOnMismatch) {
    Tensor t({2, 3, 4}, 0.0f);
    EXPECT_THROW(t.AssertShape({2, 3, 5}, "TestFail"), std::runtime_error);
}

// ===========================================================================
// ReshapeChecked
// ===========================================================================

TEST(TensorReshape, PreservesElementCountAndRowMajorOrder) {
    Tensor t({2, 3}, 0.0f);
    for (int i = 0; i < 6; ++i) {
        t[i] = static_cast<float>(i);
    }
    Tensor r = t.ReshapeChecked({3, 2}, "Reshape");
    EXPECT_EQ(r.num_dims(), 2);
    EXPECT_EQ(r.shape[0], 3);
    EXPECT_EQ(r.shape[1], 2);
    EXPECT_EQ(r.size(), 6u);
    EXPECT_FLOAT_EQ(r.At2(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(r.At2(2, 1), 5.0f);  // 最后一个元素保序
}

TEST(TensorReshape, ThrowsWhenElementCountDiffers) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.ReshapeChecked({4, 2}, "ReshapeFail"), std::runtime_error);
}

TEST(TensorReshape, ThrowsOnNonPositiveDimension) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.ReshapeChecked({-1, 6}, "ReshapeNeg"), std::runtime_error);
}

// ===========================================================================
// ReshapeInPlace —— 原地改形，零拷贝
// ===========================================================================

TEST(TensorReshapeInPlace, ChangesShapeAndKeepsData) {
    Tensor t({2, 3}, 0.0f);
    for (int i = 0; i < 6; ++i) {
        t[i] = static_cast<float>(i);
    }
    t.ReshapeInPlace({3, 2}, "InPlace");
    EXPECT_EQ(t.num_dims(), 2);
    EXPECT_EQ(t.shape[0], 3);
    EXPECT_EQ(t.shape[1], 2);
    EXPECT_EQ(t.size(), 6u);
    EXPECT_FLOAT_EQ(t.At2(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t.At2(2, 1), 5.0f);  // 最后一个元素保序
}

TEST(TensorReshapeInPlace, DoesNotReallocateDataBuffer) {
    // 这才是"零拷贝"的定义：data 缓冲区地址不变，没有重新分配
    Tensor t({2, 3}, 0.0f);
    const float* before = t.data.data();
    t.ReshapeInPlace({3, 2}, "InPlace");
    EXPECT_EQ(t.data.data(), before);
}

TEST(TensorReshapeInPlace, ReshapesTo1D) {
    Tensor t({2, 3}, 0.0f);
    t.ReshapeInPlace({6}, "InPlaceFlat");
    EXPECT_EQ(t.num_dims(), 1);
    EXPECT_EQ(t.shape[0], 6);
    EXPECT_EQ(t.size(), 6u);
}

TEST(TensorReshapeInPlace, ThrowsWhenElementCountDiffers) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.ReshapeInPlace({4, 2}, "InPlaceFail"), std::runtime_error);
}

TEST(TensorReshapeInPlace, ThrowsOnNonPositiveDimension) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.ReshapeInPlace({-1, 6}, "InPlaceNeg"), std::runtime_error);
}

TEST(TensorReshapeInPlace, LeavesShapeUnchangedOnFailure) {
    // 异常安全：先校验、后赋值 —— 失败时原 shape 必须原封不动
    Tensor t({2, 3}, 0.0f);
    EXPECT_THROW(t.ReshapeInPlace({4, 2}, "InPlaceFail"), std::runtime_error);
    EXPECT_EQ(t.num_dims(), 2);
    EXPECT_EQ(t.shape[0], 2);
    EXPECT_EQ(t.shape[1], 3);
}

// ===========================================================================
// 格式化
// ===========================================================================

TEST(TensorFormat, ShapeStringIsBracketedCommaSeparated) {
    Tensor t({2, 3}, 0.0f);
    EXPECT_EQ(t.ShapeString(), "[2, 3]");
}

TEST(TensorFormat, ShapeStringShortAliasesShapeString) {
    Tensor t({2, 3, 4}, 0.0f);
    EXPECT_EQ(t.ShapeStringShort(), t.ShapeString());
}

// ===========================================================================
// const 正确性：const 张量必须能用只读访问
// ===========================================================================

TEST(TensorConstCorrectness, ConstAccessReadsValues) {
    Tensor t({2, 3}, 4.0f);
    const Tensor& c = t;
    EXPECT_EQ(c.num_dims(), 2);
    EXPECT_EQ(c.size(), 6u);
    EXPECT_FLOAT_EQ(c[5], 4.0f);
    EXPECT_FLOAT_EQ(c.At2(1, 2), 4.0f);
    EXPECT_FLOAT_EQ(c.At({0, 0}), 4.0f);
    const float* p = c.RowPtr(0);
    EXPECT_FLOAT_EQ(p[0], 4.0f);
}
