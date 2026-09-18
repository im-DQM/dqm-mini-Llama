#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace dqm {
    struct Tensor {
        std::vector<int> shape;
        std::vector<float> data;

        Tensor() = default;
        explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);

        // 属性
        size_t size() const { return data.size(); }
        size_t num_elements() const { return data.size(); }

        int num_dims() const { return static_cast<int>(shape.size()); }

        // 索引访问
        size_t FlatIndex(const std::vector<int>& indices) const;

        float& At(const std::vector<int>& indices);       // 可写
        float At(const std::vector<int>& indices) const;  // 只读

        // 便捷访问
        float& At1(int i);
        float At1(int i) const;
        float& At2(int i, int j);
        float At2(int i, int j) const;
        float& At3(int i, int j, int k);
        float At3(int i, int j, int k) const;
        float& At4(int i, int j, int k, int l);
        float At4(int i, int j, int k, int l) const;

        float& operator[](size_t i) { return data[i]; }
        float operator[](size_t i) const { return data[i]; }

        float* RowPtr(int row);
        const float* RowPtr(int row) const;

        // 形状操作
        void AssertShape(const std::vector<int>& expected,
                         const char* caller) const;

        Tensor ReshapeChecked(const std::vector<int>& new_shape,
                              const char* caller) const;
        void ReshapeInPlace(const std::vector<int>& new_shape,
                            const char* caller);

        // 辅助函数
        std::string ShapeStringShort() const;
        std::string ShapeString() const;
        void Print(const std::string& name = "", bool print_data = false) const;
    };

    // Create a 1D tensor
    Tensor MakeTensor1D(int d0, float fill = 0.0f);

    // Create a 2D tensor
    Tensor MakeTensor2D(int d0, int d1, float fill = 0.0f);

    // Create a 3D tensor
    Tensor MakeTensor3D(int d0, int d1, int d2, float fill = 0.0f);

    // Create a 4D tensor
    Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);
}  // namespace dqm
