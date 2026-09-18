// tensor.cc
//
// 详见 tensor-spec.md §8（辅助函数）与 §9（验收标准）。

#include "dqm_llama/tensor.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dqm {

namespace {

// ---------------------------------------------------------------------------
// 辅助函数 —— 公共方法全都建立在它们之上
// ---------------------------------------------------------------------------

// caller 为 nullptr 时返回 "Tensor"
const char* CallerName(const char* caller) {
  return caller == nullptr ? "Tensor" : caller;
}

// {2, 3} -> "[2, 3]"
std::string ShapeToString(const std::vector<int>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      s += ", ";
    }
    s += std::to_string(shape[i]);
  }
  s += "]";
  return s;
}

// 校验每维 > 0、元素数不溢出 size_t；返回元素总数。
//
// ⚠️ 顺序很重要：必须先判 dim <= 0，再做溢出判断。
//    否则 dim == 0 时会执行 max / 0 —— 整数除零是 UB，在 x86 上直接崩（SIGFPE）。
size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
  size_t total = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    const int dim = shape[axis];

    if (dim <= 0) {
      throw std::runtime_error(std::string(CallerName(caller)) +
                               ": dimension at axis " + std::to_string(axis) +
                               " must be positive, got " + std::to_string(dim) +
                               " in shape " + ShapeToString(shape));
    }

    // 先除后比。不要写 total * dim > max —— 那一步自己就溢出了，条件永远为假。
    const size_t dim_size = static_cast<size_t>(dim);
    if (total > std::numeric_limits<size_t>::max() / dim_size) {
      throw std::runtime_error(std::string(CallerName(caller)) +
                               ": shape element count overflow for " +
                               ShapeToString(shape));
    }

    total *= dim_size;
  }
  return total;
}

// 秩不符 -> std::runtime_error
void CheckRank(const Tensor& t, int expected, const char* caller) {
  if (t.num_dims() != expected) {
    throw std::runtime_error(std::string(caller) + " expected " +
                             std::to_string(expected) + "D tensor, got " +
                             ShapeToString(t.shape));
  }
}

// 下标越界 -> std::out_of_range
//
// ⚠️ 必须先判 flat_index < 0：否则 -1 拿去算下标会被转成巨大的 size_t。
void CheckAxisIndex(const Tensor& t, int axis, int flat_index,
                    const char* caller) {
  const int dim = t.shape[axis];
  if (flat_index < 0 || flat_index >= dim) {
    throw std::out_of_range(
        std::string(caller) + ": index " + std::to_string(flat_index) +
        " out of range for axis " + std::to_string(axis) + " with size " +
        std::to_string(dim) + " in tensor " + ShapeToString(t.shape));
  }
}

// 改形时的公共校验：新形状必须合法，且元素总数与原张量一致。
// 返回新形状的元素总数。失败抛 std::runtime_error。
size_t CheckedReshapeNumel(size_t current_numel,
                           const std::vector<int>& new_shape,
                           const char* caller) {
  const size_t new_total = CheckedNumel(new_shape, CallerName(caller));
  if (new_total != current_numel) {
    throw std::runtime_error(
        std::string(CallerName(caller)) + ": cannot reshape tensor with " +
        std::to_string(current_numel) + " elements into shape with " +
        std::to_string(new_total) + " elements");
  }
  return new_total;
}

}  // namespace

// ===========================================================================
// 构造
// ===========================================================================

Tensor::Tensor(const std::vector<int>& input_shape, float fill)
    : shape(input_shape) {
  const size_t total = CheckedNumel(shape, "Tensor constructor");
  data.resize(total, fill);
}

// ===========================================================================
// 索引
// ===========================================================================

size_t Tensor::FlatIndex(const std::vector<int>& indices) const {
  // 秩不符：这里期望的秩是调用方传来的下标个数，所以内联判断（用不上 CheckRank）
  if (indices.size() != shape.size()) {
    throw std::runtime_error(
        "Tensor::FlatIndex expected " + std::to_string(shape.size()) +
        " indices for tensor " + ShapeToString(shape) + ", got " +
        std::to_string(indices.size()));
  }

  for (size_t axis = 0; axis < indices.size(); ++axis) {
    CheckAxisIndex(*this, static_cast<int>(axis), indices[axis],
                   "Tensor::FlatIndex");
  }

  // 行主序：从最后一维往前累加 stride（最后一维 stride = 1）。
  // 注意 i 必须是 int —— size_t 的话 i >= 0 恒成立，会死循环 + 越界。
  size_t flat = 0;
  size_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    flat += static_cast<size_t>(indices[i]) * stride;
    stride *= static_cast<size_t>(shape[i]);
  }
  return flat;
}

float& Tensor::At(const std::vector<int>& indices) {
  return data[FlatIndex(indices)];
}

float Tensor::At(const std::vector<int>& indices) const {
  return data[FlatIndex(indices)];
}

// ---------------------------------------------------------------------------
// At1 ~ At4：快路径访问器
//
// 和 At(vector) 的区别：固定个数的参数 + 内联的 Horner 展开算式，
// 避免每次调用都构造一个 std::vector<int>（那是一次堆分配）。
//
// 行主序算式（每步的 shape[k] 都是"后面所有轴的乘积"的因子）：
//   At1:  i
//   At2:  i*d1 + j
//   At3:  (i*d1 + j)*d2 + k
//   At4:  ((i*d1 + j)*d2 + k)*d3 + l
// ---------------------------------------------------------------------------

float& Tensor::At1(int i) {
  CheckRank(*this, 1, "At1");
  CheckAxisIndex(*this, 0, i, "At1");
  return data[i];
}

float Tensor::At1(int i) const {
  CheckRank(*this, 1, "At1");
  CheckAxisIndex(*this, 0, i, "At1");
  return data[i];
}

float& Tensor::At2(int i, int j) {
  CheckRank(*this, 2, "At2");
  CheckAxisIndex(*this, 0, i, "At2");
  CheckAxisIndex(*this, 1, j, "At2");
  return data[i * shape[1] + j];
}

float Tensor::At2(int i, int j) const {
  CheckRank(*this, 2, "At2");
  CheckAxisIndex(*this, 0, i, "At2");
  CheckAxisIndex(*this, 1, j, "At2");
  return data[i * shape[1] + j];
}

float& Tensor::At3(int i, int j, int k) {
  CheckRank(*this, 3, "At3");
  CheckAxisIndex(*this, 0, i, "At3");
  CheckAxisIndex(*this, 1, j, "At3");
  CheckAxisIndex(*this, 2, k, "At3");
  return data[(i * shape[1] + j) * shape[2] + k];
}

float Tensor::At3(int i, int j, int k) const {
  CheckRank(*this, 3, "At3");
  CheckAxisIndex(*this, 0, i, "At3");
  CheckAxisIndex(*this, 1, j, "At3");
  CheckAxisIndex(*this, 2, k, "At3");
  return data[(i * shape[1] + j) * shape[2] + k];
}

float& Tensor::At4(int i, int j, int k, int l) {
  CheckRank(*this, 4, "At4");
  CheckAxisIndex(*this, 0, i, "At4");
  CheckAxisIndex(*this, 1, j, "At4");
  CheckAxisIndex(*this, 2, k, "At4");
  CheckAxisIndex(*this, 3, l, "At4");
  return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

float Tensor::At4(int i, int j, int k, int l) const {
  CheckRank(*this, 4, "At4");
  CheckAxisIndex(*this, 0, i, "At4");
  CheckAxisIndex(*this, 1, j, "At4");
  CheckAxisIndex(*this, 2, k, "At4");
  CheckAxisIndex(*this, 3, l, "At4");
  return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

// ---------------------------------------------------------------------------
// RowPtr：返回某一行的首地址
//
// 设计意图是"热路径逃生口"：检查一次，之后用裸指针零开销遍历。
// 注意返回的指针不带长度信息，越界读取不会报错 —— 责任在调用方。
// ---------------------------------------------------------------------------

float* Tensor::RowPtr(int row) {
  CheckRank(*this, 2, "row_ptr");
  CheckAxisIndex(*this, 0, row, "row_ptr");
  return data.data() + static_cast<size_t>(row) * shape[1];
}

const float* Tensor::RowPtr(int row) const {
  CheckRank(*this, 2, "row_ptr");
  CheckAxisIndex(*this, 0, row, "row_ptr");
  return data.data() + static_cast<size_t>(row) * shape[1];
}

// ===========================================================================
// 形状
// ===========================================================================

void Tensor::AssertShape(const std::vector<int>& expected,
                         const char* caller) const {
  if (shape != expected) {
    throw std::runtime_error(std::string(CallerName(caller)) +
                             ": shape mismatch. expected " +
                             ShapeToString(expected) + ", got " +
                             ShapeToString(shape));
  }
}

Tensor Tensor::ReshapeChecked(const std::vector<int>& new_shape,
                              const char* caller) const {
  CheckedReshapeNumel(data.size(), new_shape, caller);
  Tensor r = *this;  // 深拷贝一份 data（O(n) —— 见 spec §7.1）
  r.shape = new_shape;
  return r;
}

void Tensor::ReshapeInPlace(const std::vector<int>& new_shape,
                            const char* caller) {
  // 先校验、后赋值 —— 失败时原 shape 必须原封不动（异常安全）
  CheckedReshapeNumel(data.size(), new_shape, caller);
  shape = new_shape;  // 只改 shape 元数据，data 一个字节都不动（零拷贝）
}

// ===========================================================================
// 打印
// ===========================================================================

std::string Tensor::ShapeStringShort() const { return ShapeToString(shape); }

std::string Tensor::ShapeString() const { return ShapeStringShort(); }

void Tensor::Print(const std::string& name, bool print_data) const {
  if (!name.empty()) {
    std::cout << name << " ";
  }
  std::cout << "shape=" << ShapeStringShort() << " size=" << size()
            << std::endl;
  if (print_data) {
    for (size_t i = 0; i < data.size(); ++i) {
      std::cout << std::fixed << std::setprecision(6) << data[i];
      if (i + 1 < data.size()) {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }
}

// ===========================================================================
// 工厂
// ===========================================================================

Tensor MakeTensor1D(int d0, float fill) { return Tensor({d0}, fill); }

Tensor MakeTensor2D(int d0, int d1, float fill) {
  return Tensor({d0, d1}, fill);
}

Tensor MakeTensor3D(int d0, int d1, int d2, float fill) {
  return Tensor({d0, d1, d2}, fill);
}

Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill) {
  return Tensor({d0, d1, d2, d3}, fill);
}

}  // namespace dqm
