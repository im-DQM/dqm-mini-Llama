#include "dqm_llama/quantized_tensor.h"

#include <stdexcept>

namespace dqm {

namespace {

std::string FormatShape(const std::vector<int>& shape) {
  std::string s("[");
  for (size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      s += ", ";
    }
  }
  s += ']';
  return s;
}

}  // namespace

size_t QuantizedTensor::num_elements() const {
  size_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    n *= static_cast<size_t>(shape[i]);
  }
  return n;
}

std::string QuantizedTensor::ShapeStringShort() const {
  return FormatShape(shape);
}

void QuantizedTensor::AssertShape(const std::vector<int>& expected,
                                  const char* caller) const {
  if (shape != expected) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch, expected " +
                             FormatShape(expected) + ", got " +
                             ShapeStringShort());
  }
}

Tensor ToTensor(const QuantizedTensor& q) {
  if (q.type != QuantType::kF32) {
    throw std::runtime_error("ToTensor: need type kf32");
  }
  Tensor t(q.shape);
  if (q.f32_data.size() == t.data.size()) {
    t.data = q.f32_data;
  }
  return t;
}

QuantizedTensor ToQuantizedTensor(const Tensor& t) {
  QuantizedTensor qt;
  qt.type = QuantType::kF32;
  qt.shape = t.shape;
  qt.f32_data = t.data;
  return qt;
}

}  // namespace dqm
