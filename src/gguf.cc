// gguf.cc —— GGUF v3 文件读取器。

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "dqm_llama/gguf.h"

namespace dqm {

// ---------------------------------------------------------------------------
// 类型名
// ---------------------------------------------------------------------------

const char* GgmlTypeName(uint32_t type) {
  switch (type) {
    case kGgmlTypeF32:
      return "F32";
    case kGgmlTypeF16:
      return "F16";
    case kGgmlTypeQ40:
      return "Q4_0";
    case kGgmlTypeQ41:
      return "Q4_1";
    case kGgmlTypeQ80:
      return "Q8_0";
    default:
      return "UNKNOWN";
  }
}

const char* GgufValueTypeName(uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
      return "uint8";
    case kGgufTypeInt8:
      return "int8";
    case kGgufTypeUint16:
      return "uint16";
    case kGgufTypeInt16:
      return "int16";
    case kGgufTypeUint32:
      return "uint32";
    case kGgufTypeInt32:
      return "int32";
    case kGgufTypeFloat32:
      return "float32";
    case kGgufTypeBool:
      return "bool";
    case kGgufTypeString:
      return "string";
    case kGgufTypeArray:
      return "array";
    case kGgufTypeUint64:
      return "uint64";
    case kGgufTypeInt64:
      return "int64";
    case kGgufTypeFloat64:
      return "float64";
    default:
      return "unknown";
  }
}

// ---------------------------------------------------------------------------
// 内部辅助：小端读取 / 跳过 / 溢出安全算术 / 对齐
// ---------------------------------------------------------------------------

namespace {

constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;

// 量化块的实际字节数：Q4_0=2+16、Q4_1=2+2+16、Q8_0=2+32。
constexpr uint64_t kQ40BlockBytes = 18;
constexpr uint64_t kQ41BlockBytes = 20;
constexpr uint64_t kQ80BlockBytes = 34;

template <typename T>
bool ReadLe(std::ifstream& f, T& out) {
  char buf[sizeof(T)];
  if (!f.read(buf, static_cast<std::streamsize>(sizeof(T)))) return false;
  std::memcpy(&out, buf, sizeof(T));
  return true;
}

bool ReadU32(std::ifstream& f, uint32_t& out) {
  return ReadLe<uint32_t>(f, out);
}

bool ReadU64(std::ifstream& f, uint64_t& out) {
  return ReadLe<uint64_t>(f, out);
}

bool ReadString(std::ifstream& f, std::string& out) {
  uint64_t len = 0;
  if (!ReadU64(f, len)) return false;
  if (len > kMaxStringBytes ||
      len > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  out.resize(static_cast<size_t>(len));
  if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
    return false;
  }
  return true;
}

bool SkipValue(std::ifstream& f, uint32_t type) {
  char dummy[8];
  auto skip_bytes = [&](std::streamsize n) -> bool {
    return static_cast<bool>(f.read(dummy, n));
  };
  switch (type) {
    case kGgufTypeUint8:
    case kGgufTypeInt8:
    case kGgufTypeBool:
      return skip_bytes(1);
    case kGgufTypeUint16:
    case kGgufTypeInt16:
      return skip_bytes(2);
    case kGgufTypeUint32:
    case kGgufTypeInt32:
    case kGgufTypeFloat32:
      return skip_bytes(4);
    case kGgufTypeUint64:
    case kGgufTypeInt64:
    case kGgufTypeFloat64:
      return skip_bytes(8);
    case kGgufTypeString: {
      std::string tmp;
      return ReadString(f, tmp);
    }
    case kGgufTypeArray: {
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type)) return false;
      if (!ReadU64(f, len)) return false;
      if (len > kMaxArrayElements) return false;
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, elem_type)) return false;
      }
      return true;
    }
    default:
      return false;
  }
}

bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
  out = a * b;
  return true;
}

bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out) {
  uint64_t numel = 1;
  for (uint64_t dim : tensor.shape) {
    if (dim == 0) return false;
    if (!MultiplyChecked(numel, dim, numel)) return false;
  }
  out = numel;
  return true;
}

bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
  uint64_t numel = 0;
  if (!TensorNumel(tensor, numel)) return false;
  switch (tensor.type) {
    case kGgmlTypeF32:
      return MultiplyChecked(numel, static_cast<uint64_t>(sizeof(float)), out);
    case kGgmlTypeF16:
      return MultiplyChecked(numel, static_cast<uint64_t>(2), out);
    case kGgmlTypeQ40:
    case kGgmlTypeQ41:
    case kGgmlTypeQ80: {
      if (numel % kQuantBlockSize != 0) return false;
      uint64_t blocks = numel / kQuantBlockSize;
      uint64_t block_bytes = tensor.type == kGgmlTypeQ40 ? kQ40BlockBytes
                             : tensor.type == kGgmlTypeQ41 ? kQ41BlockBytes
                                                           : kQ80BlockBytes;
      return MultiplyChecked(blocks, block_bytes, out);
    }
    default:
      out = 0;
      return true;
  }
}

bool AddChecked(uint64_t a, uint64_t b, uint64_t& out) {
  if (b > std::numeric_limits<uint64_t>::max() - a) return false;
  out = a + b;
  return true;
}

bool ReadValueAsString(std::ifstream& f, uint32_t type, std::string& out) {
  std::ostringstream oss;
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << static_cast<unsigned>(v);
      break;
    }
    case kGgufTypeInt8: {
      int8_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << static_cast<int>(v);
      break;
    }
    case kGgufTypeUint16: {
      uint16_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << v;
      break;
    }
    case kGgufTypeInt16: {
      int16_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << v;
      break;
    }
    case kGgufTypeUint32: {
      uint32_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << v;
      break;
    }
    case kGgufTypeInt32: {
      int32_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << v;
      break;
    }
    case kGgufTypeUint64: {
      uint64_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << v;
      break;
    }
    case kGgufTypeInt64: {
      int64_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << v;
      break;
    }
    case kGgufTypeFloat32: {
      float v = 0.0f;
      if (!ReadLe(f, v)) return false;
      oss << v;
      break;
    }
    case kGgufTypeFloat64: {
      double v = 0.0;
      if (!ReadLe(f, v)) return false;
      oss << v;
      break;
    }
    case kGgufTypeBool: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) return false;
      oss << (v != 0 ? "true" : "false");
      break;
    }
    case kGgufTypeString:
      return ReadString(f, out);
    case kGgufTypeArray: {
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type)) return false;
      if (!ReadU64(f, len)) return false;
      if (len > kMaxArrayElements) return false;
      oss << "[" << len << " x " << GgufValueTypeName(elem_type) << "]";
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, elem_type)) return false;
      }
      break;
    }
    default:
      return false;
  }
  out = oss.str();
  return true;
}

uint64_t ParseAlignment(const std::vector<GgufMetadataKv>& metadata) {
  for (const auto& kv : metadata) {
    if (kv.key != "general.alignment") continue;
    std::size_t parsed = 0;
    uint64_t alignment = 0;
    try {
      alignment = static_cast<uint64_t>(std::stoull(kv.value_str, &parsed));
    } catch (const std::exception&) {
      return 0;
    }
    if (parsed != kv.value_str.size()) return 0;
    if (alignment == 0 || alignment > kMaxAlignment) return 0;
    if ((alignment & (alignment - 1)) != 0) return 0;
    return alignment;
  }
  return kDefaultAlignment;
}

}  // namespace

// ---------------------------------------------------------------------------
// GgufReader::Load
// ---------------------------------------------------------------------------

bool GgufReader::Load(const std::string& path) {
  version = 0;
  n_tensors = 0;
  n_metadata = 0;
  metadata.clear();
  tensors.clear();
  data_offset = 0;
  load_error.clear();

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    load_error = "failed to open: " + path;
    return false;
  }

  char magic[4];
  if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    load_error = "invalid GGUF magic";
    return false;
  }

  if (!ReadU32(f, version)) {
    load_error = "failed to read version";
    return false;
  }
  if (version != 3) {
    load_error = "unsupported GGUF version: " + std::to_string(version);
    return false;
  }

  if (!ReadU64(f, n_tensors)) {
    load_error = "failed to read tensor count";
    return false;
  }
  if (!ReadU64(f, n_metadata)) {
    load_error = "failed to read metadata count";
    return false;
  }
  if (n_tensors > kMaxTensorEntries || n_metadata > kMaxMetadataEntries) {
    load_error = "too many GGUF entries";
    return false;
  }

  std::unordered_set<std::string> keys;
  for (uint64_t i = 0; i < n_metadata; ++i) {
    GgufMetadataKv kv;
    if (!ReadString(f, kv.key)) {
      load_error = "failed to read metadata key";
      return false;
    }
    if (!keys.insert(kv.key).second) {
      load_error = "duplicate metadata key: " + kv.key;
      return false;
    }
    if (!ReadU32(f, kv.type)) {
      load_error = "failed to read metadata type for: " + kv.key;
      return false;
    }
    if (!ReadValueAsString(f, kv.type, kv.value_str)) {
      load_error = "failed to read metadata value for: " + kv.key;
      return false;
    }
    metadata.push_back(std::move(kv));
  }

  for (uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo info;
    if (!ReadString(f, info.name)) {
      load_error = "failed to read tensor name";
      return false;
    }
    uint32_t n_dims = 0;
    if (!ReadU32(f, n_dims)) {
      load_error = "failed to read tensor n_dims for: " + info.name;
      return false;
    }
    if (n_dims == 0 || n_dims > kMaxTensorDims) {
      load_error = "invalid tensor n_dims for: " + info.name;
      return false;
    }
    info.shape.resize(static_cast<size_t>(n_dims));
    for (uint32_t d = 0; d < n_dims; ++d) {
      if (!ReadU64(f, info.shape[d])) {
        load_error = "failed to read tensor shape for: " + info.name;
        return false;
      }
      if (info.shape[d] == 0) {
        load_error = "invalid zero tensor dimension for: " + info.name;
        return false;
      }
    }
    if (!ReadU32(f, info.type)) {
      load_error = "failed to read tensor type for: " + info.name;
      return false;
    }
    if (!ReadU64(f, info.offset)) {
      load_error = "failed to read tensor offset for: " + info.name;
      return false;
    }
    tensors.push_back(std::move(info));
  }

  std::unordered_set<std::string> names;
  for (const auto& tensor : tensors) {
    if (!names.insert(tensor.name).second) {
      load_error = "duplicate tensor name: " + tensor.name;
      return false;
    }
  }

  // 数据段起点：走完 metadata + 目录后的文件指针，再向上取整到 alignment。
  std::streampos current = f.tellg();
  if (current < 0) {
    load_error = "failed to determine GGUF data offset";
    return false;
  }
  data_offset = static_cast<uint64_t>(current);
  uint64_t alignment = ParseAlignment(metadata);
  if (alignment == 0) {
    load_error = "invalid general.alignment";
    return false;
  }
  uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
  data_offset += padding;

  f.seekg(0, std::ios::end);
  std::streampos end = f.tellg();
  if (end < 0 || static_cast<uint64_t>(end) < data_offset) {
    load_error = "GGUF file ended before tensor data section";
    return false;
  }
  uint64_t file_size = static_cast<uint64_t>(end);

  for (const auto& tensor : tensors) {
    uint64_t tensor_size = 0;
    if (!TensorDataSize(tensor, tensor_size)) {
      load_error = "failed to compute tensor size for: " + tensor.name;
      return false;
    }
    if (tensor_size == 0) continue;
    uint64_t tensor_start = 0;
    if (!AddChecked(data_offset, tensor.offset, tensor_start)) {
      load_error = "tensor offset overflow for: " + tensor.name;
      return false;
    }
    uint64_t tensor_end = 0;
    if (!AddChecked(tensor_start, tensor_size, tensor_end)) {
      load_error = "tensor end overflow for: " + tensor.name;
      return false;
    }
    if (tensor_end > file_size) {
      load_error = "tensor data outside file bounds: " + tensor.name;
      return false;
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// 打印
// ---------------------------------------------------------------------------

void InspectGguf(const GgufReader& reader) {
  std::cout << "version: " << reader.version << "\n";
  std::cout << "n_tensors: " << reader.n_tensors << "\n";
  std::cout << "n_metadata: " << reader.n_metadata << "\n";
  std::cout << "data_offset: " << reader.data_offset << "\n";

  std::cout << "metadata:\n";
  for (const auto& kv : reader.metadata) {
    std::cout << "  " << kv.key << " (" << GgufValueTypeName(kv.type)
              << "): " << kv.value_str << "\n";
  }

  std::cout << "tensors:\n";
  for (const auto& t : reader.tensors) {
    std::cout << "  " << t.name << " [";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << t.shape[i];
    }
    std::cout << "] " << GgmlTypeName(t.type) << " offset=" << t.offset << "\n";
  }
}

// ---------------------------------------------------------------------------
// 读张量原始字节
// ---------------------------------------------------------------------------

std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset) {
  std::vector<uint8_t> result;
  uint64_t tensor_size = 0;
  if (!TensorDataSize(info, tensor_size) || tensor_size == 0) {
    return result;
  }

  std::ifstream f(path, std::ios::binary);
  if (!f) return result;

  uint64_t tensor_start = 0;
  if (!AddChecked(data_offset, info.offset, tensor_start)) return result;
  if (tensor_start >
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    return result;
  }

  f.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
  if (!f) return result;

  result.resize(static_cast<size_t>(tensor_size));
  if (!f.read(reinterpret_cast<char*>(result.data()),
              static_cast<std::streamsize>(tensor_size))) {
    return {};
  }
  return result;
}

// ---------------------------------------------------------------------------
// 内部：跳过 header / metadata，linear 找 key
// ---------------------------------------------------------------------------

bool OpenAndFindMetadataKey(std::ifstream& f, const std::string& path,
                            const std::string& key, uint32_t& out_type) {
  f.open(path, std::ios::binary);
  if (!f) return false;
  f.seekg(8, std::ios::beg);  // 跳过 magic(4) + version(4)
  if (!f) return false;

  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  if (!ReadU64(f, n_tensors)) return false;
  if (!ReadU64(f, n_metadata)) return false;
  if (n_metadata > kMaxMetadataEntries) return false;

  for (uint64_t i = 0; i < n_metadata; ++i) {
    std::string read_key;
    uint32_t type = 0;
    if (!ReadString(f, read_key)) return false;
    if (!ReadU32(f, type)) return false;
    if (read_key == key) {
      out_type = type;  // 流停在"类型之后、值之前"
      return true;
    }
    if (!SkipValue(f, type)) return false;
  }
  return false;
}

// ---------------------------------------------------------------------------
// 按 key 取真值
// ---------------------------------------------------------------------------

bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) return false;
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) return false;
      out = v;
      return true;
    }
    case kGgufTypeInt8: {
      int8_t v = 0;
      if (!ReadLe(f, v)) return false;
      out = v;
      return true;
    }
    case kGgufTypeUint16: {
      uint16_t v = 0;
      if (!ReadLe(f, v)) return false;
      out = v;
      return true;
    }
    case kGgufTypeInt16: {
      int16_t v = 0;
      if (!ReadLe(f, v)) return false;
      out = v;
      return true;
    }
    case kGgufTypeUint32: {
      uint32_t v = 0;
      if (!ReadLe(f, v)) return false;
      out = v;
      return true;
    }
    case kGgufTypeInt32: {
      int32_t v = 0;
      if (!ReadLe(f, v)) return false;
      out = v;
      return true;
    }
    case kGgufTypeUint64: {
      uint64_t v = 0;
      if (!ReadLe(f, v)) return false;
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt64: {
      int64_t v = 0;
      if (!ReadLe(f, v)) return false;
      out = v;
      return true;
    }
    default:
      return false;
  }
}

bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) return false;
  switch (type) {
    case kGgufTypeFloat32: {
      float v = 0.0f;
      if (!ReadLe(f, v)) return false;
      out = v;
      return true;
    }
    case kGgufTypeFloat64:
      return ReadLe(f, out);
    default:
      return false;
  }
}

bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) return false;
  if (type != kGgufTypeString) return false;
  return ReadString(f, out);
}

bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) return false;
  if (type != kGgufTypeArray) return false;

  uint32_t elem_type = 0;
  uint64_t len = 0;
  if (!ReadU32(f, elem_type)) return false;
  if (!ReadU64(f, len)) return false;
  if (elem_type != kGgufTypeString) return false;
  if (len > kMaxArrayElements) return false;

  out.clear();
  out.reserve(static_cast<size_t>(len));
  for (uint64_t i = 0; i < len; ++i) {
    std::string s;
    if (!ReadString(f, s)) return false;
    out.push_back(std::move(s));
  }
  return true;
}

bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) return false;
  if (type != kGgufTypeArray) return false;

  uint32_t elem_type = 0;
  uint64_t len = 0;
  if (!ReadU32(f, elem_type)) return false;
  if (!ReadU64(f, len)) return false;
  if (elem_type != kGgufTypeInt32) return false;
  if (len > kMaxArrayElements) return false;

  out.clear();
  out.reserve(static_cast<size_t>(len));
  for (uint64_t i = 0; i < len; ++i) {
    int32_t v = 0;
    if (!ReadLe(f, v)) return false;
    out.push_back(v);
  }
  return true;
}

}  // namespace dqm
