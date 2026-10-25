#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dqm {

// ---------------------------------------------------------------------------
// GGML 张量类型（GGUF 用到的子集；数字是文件里写死的编码，不是连续枚举）
// ---------------------------------------------------------------------------
enum GgmlType : uint32_t {
  kGgmlTypeF32 = 0,
  kGgmlTypeF16 = 1,
  kGgmlTypeQ40 = 2,
  kGgmlTypeQ41 = 3,
  kGgmlTypeQ80 = 8,
};

const char* GgmlTypeName(uint32_t type);

// ---------------------------------------------------------------------------
// GGUF metadata 值的类型（数字同样写在文件里）
// ---------------------------------------------------------------------------
enum GgufValueType : uint32_t {
  kGgufTypeUint8 = 0,
  kGgufTypeInt8 = 1,
  kGgufTypeUint16 = 2,
  kGgufTypeInt16 = 3,
  kGgufTypeUint32 = 4,
  kGgufTypeInt32 = 5,
  kGgufTypeFloat32 = 6,
  kGgufTypeBool = 7,
  kGgufTypeString = 8,
  kGgufTypeArray = 9,
  kGgufTypeUint64 = 10,
  kGgufTypeInt64 = 11,
  kGgufTypeFloat64 = 12,
};

const char* GgufValueTypeName(uint32_t type);

// ---------------------------------------------------------------------------
// 张量目录里的一条。offset 是「相对数据段起点」的字节偏移。
// ---------------------------------------------------------------------------
struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> shape;  // 按 GGUF 顺序：shape[0] 最快变化
  uint32_t type = 0;            // GgmlType
  uint64_t offset = 0;
};

// ---------------------------------------------------------------------------
// metadata 里的一条。value_str 是值的「人类可读表示」，不是原始字节。
// ---------------------------------------------------------------------------
struct GgufMetadataKv {
  std::string key;
  uint32_t type = 0;  // GgufValueType
  std::string value_str;
};

// ---------------------------------------------------------------------------
// GGUF v3 读取器的解析结果。Load 成功/失败都用返回值，失败原因在 load_error。
// ---------------------------------------------------------------------------
struct GgufReader {
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  std::vector<GgufMetadataKv> metadata;
  std::vector<GgufTensorInfo> tensors;
  uint64_t data_offset = 0;  // 数据段在文件里的绝对偏移（已按 alignment 对齐）

  bool Load(const std::string& path);

  std::string load_error;
};

// 把 GGUF 内容打到 stdout（inspect-gguf 用）。
void InspectGguf(const GgufReader& reader);

// ---------------------------------------------------------------------------
// 从文件数据段读一段张量的原始字节。data_offset 传 GgufReader::data_offset。
// 失败返回空 vector。
// ---------------------------------------------------------------------------
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset);

// ---------------------------------------------------------------------------
// 按 key 取 metadata 的真值（重新打开文件、linear 找 key，类型要匹配）。
// 找到且类型匹配 -> true 并写 out；否则 false（不动 out）。
// ---------------------------------------------------------------------------
bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out);
bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out);
bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out);
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out);
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out);

}  // namespace dqm
