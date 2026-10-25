#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "dqm_llama/gguf.h"

// 源码根目录，由 CMake 编译期传入（见 CMakeLists.txt）。兜底：退回当前目录。
#ifndef DQM_TEST_DATA_DIR
#define DQM_TEST_DATA_DIR "."
#endif

namespace {

std::string TestDataPath(const std::string& rel) {
  return std::string(DQM_TEST_DATA_DIR) + "/" + rel;
}

std::string TestGgufPath() { return TestDataPath("models/tiny/test.gguf"); }

std::string TinyManifestPath() {
  return TestDataPath("models/tiny/model.json");
}

std::vector<char> ReadBinaryFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
}

// 临时文件：构造时写入，析构时删掉（异常路径也不留垃圾）。
class TempFile {
 public:
  TempFile(const std::string& name, const std::vector<char>& data)
      : path_(std::filesystem::temp_directory_path() / ("dqm_gguf_" + name)) {
    std::ofstream out(path_, std::ios::binary);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
  }
  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

// --- 手搓 GGUF 字节用的小工具（小端）----------------------------------------

void AppendU32Le(std::vector<char>& data, uint32_t value) {
  data.push_back(static_cast<char>(value & 0xff));
  data.push_back(static_cast<char>((value >> 8) & 0xff));
  data.push_back(static_cast<char>((value >> 16) & 0xff));
  data.push_back(static_cast<char>((value >> 24) & 0xff));
}

void AppendU64Le(std::vector<char>& data, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
  }
}

void AppendString(std::vector<char>& data, const std::string& value) {
  AppendU64Le(data, value.size());
  data.insert(data.end(), value.begin(), value.end());
}

// 就地改一个已存在的 u32 / u64 字段（用于"在真文件上只改一处"）。
void WriteU32Le(std::vector<char>& data, size_t offset, uint32_t value) {
  data[offset + 0] = static_cast<char>(value & 0xff);
  data[offset + 1] = static_cast<char>((value >> 8) & 0xff);
  data[offset + 2] = static_cast<char>((value >> 16) & 0xff);
  data[offset + 3] = static_cast<char>((value >> 24) & 0xff);
}

void WriteU64Le(std::vector<char>& data, size_t offset, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    data[offset + i] = static_cast<char>((value >> (i * 8)) & 0xff);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 主路径
// ---------------------------------------------------------------------------

TEST(Gguf, LoadTestFile) {
  dqm::GgufReader reader;
  ASSERT_TRUE(reader.Load(TestGgufPath())) << reader.load_error;
  EXPECT_EQ(reader.version, 3u);
  EXPECT_EQ(reader.n_tensors, 1u);
  EXPECT_EQ(reader.n_metadata, 3u);
}

TEST(Gguf, Metadata) {
  dqm::GgufReader reader;
  ASSERT_TRUE(reader.Load(TestGgufPath())) << reader.load_error;

  ASSERT_EQ(reader.metadata.size(), 3u);
  EXPECT_EQ(reader.metadata[0].key, "general.architecture");
  EXPECT_EQ(reader.metadata[0].value_str, "llama");
  EXPECT_EQ(reader.metadata[1].key, "general.name");
  EXPECT_EQ(reader.metadata[1].value_str, "tiny-test");
  EXPECT_EQ(reader.metadata[2].key, "general.alignment");
  EXPECT_EQ(reader.metadata[2].value_str, "64");
}

TEST(Gguf, TensorInfo) {
  dqm::GgufReader reader;
  ASSERT_TRUE(reader.Load(TestGgufPath())) << reader.load_error;

  ASSERT_EQ(reader.tensors.size(), 1u);
  EXPECT_EQ(reader.tensors[0].name, "token_embd.weight");
  ASSERT_EQ(reader.tensors[0].shape.size(), 2u);
  EXPECT_EQ(reader.tensors[0].shape[0], 128u);
  EXPECT_EQ(reader.tensors[0].shape[1], 32u);
  EXPECT_EQ(reader.tensors[0].type, dqm::kGgmlTypeF32);
  EXPECT_EQ(reader.tensors[0].offset, 0u);
}

TEST(Gguf, DataOffsetComputed) {
  dqm::GgufReader reader;
  ASSERT_TRUE(reader.Load(TestGgufPath())) << reader.load_error;

  // data_offset > 0，且按 general.alignment(64) 对齐。
  EXPECT_GT(reader.data_offset, 0u);
  EXPECT_EQ(reader.data_offset % 64, 0u);
}

TEST(Gguf, RepeatedLoadResetsState) {
  dqm::GgufReader reader;
  ASSERT_TRUE(reader.Load(TestGgufPath())) << reader.load_error;

  // 第二次加载一个非 GGUF 文件 —— 失败，且状态被重置。
  EXPECT_FALSE(reader.Load(TinyManifestPath()));
  EXPECT_EQ(reader.version, 0u);
  EXPECT_EQ(reader.n_tensors, 0u);
  EXPECT_EQ(reader.n_metadata, 0u);
  EXPECT_TRUE(reader.metadata.empty());
  EXPECT_TRUE(reader.tensors.empty());
  EXPECT_EQ(reader.data_offset, 0u);
  EXPECT_EQ(reader.load_error, "invalid GGUF magic");
}

// ---------------------------------------------------------------------------
// 错误路径
// ---------------------------------------------------------------------------

TEST(Gguf, UnsupportedVersionRejected) {
  std::vector<char> data = ReadBinaryFile(TestGgufPath());
  ASSERT_FALSE(data.empty());
  WriteU32Le(data, 4, 2);  // 版本字段改成 2

  TempFile file("bad_version.gguf", data);
  dqm::GgufReader reader;
  EXPECT_FALSE(reader.Load(file.path()));
  EXPECT_EQ(reader.load_error, "unsupported GGUF version: 2");
}

TEST(Gguf, TruncatedFileRejected) {
  std::vector<char> data = ReadBinaryFile(TestGgufPath());
  ASSERT_GT(data.size(), 12u);
  data.resize(12);  // 砍到只剩 header 开头

  TempFile file("truncated.gguf", data);
  dqm::GgufReader reader;
  EXPECT_FALSE(reader.Load(file.path()));
  EXPECT_FALSE(reader.load_error.empty());
}

TEST(Gguf, MissingFile) {
  dqm::GgufReader reader;
  EXPECT_FALSE(reader.Load(TestDataPath("models/tiny/nonexistent.gguf")));
  EXPECT_FALSE(reader.load_error.empty());
}

TEST(Gguf, InvalidMagic) {
  dqm::GgufReader reader;
  EXPECT_FALSE(reader.Load(TinyManifestPath()));  // model.json 不是 GGUF
  EXPECT_EQ(reader.load_error, "invalid GGUF magic");
}

TEST(Gguf, InvalidAlignmentRejected) {
  std::vector<char> data = ReadBinaryFile(TestGgufPath());

  // general.alignment 的值紧跟在 key 串之后 4 字节（u32 类型码）。
  const std::string needle = "general.alignment";
  auto it = std::search(data.begin(), data.end(), needle.begin(), needle.end());
  ASSERT_TRUE(it != data.end());
  size_t alignment_value_offset =
      static_cast<size_t>(it - data.begin()) + needle.size() + 4;
  WriteU32Le(data, alignment_value_offset, 48);  // 48 不是 2 的幂

  TempFile file("bad_alignment.gguf", data);
  dqm::GgufReader reader;
  EXPECT_FALSE(reader.Load(file.path()));
  EXPECT_EQ(reader.load_error, "invalid general.alignment");
}

TEST(Gguf, DuplicateMetadataKeyRejected) {
  // 手搓一个极小的 GGUF：两条 key 相同的 metadata。
  std::vector<char> data;
  data.insert(data.end(), {'G', 'G', 'U', 'F'});
  AppendU32Le(data, 3);   // version
  AppendU64Le(data, 0);   // n_tensors
  AppendU64Le(data, 2);   // n_metadata

  AppendString(data, "general.name");
  AppendU32Le(data, dqm::kGgufTypeString);
  AppendString(data, "first");

  AppendString(data, "general.name");
  AppendU32Le(data, dqm::kGgufTypeString);
  AppendString(data, "second");

  TempFile file("duplicate_metadata.gguf", data);
  dqm::GgufReader reader;
  EXPECT_FALSE(reader.Load(file.path()));
  EXPECT_EQ(reader.load_error, "duplicate metadata key: general.name");
}

TEST(Gguf, ZeroTensorDimensionRejected) {
  std::vector<char> data = ReadBinaryFile(TestGgufPath());

  const std::string tensor_name = "token_embd.weight";
  auto it = std::search(data.begin(), data.end(), tensor_name.begin(),
                        tensor_name.end());
  ASSERT_TRUE(it != data.end());
  // name 之后是 n_dims(u32)，再之后才是 dim[0](u64)。
  size_t first_dim_offset =
      static_cast<size_t>(it - data.begin()) + tensor_name.size() + 4;
  WriteU64Le(data, first_dim_offset, 0);

  TempFile file("zero_dim.gguf", data);
  dqm::GgufReader reader;
  EXPECT_FALSE(reader.Load(file.path()));
  EXPECT_EQ(reader.load_error,
            "invalid zero tensor dimension for: token_embd.weight");
}

TEST(Gguf, TensorDataBoundsChecked) {
  std::vector<char> data = ReadBinaryFile(TestGgufPath());

  const std::string tensor_name = "token_embd.weight";
  auto it = std::search(data.begin(), data.end(), tensor_name.begin(),
                        tensor_name.end());
  ASSERT_TRUE(it != data.end());
  // name + n_dims(u32) + dim0(u64) + dim1(u64) + type(u32) 之后是 offset(u64)。
  size_t offset_field = static_cast<size_t>(it - data.begin()) +
                        tensor_name.size() + 4 + 8 + 8 + 4;
  WriteU64Le(data, offset_field, static_cast<uint64_t>(data.size()));

  TempFile file("bad_tensor_bounds.gguf", data);
  dqm::GgufReader reader;
  EXPECT_FALSE(reader.Load(file.path()));
  EXPECT_EQ(reader.load_error,
            "tensor data outside file bounds: token_embd.weight");
}

// ---------------------------------------------------------------------------
// 类型名
// ---------------------------------------------------------------------------

TEST(Gguf, GgmlTypeNameMapping) {
  EXPECT_EQ(std::string(dqm::GgmlTypeName(dqm::kGgmlTypeF32)), "F32");
  EXPECT_EQ(std::string(dqm::GgmlTypeName(dqm::kGgmlTypeF16)), "F16");
  EXPECT_EQ(std::string(dqm::GgmlTypeName(dqm::kGgmlTypeQ40)), "Q4_0");
  EXPECT_EQ(std::string(dqm::GgmlTypeName(dqm::kGgmlTypeQ80)), "Q8_0");
  EXPECT_EQ(std::string(dqm::GgmlTypeName(999)), "UNKNOWN");
}

TEST(Gguf, GgufValueTypeNameMapping) {
  EXPECT_EQ(std::string(dqm::GgufValueTypeName(dqm::kGgufTypeUint32)), "uint32");
  EXPECT_EQ(std::string(dqm::GgufValueTypeName(dqm::kGgufTypeString)), "string");
  EXPECT_EQ(std::string(dqm::GgufValueTypeName(dqm::kGgufTypeArray)), "array");
  EXPECT_EQ(std::string(dqm::GgufValueTypeName(999)), "unknown");
}
