cmake_minimum_required(VERSION 3.17)
project(dqm-mini-llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- GoogleTest（手动 clone 到 third_party/ 并接入）---
add_subdirectory(third_party/googletest)

enable_testing()

# --- 项目代码库：自动收集 src/ 下所有 .cc ---
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS src/*.cc)
add_library(dqm-llama-core STATIC ${CORE_SOURCES})
target_include_directories(dqm-llama-core PUBLIC include)

# --- 单元测试：自动收集 tests/ 下所有 .cc ---
file(GLOB GTEST_TEST_FILES CONFIGURE_DEPENDS tests/*.cc)
add_executable(test-Gtest ${GTEST_TEST_FILES})
target_link_libraries(test-Gtest PRIVATE dqm-llama-core GTest::gtest_main)

# 把源码根目录传给测试，用于定位 models/tiny/... 等数据文件
target_compile_definitions(test-Gtest PRIVATE
  DQM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_test(NAME test-Gtest COMMAND test-Gtest)
