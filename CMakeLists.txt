cmake_minimum_required(VERSION 3.17)
project(dqm-mini-llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 默认构建类型 ---
# 不显式指定时默认 Release（-O2）：-O0 下即使 -mavx2 编进去，
# 标量代码的函数/循环开销也会吃掉大部分 SIMD 收益。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# std::thread 需要（thread_pool 模块）。
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- 可选 CUDA 后端 ---
option(DQM_LLAMA_CUDA "Build NVIDIA CUDA backend" OFF)
option(DQM_LLAMA_ASAN "Build with AddressSanitizer" OFF)
if(DQM_LLAMA_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
endif()
if(DQM_LLAMA_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# --- GoogleTest（手动 clone 到 third_party/ 并接入）---
add_subdirectory(third_party/googletest)

enable_testing()

# --- 项目代码库：自动收集 src/ 下所有 .cc（main.cc 除外 —— 它属于可执行）---
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS src/*.cc)
list(REMOVE_ITEM CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)
if(DQM_LLAMA_CUDA)
  list(APPEND CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_attention.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_ops.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_quant.cu)
endif()
add_library(dqm-llama-core STATIC ${CORE_SOURCES})
target_include_directories(dqm-llama-core PUBLIC include)
target_link_libraries(dqm-llama-core PUBLIC Threads::Threads)
if(DQM_LLAMA_CUDA)
  target_compile_definitions(dqm-llama-core PUBLIC DQM_LLAMA_USE_CUDA=1)
  target_link_libraries(dqm-llama-core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# --- CLI 可执行 ---
add_executable(dqm-llama src/main.cc)
target_link_libraries(dqm-llama PRIVATE dqm-llama-core)

# --- 单元测试：自动收集 tests/ 下所有 .cc ---
file(GLOB GTEST_TEST_FILES CONFIGURE_DEPENDS tests/*.cc)
add_executable(test-Gtest ${GTEST_TEST_FILES})
target_link_libraries(test-Gtest PRIVATE dqm-llama-core GTest::gtest_main)

# 把源码根目录传给测试，用于定位 models/tiny/... 等数据文件
target_compile_definitions(test-Gtest PRIVATE
  DQM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_test(NAME test-Gtest COMMAND test-Gtest
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# --- Python 回归（与参考一致：CTest 调 scripts/*.py）---
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  set(DQM_SCRIPT_ENV "MINI_LLAMA_BIN=$<TARGET_FILE:dqm-llama>")
  foreach(script mvp-regression:test_mvp.py
                 golden-logits:test_golden.py
                 quant-cli-smoke:test_quant_cli.py
                 threaded-cli-smoke:test_threaded_cli.py
                 backend-cli-smoke:test_backend_cli.py
                 cli-help:test_cli_help.py)
    string(REPLACE ":" ";" pair "${script}")
    list(GET pair 0 test_name)
    list(GET pair 1 script_file)
    add_test(NAME ${test_name}
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${script_file}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(${test_name} PROPERTIES
      ENVIRONMENT "${DQM_SCRIPT_ENV}")
  endforeach()
  if(DQM_LLAMA_CUDA)
    add_test(NAME cuda-chat-smoke
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_cuda_chat_smoke.py
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(cuda-chat-smoke PROPERTIES
      ENVIRONMENT "${DQM_SCRIPT_ENV}")
  endif()
endif()

# --- 严格警告（GCC / Clang）---
# 之前一直没开，导致"各模块 0 警告"的结论无从复现（2026-09-23 补）。
if(NOT MSVC)
  target_compile_options(dqm-llama-core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Wsign-compare>)
  target_compile_options(dqm-llama PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Wsign-compare>)
  target_compile_options(test-Gtest PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Wsign-compare>)
endif()

# --- SIMD 开关 ---
# x86_64 上开 AVX2 + FMA，否则 matmul_dispatch.cc 的 __AVX2__ 分支不被编译
# （__AVX2__ / __FMA__ 是编译器按 -m 开关定义的宏，不开就整段成为死代码）。
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dqm-llama-core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>)
  target_compile_options(dqm-llama PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>)
  target_compile_options(test-Gtest PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>)
endif()
