cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(kda_finalize_catlass LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
    message(FATAL_ERROR "Set CATLASS_ROOT to the pinned CATLASS source directory")
endif()
set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
set_source_files_properties(op_host/stage0.cpp op_host/finalize.cpp PROPERTIES LANGUAGE ASC)
add_executable(finalize_stage0 op_host/stage0.cpp)
add_executable(finalize op_host/finalize.cpp)
foreach(target finalize_stage0 finalize)
target_include_directories(${target} PRIVATE
    "${CATLASS_ROOT}/include" "${CANN_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
target_compile_definitions(${target} PRIVATE CATLASS_ARCH=2201 _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(${target} PRIVATE
    --npu-arch=dav-2201 -Wno-macro-redefined -Wno-ignored-attributes -Wno-unused-value
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")
target_link_directories(${target} PRIVATE
    "${CANN_ROOT}/lib64" "${CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib")
target_link_libraries(${target} PRIVATE
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
    tiling_api nnopbase platform unified_dlog ascend_hal dl)
endforeach()
