cmake_minimum_required(VERSION 3.10)
project(QRGeneratorCUDA LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

# Ajouter des flags d'optimisation CUDA
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -use_fast_math")

add_executable(qr_generator main.cpp qr_kernel.cu)

set_target_properties(qr_generator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86")

target_include_directories(qr_generator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(qr_generator PRIVATE qrencode)

target_compile_features(qr_generator PRIVATE cxx_std_17)

# Optionnel: Pour mode Release
set(CMAKE_BUILD_TYPE Release)
