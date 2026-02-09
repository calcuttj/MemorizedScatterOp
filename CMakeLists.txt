cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(NewScatterOp)

find_package(Torch REQUIRED)

# Define our library target
add_library(my_custom_ops SHARED MyScatterNode.cpp)
target_include_directories(
    my_custom_ops
        PRIVATE "/usr/include/python3.11/"
        PRIVATE "$ENV{TORCHSCATTER_INCLUDE}"
)

set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch & scatter
target_link_libraries(
    my_custom_ops 
    "${TORCH_LIBRARIES}"
    "$ENV{TORCHSCATTER_LIBRARY}"
)
