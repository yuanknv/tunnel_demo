cmake_minimum_required(VERSION 3.8)
project(tunnel_demo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(torch_buffer REQUIRED)
find_package(torch_buffer_backend REQUIRED)
find_package(cuda_buffer_backend REQUIRED)
find_package(cuda_buffer REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA nvcc setup (before find_package(Torch) which enables CUDA internally)
find_program(CUDA_NVCC_EXECUTABLE nvcc
  PATHS /usr/local/cuda/bin ENV CUDA_PATH
)
if(CUDA_NVCC_EXECUTABLE AND NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

set(LIBTORCH_ROOT "$ENV{PIXI_PROJECT_ROOT}/.pixi/envs/default/libtorch")
if(EXISTS "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
  list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
else()
  execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PYTHON_RESULT
  )
  if(TORCH_PYTHON_RESULT EQUAL 0 AND TORCH_CMAKE_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
  endif()
endif()
find_package(Torch REQUIRED)

# Compile CUDA kernel via nvcc custom command
set(TUNNEL_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/tunnel_kernel.cu)
set(TUNNEL_KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/tunnel_kernel.o)
add_custom_command(
  OUTPUT ${TUNNEL_KERNEL_OBJ}
  COMMAND ${CMAKE_CUDA_COMPILER}
    -c ${TUNNEL_KERNEL_SRC}
    -o ${TUNNEL_KERNEL_OBJ}
    --compiler-bindir ${CMAKE_CXX_COMPILER}
    -gencode arch=compute_80,code=sm_80
    -gencode arch=compute_86,code=sm_86
    -gencode arch=compute_89,code=sm_89
    -gencode arch=compute_90,code=sm_90
    -Xcompiler -fPIC
  DEPENDS ${TUNNEL_KERNEL_SRC}
)
add_custom_target(tunnel_kernel_obj DEPENDS ${TUNNEL_KERNEL_OBJ})

find_package(std_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(launch_testing_ament_cmake REQUIRED)
find_package(OpenGL REQUIRED)
find_library(GLFW_LIBRARY glfw PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)

# Renderer node (publisher)
add_library(tunnel_renderer_component SHARED src/tunnel_renderer_node.cpp)
add_dependencies(tunnel_renderer_component tunnel_kernel_obj)
target_include_directories(tunnel_renderer_component PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(tunnel_renderer_component
  ${TUNNEL_KERNEL_OBJ}
  ${TORCH_LIBRARIES}
  torch_buffer::torch_buffer
  torch_buffer_backend::torch_buffer_backend
  cuda_buffer::cuda_buffer
  rclcpp::rclcpp
  rclcpp_components::component
  ${sensor_msgs_TARGETS}
  CUDA::cudart
  CUDA::cuda_driver
)
rclcpp_components_register_node(tunnel_renderer_component
  PLUGIN "TunnelRenderer"
  EXECUTABLE tunnel_renderer_node
)
install(TARGETS tunnel_renderer_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Display node (subscriber, OpenGL + headless fallback)
add_library(tunnel_display_component SHARED src/tunnel_display_node.cpp)
target_link_libraries(tunnel_display_component
  ${TORCH_LIBRARIES}
  torch_buffer::torch_buffer
  torch_buffer_backend::torch_buffer_backend
  cuda_buffer::cuda_buffer
  rclcpp::rclcpp
  rclcpp_components::component
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  CUDA::cudart
  CUDA::cuda_driver
  OpenGL::GL
  ${GLFW_LIBRARY}
)
rclcpp_components_register_node(tunnel_display_component
  PLUGIN "TunnelDisplay"
  EXECUTABLE tunnel_display_node
)
install(TARGETS tunnel_display_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Launch test
add_launch_test(test/test_tunnel_demo_launch.py
  TARGET test_tunnel_demo_launch
  TIMEOUT 60
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
