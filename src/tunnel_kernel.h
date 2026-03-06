#pragma once
#include <cuda_runtime.h>

extern "C" void launch_moving_objects(
  unsigned char* d_rgb, int width, int height,
  float time, cudaStream_t stream = nullptr);

extern "C" void launch_tunnel_rings(
  unsigned char* d_rgb, int width, int height,
  float time, cudaStream_t stream = nullptr);

extern "C" void launch_render_frame_number(
  unsigned char* d_rgb, int width, int height,
  unsigned int frame_number,
  cudaStream_t stream = nullptr);
