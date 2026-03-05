#pragma once
#include <cuda_runtime.h>

extern "C" void launch_tunnel_effect(
  unsigned char* d_rgb,
  int width,
  int height,
  float time,
  cudaStream_t stream = nullptr);
