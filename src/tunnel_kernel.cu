// Copyright 2024 NVIDIA Corporation
// Licensed under the Apache License, Version 2.0

// Animated neon tunnel effect rendered entirely on the GPU.
// Thin glowing rings in blue/cyan/magenta flow steadily toward the camera.

#include <cuda_runtime.h>
#include <cmath>

#define TUNNEL_RADIUS  0.45f
#define CAMERA_SPEED   0.25f
#define RING_SPACING   0.5f
#define NUM_RINGS      8
#define COLOR_SPEED    0.1f

__device__ void neon_palette(float t, float* r, float* g, float* b)
{
  const float pi = 3.14159265f;

  // Equal-weight RGB cosine palette with 120-degree phase offsets
  // Cycles evenly: red -> yellow -> green -> cyan -> blue -> magenta -> red
  *r = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.00f));
  *g = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.33f));
  *b = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.67f));
}

__global__ void tunnel_kernel(
  unsigned char* __restrict__ rgb,
  int width,
  int height,
  float time)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float aspect = (float)width / (float)height;
  float u = (2.0f * (x + 0.5f) / width  - 1.0f) * aspect;
  float v =  2.0f * (y + 0.5f) / height - 1.0f;
  float r_pixel = sqrtf(u * u + v * v);
  if (r_pixel < 0.001f) r_pixel = 0.001f;

  float z_cam = time * CAMERA_SPEED;
  float depth = TUNNEL_RADIUS / r_pixel;
  float z_world = depth + z_cam;

  float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
  float ring_phase = z_world / RING_SPACING;
  int ring_base = (int)floorf(ring_phase);

  for (int i = -1; i < NUM_RINGS; i++) {
    float rz = (float)(ring_base + i) * RING_SPACING;
    float ring_depth = rz - z_cam;
    if (ring_depth <= 0.05f) continue;

    float r_ring = TUNNEL_RADIUS / ring_depth;
    float dist = fabsf(r_pixel - r_ring);

    float glow_w = 0.010f;
    float core = expf(-dist * dist / (2.0f * glow_w * glow_w));
    float bloom = 0.25f * expf(-dist * dist / (12.0f * glow_w * glow_w));
    float glow = core + bloom;

    float cr, cg, cb;
    neon_palette(rz * 0.15f + time * COLOR_SPEED, &cr, &cg, &cb);

    float fade = 1.0f / (1.0f + ring_depth * ring_depth * 0.015f);
    float intensity = glow * fade;

    acc_r += cr * intensity;
    acc_g += cg * intensity;
    acc_b += cb * intensity;
  }

  // Subtle center glow
  float center_glow = 0.25f * expf(-r_pixel * r_pixel / 0.06f);
  acc_r += center_glow * 0.4f;
  acc_g += center_glow * 0.5f;
  acc_b += center_glow * 0.7f;

  // Clamp
  int base = (y * width + x) * 3;
  rgb[base + 0] = (unsigned char)fminf(255.0f, acc_r * 255.0f);
  rgb[base + 1] = (unsigned char)fminf(255.0f, acc_g * 255.0f);
  rgb[base + 2] = (unsigned char)fminf(255.0f, acc_b * 255.0f);
}

extern "C" void launch_tunnel_effect(
  unsigned char* d_rgb,
  int width,
  int height,
  float time,
  cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  tunnel_kernel<<<grid, block, 0, stream>>>(d_rgb, width, height, time);
}
