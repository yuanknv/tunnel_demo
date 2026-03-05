// Copyright 2024 NVIDIA Corporation
// Licensed under the Apache License, Version 2.0

// Animated tunnel effect rendered entirely on the GPU.
// The tunnel walls have a spinning checkerboard pattern with color cycling,
// making frame-rate differences immediately visible: at high FPS (CUDA
// transport) the rotation is smooth, while at low FPS (CPU transport) the
// stutter is obvious.

#include <cuda_runtime.h>
#include <cmath>

#define TUNNEL_RADIUS  0.45f
#define CAMERA_SPEED   4.0f
#define ROTATION_SPEED 1.5f
#define CHECKER_FREQ   8.0f
#define GLOW_WIDTH     0.012f
#define NUM_RINGS      6

__device__ void palette(float t, float* r, float* g, float* b)
{
  const float pi = 3.14159265f;
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

  const float pi = 3.14159265f;
  float aspect = (float)width / (float)height;
  float u = (2.0f * (x + 0.5f) / width  - 1.0f) * aspect;
  float v =  2.0f * (y + 0.5f) / height - 1.0f;
  float r_pixel = sqrtf(u * u + v * v);

  if (r_pixel < 0.001f) r_pixel = 0.001f;

  float angle = atan2f(v, u);
  float z_cam = time * CAMERA_SPEED;
  float depth = TUNNEL_RADIUS / r_pixel;
  float z_world = depth + z_cam;

  // Rotating checkerboard on the tunnel walls
  float rot_angle = angle + time * ROTATION_SPEED;
  float tile_u = rot_angle * CHECKER_FREQ / pi;
  float tile_v = z_world * 2.0f;
  int check = ((int)floorf(tile_u) ^ (int)floorf(tile_v)) & 1;
  float checker_val = check ? 0.7f : 0.3f;

  // Base wall color from depth + rotation
  float cr, cg, cb;
  palette(z_world * 0.08f + time * 0.5f, &cr, &cg, &cb);
  float fog = 1.0f / (1.0f + depth * 0.15f);
  float wall_r = cr * checker_val * fog;
  float wall_g = cg * checker_val * fog;
  float wall_b = cb * checker_val * fog;

  // Glowing ring overlays
  float ring_r = 0.0f, ring_g = 0.0f, ring_b = 0.0f;
  int ring_base = (int)floorf(z_world);
  for (int i = 0; i < NUM_RINGS; i++) {
    float rz = (float)(ring_base + i);
    float ring_depth = rz - z_cam;
    if (ring_depth <= 0.01f) continue;
    float r_ring = TUNNEL_RADIUS / ring_depth;
    float dist = fabsf(r_pixel - r_ring);
    float glow = expf(-dist * dist / (2.0f * GLOW_WIDTH * GLOW_WIDTH));
    float rcr, rcg, rcb;
    palette(rz * 0.15f + time * 0.8f, &rcr, &rcg, &rcb);
    float fade = 1.0f / (1.0f + ring_depth * 0.08f);
    float intensity = glow * fade * 1.5f;
    ring_r += rcr * intensity;
    ring_g += rcg * intensity;
    ring_b += rcb * intensity;
  }

  float out_r = wall_r + ring_r;
  float out_g = wall_g + ring_g;
  float out_b = wall_b + ring_b;

  // Vignette darkening at edges
  float vig = 1.0f - 0.4f * r_pixel * r_pixel;
  out_r *= vig;
  out_g *= vig;
  out_b *= vig;

  int base = (y * width + x) * 3;
  rgb[base + 0] = (unsigned char)fminf(255.0f, out_r * 255.0f);
  rgb[base + 1] = (unsigned char)fminf(255.0f, out_g * 255.0f);
  rgb[base + 2] = (unsigned char)fminf(255.0f, out_b * 255.0f);
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
