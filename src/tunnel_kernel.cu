#include <cuda_runtime.h>
#include <cmath>

#define TUNNEL_RADIUS 0.4f
#define RING_GLOW_WIDTH 0.015f
#define CAMERA_SPEED 2.0f
#define PALETTE_SPEED 0.3f

__device__ void palette(float t, float* r, float* g, float* b)
{
  const float pi = 3.14159265f;
  *r = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.0f));
  *g = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.33f));
  *b = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.66f));
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
  float u = (2.0f * (x + 0.5f) / width - 1.0f) * aspect;
  float v = 2.0f * (y + 0.5f) / height - 1.0f;
  float r_pixel = sqrtf(u * u + v * v);
  if (r_pixel > 1.0f) {
    int base = (y * width + x) * 3;
    rgb[base + 0] = 0;
    rgb[base + 1] = 0;
    rgb[base + 2] = 0;
    return;
  }

  float z_cam = time * CAMERA_SPEED;
  float depth = TUNNEL_RADIUS / fmaxf(r_pixel, 0.0001f);
  float z_world = depth + z_cam;

  float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
  int ring_z_base = (int)floorf(z_world);
  for (int i = 0; i < 4; i++) {
    float rz = (float)(ring_z_base + i);
    float ring_depth = rz - z_cam;
    if (ring_depth <= 0.01f) continue;
    float r_ring = TUNNEL_RADIUS / ring_depth;
    float dist = fabsf(r_pixel - r_ring);
    float glow = expf(-dist * dist / (2.0f * RING_GLOW_WIDTH * RING_GLOW_WIDTH));
    float ring_t = rz * 0.1f + time * PALETTE_SPEED;
    float cr, cg, cb;
    palette(ring_t, &cr, &cg, &cb);
    float fade = 1.0f / (1.0f + ring_depth * 0.1f);
    acc_r += cr * glow * fade;
    acc_g += cg * glow * fade;
    acc_b += cb * glow * fade;
  }

  float m = fmaxf(acc_r, fmaxf(acc_g, acc_b));
  if (m > 1.0f) {
    acc_r /= m;
    acc_g /= m;
    acc_b /= m;
  }
  int base = (y * width + x) * 3;
  rgb[base + 0] = (unsigned char)(fminf(255.0f, acc_r * 255.0f));
  rgb[base + 1] = (unsigned char)(fminf(255.0f, acc_g * 255.0f));
  rgb[base + 2] = (unsigned char)(fminf(255.0f, acc_b * 255.0f));
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
