// Copyright 2024 NVIDIA Corporation
// Licensed under the Apache License, Version 2.0

// Raymarched SDF scene rendered entirely on the GPU.
// An orbiting camera circles animated floating shapes above a
// checkerboard ground plane. The scene is GPU-heavy (~8-16ms per
// frame) so CUDA transport lands at 60-120 FPS while CPU transport
// drops to 15-30 FPS, making the smoothness difference obvious
// on a standard monitor.

#include <cuda_runtime.h>
#include <cmath>

#define MAX_STEPS    150
#define MAX_DIST     40.0f
#define SURF_DIST    0.001f
#define SHADOW_STEPS 64
#define AO_STEPS     8

#define ORBIT_RADIUS 6.0f
#define ORBIT_SPEED  0.5f
#define CAM_HEIGHT   3.0f

// ===== Vector helpers =====

struct float3_t { float x, y, z; };

__device__ float3_t f3(float x, float y, float z) { return {x, y, z}; }
__device__ float3_t operator+(float3_t a, float3_t b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ float3_t operator-(float3_t a, float3_t b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ float3_t operator*(float3_t a, float s) { return {a.x*s, a.y*s, a.z*s}; }
__device__ float3_t operator*(float s, float3_t a) { return a * s; }
__device__ float dot(float3_t a, float3_t b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ float length(float3_t v) { return sqrtf(dot(v, v)); }
__device__ float3_t normalize(float3_t v) { float l = length(v); return {v.x/l, v.y/l, v.z/l}; }
__device__ float3_t cross(float3_t a, float3_t b) {
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__device__ float3_t reflect(float3_t i, float3_t n) { return i - 2.0f * dot(i, n) * n; }
__device__ float clampf(float x, float lo, float hi) { return fminf(hi, fmaxf(lo, x)); }

// ===== SDF primitives =====

__device__ float sdSphere(float3_t p, float r) { return length(p) - r; }

__device__ float sdBox(float3_t p, float3_t b) {
  float3_t q = {fabsf(p.x)-b.x, fabsf(p.y)-b.y, fabsf(p.z)-b.z};
  float3_t qp = {fmaxf(q.x,0.0f), fmaxf(q.y,0.0f), fmaxf(q.z,0.0f)};
  return length(qp) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
}

__device__ float sdTorus(float3_t p, float R, float r) {
  float qx = sqrtf(p.x*p.x + p.z*p.z) - R;
  return sqrtf(qx*qx + p.y*p.y) - r;
}

__device__ float sdPlane(float3_t p) { return p.y; }

__device__ float smin(float a, float b, float k) {
  float h = clampf(0.5f + 0.5f*(b-a)/k, 0.0f, 1.0f);
  return b*(1.0f-h) + a*h - k*h*(1.0f-h);
}

// ===== Scene =====

struct HitInfo { float dist; int mat_id; };

__device__ HitInfo scene(float3_t p, float time)
{
  float ground = sdPlane(p);

  // Animated spheres orbiting the center
  float spheres = MAX_DIST;
  for (int i = 0; i < 5; i++) {
    float angle = time * 0.8f + (float)i * 1.2566f;
    float r = 2.0f + 0.5f * sinf(time * 0.5f + (float)i);
    float3_t center = {r * cosf(angle), 1.2f + 0.6f * sinf(time * 1.2f + (float)i * 0.7f), r * sinf(angle)};
    float s = sdSphere(p - center, 0.45f + 0.1f * sinf(time * 2.0f + (float)i));
    spheres = smin(spheres, s, 0.3f);
  }

  // Rotating torus at center
  float ca = cosf(time * 0.6f), sa = sinf(time * 0.6f);
  float3_t tp = {p.x * ca + p.z * sa, p.y - 2.0f, -p.x * sa + p.z * ca};
  float cb = cosf(time * 0.3f), sb = sinf(time * 0.3f);
  float3_t tp2 = {tp.x, tp.y * cb - tp.z * sb, tp.y * sb + tp.z * cb};
  float torus = sdTorus(tp2, 1.0f, 0.3f);

  // Bouncing box
  float3_t bp = p - f3(0.0f, 0.8f + 0.5f * fabsf(sinf(time * 1.5f)), 0.0f);
  float cbox = cosf(time * 1.0f), sbox = sinf(time * 1.0f);
  float3_t rbp = {bp.x * cbox + bp.z * sbox, bp.y, -bp.x * sbox + bp.z * cbox};
  float box = sdBox(rbp, f3(0.5f, 0.5f, 0.5f));

  // Combine objects
  float objects = smin(spheres, torus, 0.2f);
  objects = smin(objects, box, 0.2f);

  if (ground < objects) {
    return {ground, 0};
  }
  return {objects, 1};
}

__device__ float3_t calcNormal(float3_t p, float time)
{
  const float e = 0.001f;
  float d = scene(p, time).dist;
  return normalize(f3(
    scene(p + f3(e,0,0), time).dist - d,
    scene(p + f3(0,e,0), time).dist - d,
    scene(p + f3(0,0,e), time).dist - d));
}

__device__ float softShadow(float3_t ro, float3_t rd, float mint, float maxt, float time)
{
  float res = 1.0f;
  float t = mint;
  for (int i = 0; i < SHADOW_STEPS && t < maxt; i++) {
    float h = scene(ro + rd * t, time).dist;
    res = fminf(res, 8.0f * h / t);
    if (h < 0.001f) return 0.0f;
    t += clampf(h, 0.02f, 0.2f);
  }
  return clampf(res, 0.0f, 1.0f);
}

__device__ float ambientOcclusion(float3_t p, float3_t n, float time)
{
  float occ = 0.0f;
  float scale = 1.0f;
  for (int i = 0; i < AO_STEPS; i++) {
    float h = 0.01f + 0.12f * (float)i;
    float d = scene(p + n * h, time).dist;
    occ += (h - d) * scale;
    scale *= 0.7f;
  }
  return clampf(1.0f - 1.5f * occ, 0.0f, 1.0f);
}

// ===== Main kernel =====

__global__ void tunnel_kernel(
  unsigned char* __restrict__ rgb,
  int width, int height, float time)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float aspect = (float)width / (float)height;
  float u = (2.0f * (x + 0.5f) / width  - 1.0f) * aspect;
  float v = 1.0f - 2.0f * (y + 0.5f) / height;

  // Orbiting camera
  float3_t eye = {
    ORBIT_RADIUS * cosf(time * ORBIT_SPEED),
    CAM_HEIGHT + 0.5f * sinf(time * 0.3f),
    ORBIT_RADIUS * sinf(time * ORBIT_SPEED)};
  float3_t target = f3(0.0f, 1.0f, 0.0f);
  float3_t fwd = normalize(target - eye);
  float3_t right = normalize(cross(fwd, f3(0,1,0)));
  float3_t up = cross(right, fwd);
  float3_t rd = normalize(fwd * 1.5f + right * u + up * v);

  // Raymarch
  float t = 0.0f;
  int mat = -1;
  for (int i = 0; i < MAX_STEPS; i++) {
    float3_t p = eye + rd * t;
    HitInfo h = scene(p, time);
    if (h.dist < SURF_DIST) { mat = h.mat_id; break; }
    if (t > MAX_DIST) break;
    t += h.dist;
  }

  float3_t col;
  if (mat >= 0) {
    float3_t p = eye + rd * t;
    float3_t n = calcNormal(p, time);
    float3_t lightDir = normalize(f3(0.6f, 0.8f, -0.4f));

    // Material color
    float3_t albedo;
    if (mat == 0) {
      // Checkerboard ground
      int cx = (int)floorf(p.x * 0.5f);
      int cz = (int)floorf(p.z * 0.5f);
      float check = ((cx ^ cz) & 1) ? 0.6f : 0.2f;
      albedo = f3(check, check, check * 1.1f);
    } else {
      float pi = 3.14159265f;
      float hue = atan2f(p.z, p.x) / (2.0f * pi) + 0.5f + time * 0.05f;
      albedo = f3(
        0.5f + 0.5f * cosf(2.0f * pi * hue),
        0.5f + 0.5f * cosf(2.0f * pi * (hue + 0.33f)),
        0.5f + 0.5f * cosf(2.0f * pi * (hue + 0.67f)));
    }

    float diff = clampf(dot(n, lightDir), 0.0f, 1.0f);
    float shadow = softShadow(p + n * 0.01f, lightDir, 0.02f, 10.0f, time);
    float ao = ambientOcclusion(p, n, time);

    // Specular highlight
    float3_t refl = reflect(rd, n);
    float spec = powf(clampf(dot(refl, lightDir), 0.0f, 1.0f), 16.0f);

    float3_t ambient = albedo * 0.15f;
    col = ambient + albedo * diff * shadow * 0.7f + f3(1,1,1) * spec * shadow * 0.3f;
    col = col * ao;

    // Distance fog
    float fog = expf(-t * 0.04f);
    float3_t fogCol = f3(0.05f, 0.05f, 0.1f);
    col = col * fog + fogCol * (1.0f - fog);
  } else {
    // Sky gradient
    float sky = 0.5f + 0.5f * v;
    col = f3(0.02f + 0.03f * sky, 0.02f + 0.04f * sky, 0.05f + 0.08f * sky);
  }

  // Gamma correction
  col = f3(powf(clampf(col.x, 0.0f, 1.0f), 0.4545f),
           powf(clampf(col.y, 0.0f, 1.0f), 0.4545f),
           powf(clampf(col.z, 0.0f, 1.0f), 0.4545f));

  int base = (y * width + x) * 3;
  rgb[base + 0] = (unsigned char)(col.x * 255.0f);
  rgb[base + 1] = (unsigned char)(col.y * 255.0f);
  rgb[base + 2] = (unsigned char)(col.z * 255.0f);
}

extern "C" void launch_tunnel_effect(
  unsigned char* d_rgb,
  int width, int height, float time,
  cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  tunnel_kernel<<<grid, block, 0, stream>>>(d_rgb, width, height, time);
}
