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

#define MAX_STEPS    100
#define MAX_DIST     40.0f
#define SURF_DIST    0.001f
#define SHADOW_STEPS 40
#define AO_STEPS     5

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

// ===== SDF scene kernel =====

__global__ void moving_objects_kernel(
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

extern "C" void launch_moving_objects(
  unsigned char* d_rgb,
  int width, int height, float time,
  cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  moving_objects_kernel<<<grid, block, 0, stream>>>(d_rgb, width, height, time);
}

// ===== Tunnel rings kernel =====

#define T_RADIUS      0.4f
#define T_GLOW_W      0.006f
#define T_CAM_SPEED   2.0f
#define T_PAL_SPEED   0.3f
#define T_RING_COUNT  16
#define T_WALL_STEPS  24
#define T_FOG_DENSITY 0.04f

__device__ void t_palette(float t, float &r, float &g, float &b)
{
  const float pi = 3.14159265f;
  r = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.0f));
  g = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.33f));
  b = 0.5f + 0.5f * __cosf(2.0f * pi * (t + 0.66f));
}

__device__ float t_hash(float n)
{
  return fmodf(sinf(n) * 43758.5453f, 1.0f);
  // fmod result can be negative; clamp to [0,1]
}

__device__ float t_noise(float z)
{
  float i = floorf(z);
  float f = z - i;
  f = f * f * (3.0f - 2.0f * f);
  return fabsf(t_hash(i)) * (1.0f - f) + fabsf(t_hash(i + 1.0f)) * f;
}

__global__ void tunnel_rings_kernel(
  unsigned char* __restrict__ rgb,
  int width, int height, float time)
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

  float z_cam = time * T_CAM_SPEED;

  float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;

  if (r_pixel < 0.0001f) r_pixel = 0.0001f;
  float depth = T_RADIUS / r_pixel;
  float z_world = depth + z_cam;
  int ring_z_base = (int)floorf(z_world) - T_RING_COUNT / 2;

  for (int i = 0; i < T_RING_COUNT; i++) {
    float rz = (float)(ring_z_base + i);
    float ring_depth = rz - z_cam;
    if (ring_depth <= 0.05f) continue;

    float r_ring = T_RADIUS / ring_depth;
    float dist = fabsf(r_pixel - r_ring);

    float glow_w = T_GLOW_W * (1.0f + 0.15f * sinf(rz * 2.7f + time * 1.5f));
    float glow = expf(-dist * dist / (2.0f * glow_w * glow_w));

    float ring_t = rz * 0.08f + time * T_PAL_SPEED;
    float cr, cg, cb;
    t_palette(ring_t, cr, cg, cb);

    float fade = 1.0f / (1.0f + ring_depth * ring_depth * 0.02f);
    float pulse = 0.7f + 0.3f * sinf(rz * 1.3f + time * 3.0f);

    acc_r += cr * glow * fade * pulse;
    acc_g += cg * glow * fade * pulse;
    acc_b += cb * glow * fade * pulse;
  }

  // Wall illumination: march along the tunnel depth, accumulate scattered ring light
  float angle = atan2f(v, u);
  for (int s = 0; s < T_WALL_STEPS; s++) {
    float sd = 0.3f + (float)s * 0.4f;
    float sz = sd + z_cam;
    float wall_r = T_RADIUS / sd;

    float wall_tex = 0.03f + 0.02f * t_noise(sz * 4.0f + angle * 3.0f);

    int nearest_ring = (int)floorf(sz + 0.5f);
    float ring_dist = fabsf(sz - (float)nearest_ring);
    float ring_light = expf(-ring_dist * ring_dist * 8.0f);

    float rt = (float)nearest_ring * 0.08f + time * T_PAL_SPEED;
    float wr, wg, wb;
    t_palette(rt, wr, wg, wb);

    float prox = expf(-fabsf(r_pixel - wall_r) * 120.0f);
    float depth_fade = expf(-sd * T_FOG_DENSITY * 3.0f);

    float contrib = wall_tex * ring_light * prox * depth_fade;
    acc_r += wr * contrib;
    acc_g += wg * contrib;
    acc_b += wb * contrib;
  }

  // Depth fog
  float fog = expf(-depth * T_FOG_DENSITY);
  acc_r *= fog;
  acc_g *= fog;
  acc_b *= fog;

  // Vignette (soft, only near the outer edge)
  float vig = 1.0f - fmaxf(0.0f, r_pixel - 0.6f) * 1.5f;
  vig = fmaxf(0.0f, vig);
  acc_r *= vig;
  acc_g *= vig;
  acc_b *= vig;

  // Gamma (no tone mapping -- keep the punchy contrast)
  acc_r = powf(fminf(acc_r, 1.0f), 0.45f);
  acc_g = powf(fminf(acc_g, 1.0f), 0.45f);
  acc_b = powf(fminf(acc_b, 1.0f), 0.45f);

  int base = (y * width + x) * 3;
  rgb[base + 0] = (unsigned char)(fminf(255.0f, acc_r * 255.0f));
  rgb[base + 1] = (unsigned char)(fminf(255.0f, acc_g * 255.0f));
  rgb[base + 2] = (unsigned char)(fminf(255.0f, acc_b * 255.0f));
}

extern "C" void launch_tunnel_rings(
  unsigned char* d_rgb,
  int width, int height, float time,
  cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  tunnel_rings_kernel<<<grid, block, 0, stream>>>(d_rgb, width, height, time);
}

// ===== Frame number overlay =====

// 5x7 bitmap font for digits 0-9.  Each row is 5 bits wide (bit 4 = left).
__constant__ unsigned char FONT_5x7[10][7] = {
  {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // 0
  {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
  {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, // 2
  {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}, // 3
  {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
  {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 5
  {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 6
  {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
  {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
  {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 9
};

__device__ float rounded_rect_alpha(int px, int py, int rx, int ry, int rw, int rh, int radius)
{
  int dx = 0, dy = 0;
  if (px < rx + radius && py < ry + radius) {
    dx = rx + radius - px; dy = ry + radius - py;
  } else if (px >= rx + rw - radius && py < ry + radius) {
    dx = px - (rx + rw - radius - 1); dy = ry + radius - py;
  } else if (px < rx + radius && py >= ry + rh - radius) {
    dx = rx + radius - px; dy = py - (ry + rh - radius - 1);
  } else if (px >= rx + rw - radius && py >= ry + rh - radius) {
    dx = px - (rx + rw - radius - 1); dy = py - (ry + rh - radius - 1);
  }
  if (dx > 0 && dy > 0) {
    float dist = sqrtf((float)(dx * dx + dy * dy));
    if (dist > (float)radius) return 0.0f;
    if (dist > (float)radius - 1.5f) return 1.0f - (dist - ((float)radius - 1.5f)) / 1.5f;
  }
  if (px < rx || px >= rx + rw || py < ry || py >= ry + rh) return 0.0f;
  return 1.0f;
}

__global__ void render_number_kernel(
  unsigned char* __restrict__ rgb, int img_w, int img_h,
  unsigned int number, int num_digits, int scale,
  int box_x, int box_y, int box_w, int box_h,
  int text_x, int text_y, int corner_r)
{
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= box_w || by >= box_h) return;

  int px = box_x + bx;
  int py = box_y + by;
  if (px < 0 || px >= img_w || py < 0 || py >= img_h) return;

  float box_alpha = rounded_rect_alpha(px, py, box_x, box_y, box_w, box_h, corner_r);
  if (box_alpha <= 0.0f) return;

  int glyph_pitch = 6 * scale;
  int text_w = num_digits * glyph_pitch;
  int text_h = 7 * scale;

  int tx = px - text_x;
  int ty_local = py - text_y;

  bool lit = false;
  bool in_shadow = false;
  if (tx >= 0 && tx < text_w && ty_local >= 0 && ty_local < text_h) {
    int digit_idx = tx / glyph_pitch;
    int local_x = (tx % glyph_pitch) / scale;
    int local_y = ty_local / scale;

    unsigned int tmp = number;
    for (int i = num_digits - 1 - digit_idx; i > 0; i--) tmp /= 10;
    int d = tmp % 10;

    if (local_x < 5 && local_y < 7)
      lit = (FONT_5x7[d][local_y] >> (4 - local_x)) & 1;
  }

  // Shadow: check 1 scaled pixel down-right
  int sx = px - text_x - scale;
  int sy = py - text_y - scale;
  if (!lit && sx >= 0 && sx < text_w && sy >= 0 && sy < text_h) {
    int digit_idx = sx / glyph_pitch;
    int local_x = (sx % glyph_pitch) / scale;
    int local_y = sy / scale;

    unsigned int tmp = number;
    for (int i = num_digits - 1 - digit_idx; i > 0; i--) tmp /= 10;
    int d = tmp % 10;

    if (local_x < 5 && local_y < 7)
      in_shadow = (FONT_5x7[d][local_y] >> (4 - local_x)) & 1;
  }

  int idx = (py * img_w + px) * 3;
  unsigned char r = rgb[idx + 0];
  unsigned char g = rgb[idx + 1];
  unsigned char b = rgb[idx + 2];

  float br = r / 255.0f, bg = g / 255.0f, bb = b / 255.0f;

  // Background: dark translucent pill
  float bg_r = br * 0.18f, bg_g = bg * 0.18f, bg_b = bb * 0.18f;
  br = br * (1.0f - box_alpha * 0.82f) + bg_r * box_alpha;
  bg = bg * (1.0f - box_alpha * 0.82f) + bg_g * box_alpha;
  bb = bb * (1.0f - box_alpha * 0.82f) + bg_b * box_alpha;

  if (in_shadow) {
    br *= 0.4f; bg *= 0.4f; bb *= 0.4f;
  }

  if (lit) {
    br = 1.0f; bg = 1.0f; bb = 1.0f;
  }

  rgb[idx + 0] = (unsigned char)(fminf(br * 255.0f, 255.0f));
  rgb[idx + 1] = (unsigned char)(fminf(bg * 255.0f, 255.0f));
  rgb[idx + 2] = (unsigned char)(fminf(bb * 255.0f, 255.0f));
}

extern "C" void launch_render_frame_number(
  unsigned char* d_rgb, int width, int height,
  unsigned int frame_number, cudaStream_t stream)
{
  int num_digits = 1;
  unsigned int tmp = frame_number;
  while (tmp >= 10) { tmp /= 10; num_digits++; }

  const int scale = max(2, height / 216);
  int glyph_pitch = 6 * scale;
  int text_w = num_digits * glyph_pitch;
  int text_h = 7 * scale;
  int margin_x = scale * 3;
  int margin_y = scale * 2;
  int corner_r = scale * 2;

  int box_w = text_w + 2 * margin_x;
  int box_h = text_h + 2 * margin_y;
  int box_x = (width - box_w) / 2;
  int box_y = scale * 15;
  int text_x = box_x + margin_x;
  int text_y = box_y + margin_y;

  dim3 block(16, 16);
  dim3 grid((box_w + block.x - 1) / block.x, (box_h + block.y - 1) / block.y);
  render_number_kernel<<<grid, block, 0, stream>>>(
    d_rgb, width, height, frame_number, num_digits, scale,
    box_x, box_y, box_w, box_h, text_x, text_y, corner_r);
}
