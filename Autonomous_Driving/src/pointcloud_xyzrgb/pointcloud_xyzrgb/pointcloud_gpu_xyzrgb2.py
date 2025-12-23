# pointcloud_gpu_xyzrgb.py
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# --- 전역 캐시 ---
_cached_num_pixels = 0
_depth_gpu = None
_rgb_gpu = None
_cloud_gpu = None  # [x,y,z,rgb] float32 4개씩


kernel_code = r"""
#include <math.h>

extern "C"
__global__ void depth_rgb_to_xyzrgb_kernel(
    const float* depth,             // num_pixels
    const unsigned char* rgb,       // num_pixels * rgb_step
    float* out,                     // num_pixels * 4 (x,y,z,rgb)
    int width,
    int height,
    float fx, float fy,
    float cx, float cy,
    int rgb_step,                   // bytes per pixel (1,3,4)
    int red_offset,
    int green_offset,
    int blue_offset,
    float range_max
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = width * height;
    if (idx >= num_pixels) return;

    float d = depth[idx];
    float X, Y, Z;

    // 범위 체크
    if (d <= 0.0f || (range_max > 0.0f && d > range_max)) {
        X = NAN;
        Y = NAN;
        Z = NAN;
    } else {
        int u = idx % width;
        int v = idx / width;
        Z = d;
        X = ((float)u - cx) * d / fx;
        Y = ((float)v - cy) * d / fy;
    }

    int rgb_idx = idx * rgb_step;
    unsigned char r = rgb[rgb_idx + red_offset];
    unsigned char g = rgb[rgb_idx + green_offset];
    unsigned char b = rgb[rgb_idx + blue_offset];

    unsigned int rgb_packed =
        ((unsigned int)r << 16) |
        ((unsigned int)g << 8)  |
        (unsigned int)b;

    float rgb_f = __int_as_float(rgb_packed);

    int base = idx * 4;
    out[base + 0] = X;
    out[base + 1] = Y;
    out[base + 2] = Z;
    out[base + 3] = rgb_f;
}
"""

_mod = SourceModule(kernel_code)
_depth_rgb_to_xyzrgb_kernel = _mod.get_function("depth_rgb_to_xyzrgb_kernel")


def _encoding_to_offsets(encoding: str, channels: int):
    enc = encoding.lower()
    # 기본값
    red_offset = 0
    green_offset = 1
    blue_offset = 2
    rgb_step = channels

    if enc == "rgb8":
        red_offset, green_offset, blue_offset, rgb_step = 0, 1, 2, 3
    elif enc == "rgba8":
        red_offset, green_offset, blue_offset, rgb_step = 0, 1, 2, 4
    elif enc == "bgr8":
        red_offset, green_offset, blue_offset, rgb_step = 2, 1, 0, 3
    elif enc == "bgra8":
        red_offset, green_offset, blue_offset, rgb_step = 2, 1, 0, 4
    elif enc == "mono8":
        red_offset = green_offset = blue_offset = 0
        rgb_step = 1
    else:
        # 예상 못한 인코딩이면 RGB8로 강제 가정
        red_offset, green_offset, blue_offset, rgb_step = 0, 1, 2, channels

    return rgb_step, red_offset, green_offset, blue_offset


def depth_rgb_to_xyzrgb_gpu(
    depth_np: np.ndarray,
    rgb_np: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    encoding: str = "rgb8",
    range_max: float = 0.0,
) -> np.ndarray:
    """
    GPU에서 바로 [x,y,z,rgb] float32 배열로 만들어서 넘겨주는 함수.

    반환:
        cloud4_np: (H, W, 4) float32
            [...,0] = x
            [...,1] = y
            [...,2] = z
            [...,3] = rgb(packed as float)
    """
    assert depth_np.ndim == 2
    H, W = depth_np.shape
    assert rgb_np.shape[0] == H and rgb_np.shape[1] == W

    # depth: (H,W) float32
    depth_flat = np.ascontiguousarray(depth_np, dtype=np.float32).reshape(-1)

    # rgb: (H,W,C) uint8
    if rgb_np.ndim == 2:
        rgb_np = rgb_np[:, :, None]
    C = rgb_np.shape[2]
    rgb_flat = np.ascontiguousarray(rgb_np, dtype=np.uint8).reshape(-1)

    rgb_step, red_offset, green_offset, blue_offset = _encoding_to_offsets(encoding, C)

    num_pixels = H * W

    global _cached_num_pixels, _depth_gpu, _rgb_gpu, _cloud_gpu

    # GPU 메모리 (픽셀 수 바뀌면 재할당)
    if _cached_num_pixels != num_pixels or _depth_gpu is None:
        _cached_num_pixels = num_pixels
        _depth_gpu = cuda.mem_alloc(depth_flat.nbytes)
        _rgb_gpu = cuda.mem_alloc(rgb_flat.nbytes)
        _cloud_gpu = cuda.mem_alloc(num_pixels * 4 * np.float32().nbytes)

    cuda.memcpy_htod(_depth_gpu, depth_flat)
    cuda.memcpy_htod(_rgb_gpu, rgb_flat)

    threads_per_block = 256
    blocks = (num_pixels + threads_per_block - 1) // threads_per_block

    _depth_rgb_to_xyzrgb_kernel(
        _depth_gpu,
        _rgb_gpu,
        _cloud_gpu,
        np.int32(W),
        np.int32(H),
        np.float32(fx),
        np.float32(fy),
        np.float32(cx),
        np.float32(cy),
        np.int32(rgb_step),
        np.int32(red_offset),
        np.int32(green_offset),
        np.int32(blue_offset),
        np.float32(range_max),
        block=(threads_per_block, 1, 1),
        grid=(blocks, 1, 1),
    )

    # Host 쪽 버퍼 한 번에 복사
    cloud4_flat = np.empty(num_pixels * 4, dtype=np.float32)
    cuda.memcpy_dtoh(cloud4_flat, _cloud_gpu)

    cloud4_np = cloud4_flat.reshape(H, W, 4)
    return cloud4_np
