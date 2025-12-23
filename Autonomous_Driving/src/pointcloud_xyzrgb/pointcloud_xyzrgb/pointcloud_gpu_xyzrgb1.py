# pointcloud_gpu_xyzrgb.py
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# --- 전역 캐시 ---
_cached_num_pixels = 0
_depth_gpu = None
_rgb_gpu = None
_xyz_gpu = None
_rgb_packed_gpu = None


kernel_code = r"""
extern "C"
__global__ void depth_rgb_to_xyzrgb_kernel(
    const float *depth,
    const unsigned char *rgb,
    float *xyz_out,
    float *rgb_packed_out,
    int width,
    int height,
    float fx,
    float fy,
    float cx,
    float cy,
    int rgb_step,
    int red_offset,
    int green_offset,
    int blue_offset,
    float range_max
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = width * height;

    if (idx >= num_pixels)
        return;

    int u = idx % width;
    int v = idx / width;

    float d = depth[idx];

    if (!(d > 0.0f) || (range_max > 0.0f && d > range_max)) {
        xyz_out[3 * idx + 0] = NAN;
        xyz_out[3 * idx + 1] = NAN;
        xyz_out[3 * idx + 2] = NAN;
        rgb_packed_out[idx] = __int_as_float(0u);
        return;
    }

    float X = ((float)u - cx) * d / fx;
    float Y = ((float)v - cy) * d / fy;
    float Z = d;

    xyz_out[3 * idx + 0] = X;
    xyz_out[3 * idx + 1] = Y;
    xyz_out[3 * idx + 2] = Z;

    int rgb_index = idx * rgb_step;
    unsigned char r = rgb[rgb_index + red_offset];
    unsigned char g = rgb[rgb_index + green_offset];
    unsigned char b = rgb[rgb_index + blue_offset];

    unsigned int rgb_uint = ((unsigned int)r << 16) |
                            ((unsigned int)g << 8)  |
                            ((unsigned int)b);

    rgb_packed_out[idx] = __int_as_float(rgb_uint);
}
"""

mod = SourceModule(kernel_code)
depth_rgb_to_xyzrgb_kernel = mod.get_function("depth_rgb_to_xyzrgb_kernel")


def depth_rgb_to_xyzrgb_gpu(depth_np, rgb_np, fx, fy, cx, cy,
                            encoding="rgb8", range_max=0.0):

    """
    depth_np: (H, W) float32, meter 단위
    rgb_np:   (H, W, C) uint8, C=3 또는 4
    encoding: "rgb8", "bgr8", "rgba8", "bgra8", "mono8" 등
              (C++ point_cloud_xyzrgb와 동일한 의미로 맞추면 됨)
    반환: xyz_np (H, W, 3, float32), rgb_packed_np (H, W, float32)
    """

    assert depth_np.dtype == np.float32
    assert rgb_np.dtype == np.uint8
    H, W = depth_np.shape
    Hr, Wr = rgb_np.shape[:2]
    assert (H, W) == (Hr, Wr), "depth/rgb size mismatch"
    num_pixels = H * W

    # 1) encoding → rgb_step, offsets 매핑
    if encoding.lower() in ["rgb8", "rgb"]:
        rgb_step = 3
        red_offset, green_offset, blue_offset = 0, 1, 2
    elif encoding.lower() in ["bgr8", "bgr"]:
        rgb_step = 3
        red_offset, green_offset, blue_offset = 2, 1, 0
    elif encoding.lower() in ["rgba8", "rgba"]:
        rgb_step = 4
        red_offset, green_offset, blue_offset = 0, 1, 2
    elif encoding.lower() in ["bgra8", "bgra"]:
        rgb_step = 4
        red_offset, green_offset, blue_offset = 2, 1, 0
    elif encoding.lower() in ["mono8", "mono"]:
        rgb_step = 1
        red_offset = green_offset = blue_offset = 0
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    # 2) 입력 flatten
    depth_flat = depth_np.reshape(-1)
    if rgb_step == 1:
        rgb_flat = np.repeat(rgb_np.reshape(-1), 3).astype(np.uint8)
        rgb_step = 3
        red_offset = green_offset = blue_offset = 0  # R=G=B=gray
    else:
        rgb_flat = rgb_np.reshape(-1)

    num_pixels = H * W


    # 3) GPU 메모리 할당 (필요할 때만 새로)
    global _cached_num_pixels, _depth_gpu, _rgb_gpu, _xyz_gpu, _rgb_packed_gpu

    if _cached_num_pixels != num_pixels or _depth_gpu is None:
        _cached_num_pixels = num_pixels
        _depth_gpu = cuda.mem_alloc(depth_flat.nbytes)
        _rgb_gpu = cuda.mem_alloc(rgb_flat.nbytes)
        _xyz_gpu = cuda.mem_alloc(num_pixels * 3 * np.float32().nbytes)
        _rgb_packed_gpu = cuda.mem_alloc(num_pixels * np.float32().nbytes)

    cuda.memcpy_htod(_depth_gpu, depth_flat)
    cuda.memcpy_htod(_rgb_gpu, rgb_flat)

    xyz_np = np.empty((num_pixels * 3,), dtype=np.float32)
    rgb_packed_np = np.empty((num_pixels,), dtype=np.float32)

    threads_per_block = 256
    blocks = (num_pixels + threads_per_block - 1) // threads_per_block

    depth_rgb_to_xyzrgb_kernel(
        _depth_gpu,
        _rgb_gpu,
        _xyz_gpu,
        _rgb_packed_gpu,
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

    cuda.memcpy_dtoh(xyz_np, _xyz_gpu)
    cuda.memcpy_dtoh(rgb_packed_np, _rgb_packed_gpu)

    xyz_np = xyz_np.reshape((H, W, 3))
    rgb_packed_np = rgb_packed_np.reshape((H, W))

    return xyz_np, rgb_packed_np