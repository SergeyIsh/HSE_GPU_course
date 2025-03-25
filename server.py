import io
from PIL import Image
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from flask import Flask, request, jsonify
from contextlib import contextmanager


cuda_code = """
#include <cmath> 
__global__ void SobelFilter(
    const unsigned char* src, unsigned char* dst,
    int w, int h, int pixels_per_thread
) {
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int start_y = blockIdx.y * blockDim.y + threadIdx.y;

    int grad_tr = 75;

    int weights_x[3][3] {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int weights_y[3][3] {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    for (int i = 0; i < pixels_per_thread; i++) {
        int x = start_x + i;
        if (x >= w) return;

        float sum_x = 0, sum_y = 0;

        for (int wy = -1; wy <= 1; wy++) {
            for (int wx = -1; wx <= 1; wx++) {
                int nx = x + wx;
                int ny = start_y + wy;

                if (nx < 0) nx = -nx;
                else if (nx >= w) nx = 2 * w - nx - 1;

                if (ny < 0) ny = -ny;
                else if (ny >= h) ny = 2 * h - ny - 1;

                int idx = (ny * w + nx) * 3;

                sum_x += (0.299 * src[idx] + 0.587 * src[idx + 1] + 0.114 * src[idx + 2]) * weights_x[wx + 1][wy + 1];
                sum_y += (0.299 * src[idx] + 0.587 * src[idx + 1] + 0.114 * src[idx + 2]) * weights_y[wx + 1][wy + 1];
            }
        }
        int pixel_color = 0;
        if (pow(pow(sum_x, 2) + pow(sum_y, 2), 0.5) > grad_tr) pixel_color = 255;
        int out_idx = (start_y * w + x);
        dst[out_idx] = pixel_color;
    }
}
"""

cuda.init()
device = cuda.Device(0)
app = Flask(__name__)

@contextmanager
def gpu_ctx():
    ctx = device.make_context()
    try:
        yield ctx
    finally:
        ctx.pop()

@app.route("/sobel", methods=["POST"])
def sobel():
    try:
        buffer = io.BytesIO(request.data)
        img = np.array(Image.open(buffer))
        h, w, _ = img.shape
        out = np.zeros((h, w), dtype=np.uint8)
        pixels_per_thread = 4
        block = (16, 16, 1)
        grid = (
            (w + block[0] * pixels_per_thread - 1) // (block[0] * pixels_per_thread),
            (h + block[1] - 1) // block[1]
        )

        with gpu_ctx():
            start_time = cuda.Event()
            end_time = cuda.Event()

            start_time.record()

            d_src = cuda.mem_alloc(img.nbytes)
            d_dst = cuda.mem_alloc(out.nbytes)
            cuda.memcpy_htod(d_src, img)

            mod = SourceModule(cuda_code)
            kernel = mod.get_function("SobelFilter")
            kernel(d_src, d_dst, np.int32(w), np.int32(h), np.int32(pixels_per_thread), block=block, grid=grid)

            cuda.memcpy_dtoh(out, d_dst)

            d_src.free()
            d_dst.free()

            end_time.record()
            end_time.synchronize()

            total_gpu_time_ms = start_time.time_till(end_time)

        return jsonify({
            "image_borders": out.flatten().tolist(),
            "total_gpu_time_ms": total_gpu_time_ms
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
