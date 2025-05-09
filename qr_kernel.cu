#include <cuda_runtime.h>
#include <stdint.h>

#define QR_VERSION 1
#define QR_SIZE (QR_VERSION * 4 + 17)
#define QR_SCALE 8
#define IMAGE_SIZE (QR_SIZE * QR_SCALE)

__global__ void rasterize_kernel(uint8_t *modules, uint8_t *images, int qr_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= qr_count) return;

    uint8_t *module = &modules[idx * QR_SIZE * QR_SIZE];
    uint8_t *image = &images[idx * IMAGE_SIZE * IMAGE_SIZE];

    for (int y = 0; y < QR_SIZE; ++y) {
        for (int x = 0; x < QR_SIZE; ++x) {
            uint8_t val = module[y * QR_SIZE + x];
            for (int dy = 0; dy < QR_SCALE; ++dy) {
                for (int dx = 0; dx < QR_SCALE; ++dx) {
                    int img_y = y * QR_SCALE + dy;
                    int img_x = x * QR_SCALE + dx;
                    image[img_y * IMAGE_SIZE + img_x] = val;
                }
            }
        }
    }
}

extern "C" void rasterize_qrs_cuda(uint8_t *modules, uint8_t *images, int qr_count) {
    uint8_t *d_modules, *d_images;
    size_t module_bytes = qr_count * QR_SIZE * QR_SIZE;
    size_t image_bytes = qr_count * IMAGE_SIZE * IMAGE_SIZE;

    cudaMalloc(&d_modules, module_bytes);
    cudaMalloc(&d_images, image_bytes);

    cudaMemcpy(d_modules, modules, module_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (qr_count + threadsPerBlock - 1) / threadsPerBlock;
    rasterize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_modules, d_images, qr_count);
    cudaDeviceSynchronize();

    cudaMemcpy(images, d_images, image_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_modules);
    cudaFree(d_images);
}