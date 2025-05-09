#include <cuda_runtime.h>
#include <stdint.h>

#define QR_VERSION 1
#define QR_SIZE (QR_VERSION * 4 + 17)       
#define QR_SCALE 8
#define IMAGE_SIZE (QR_SIZE * QR_SCALE)  
#define BLOCK_SIZE 16                        

__global__ void rasterize_kernel(uint8_t *modules, uint8_t *images, int qr_count) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= IMAGE_SIZE || y >= IMAGE_SIZE) return;

    int qr_id = blockIdx.z;
    if (qr_id >= qr_count) return;
    int module_x = x / QR_SCALE;
    int module_y = y / QR_SCALE;
    int module_index = qr_id * QR_SIZE * QR_SIZE + module_y * QR_SIZE + module_x;
    uint8_t val = modules[module_index];
    int image_index = qr_id * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x;
    images[image_index] = val;
}

extern "C" void rasterize_qrs_cuda(uint8_t *modules, uint8_t *images, int qr_count) {
    uint8_t *d_modules, *d_images;
    size_t module_bytes = qr_count * QR_SIZE * QR_SIZE;
    size_t image_bytes = qr_count * IMAGE_SIZE * IMAGE_SIZE;

    cudaMalloc(&d_modules, module_bytes);
    cudaMalloc(&d_images, image_bytes);
    cudaMemcpy(d_modules, modules, module_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((IMAGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (IMAGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       qr_count);

    rasterize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_modules, d_images, qr_count);
    cudaDeviceSynchronize();

    cudaMemcpy(images, d_images, image_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_modules);
    cudaFree(d_images);
}
