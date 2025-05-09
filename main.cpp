#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <string>
#include <qrencode.h>
#include <cstdint>
#include <chrono>

#define QR_COUNT 1
#define QR_VERSION 1
#define QR_SIZE (QR_VERSION * 4 + 17)
#define QR_SCALE 8
#define IMAGE_SIZE (QR_SIZE * QR_SCALE)

extern "C" void rasterize_qrs_cuda(uint8_t *all_modules, uint8_t *all_images, int qr_count);

void save_image(const uint8_t *image, int index) {
    char filename[64];
    snprintf(filename, sizeof(filename), "qr_%04d.png", index);
    stbi_write_png(filename, IMAGE_SIZE, IMAGE_SIZE, 1, image, IMAGE_SIZE);
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> inputs(QR_COUNT);
    for (int i = 0; i < QR_COUNT; ++i) {
        inputs[i] = "QRCode #" + std::to_string(i + 1);
    }

    std::vector<uint8_t> modules(QR_COUNT * QR_SIZE * QR_SIZE);
    std::vector<uint8_t> images(QR_COUNT * IMAGE_SIZE * IMAGE_SIZE);

    for (int i = 0; i < QR_COUNT; ++i) {
        QRcode *qrcode = QRcode_encodeString(inputs[i].c_str(), QR_VERSION, QR_ECLEVEL_L, QR_MODE_8, 1);
        if (!qrcode) {
            std::cerr << "QRcode generation failed for index " << i << std::endl;
            continue;
        }
        uint8_t *dst = &modules[i * QR_SIZE * QR_SIZE];
        for (int y = 0; y < QR_SIZE; ++y) {
            for (int x = 0; x < QR_SIZE; ++x) {
                dst[y * QR_SIZE + x] = (qrcode->data[y * QR_SIZE + x] & 0x01) ? 255 : 0;
            }
        }
        QRcode_free(qrcode);
    }

    rasterize_qrs_cuda(modules.data(), images.data(), QR_COUNT);

    for (int i = 0; i < QR_COUNT; ++i) {
        save_image(&images[i * IMAGE_SIZE * IMAGE_SIZE], i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "✅ QR code generation completed successfully in " 
        << duration.count() << " seconds." << std::endl;

    return 0;
}
