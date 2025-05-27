#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <random>

// CPU-версия
void rotateImageCPU(unsigned char* input, unsigned char* output, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            output[y * width + x] = input[(width - x - 1) * height + y];
        }
    }
}

// GPU-версия (ядро CUDA)
__global__ void rotateImageCUDA(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[y * width + x] = input[(width - x - 1) * height + y];
    }
}

int main() {
    const int width = 512;
    const int height = 512;
    const int total_pixels = width * height;

    // Инициализация изображения случайными значениями (0-255)
    unsigned char* h_input = new unsigned char[total_pixels];
    unsigned char* h_output_cpu = new unsigned char[total_pixels];
    unsigned char* h_output_gpu = new unsigned char[total_pixels];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> dist(0, 255);

    for (int i = 0; i < total_pixels; ++i) {
        h_input[i] = dist(gen);
    }

    // CPU-расчёт
    auto start_cpu = std::chrono::high_resolution_clock::now();
    rotateImageCPU(h_input, h_output_cpu, width, height);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_time = end_cpu - start_cpu;

    // GPU-расчёт
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, total_pixels * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, total_pixels * sizeof(unsigned char));

    cudaMemcpy(d_input, h_input, total_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rotateImageCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_output_gpu, d_output, total_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Проверка корректности
    bool correct = true;
    for (int i = 0; i < total_pixels; ++i) {
        if (h_output_cpu[i] != h_output_gpu[i]) {
            correct = false;
            break;
        }
    }

    // Вывод результатов
    std::cout << "\nЗадача 2: Поворот изображения\n";
    std::cout << "CPU время: " << cpu_time.count() * 1000 << " мс\n";
    std::cout << "GPU время: " << gpu_time_ms << " мс\n";
    std::cout << "Результат " << (correct ? "корректен" : "некорректен") << "\n";

    // Освобождение памяти
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}