#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <chrono>
#include <random>

// CPU-версия
void powerArrayCPU(const float* A, float* B, int N, float p) {
    for (int i = 0; i < N; ++i) {
        B[i] = powf(A[i], p);
    }
}

// GPU-версия (ядро CUDA)
__global__ void powerArrayCUDA(const float* A, float* B, int N, float p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        B[idx] = powf(A[idx], p);
    }
}

int main() {
    const int N = 500000;
    const float p = 0.5f;

    // Инициализация массива A случайными значениями
    float* h_A = new float[N];
    float* h_B_cpu = new float[N];
    float* h_B_gpu = new float[N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        h_A[i] = dist(gen);
    }

    // CPU-расчёт
    auto start_cpu = std::chrono::high_resolution_clock::now();
    powerArrayCPU(h_A, h_B_cpu, N, p);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_time = end_cpu - start_cpu;

    // GPU-расчёт
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    powerArrayCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, p);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_B_gpu, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Проверка корректности
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_B_cpu[i] - h_B_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    // Вывод результатов
    std::cout << "Задача 1: Возведение в степень\n";
    std::cout << "CPU время: " << cpu_time.count() * 1000 << " мс\n";
    std::cout << "GPU время: " << gpu_time_ms << " мс\n";
    std::cout << "Результат " << (correct ? "корректен" : "некорректен") << "\n";

    // Освобождение памяти
    delete[] h_A;
    delete[] h_B_cpu;
    delete[] h_B_gpu;
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}