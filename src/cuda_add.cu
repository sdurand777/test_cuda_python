
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

// Taille des tableaux (pour l'exemple)
#define N 1024

// Kernel CUDA pour l'addition de deux tableaux
__global__ void addition_kernel(const float* a, const float* b, float* c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// Fonction exposée à Python pour l'addition des tableaux
void addition_cuda(pybind11::array_t<float> a, pybind11::array_t<float> b, pybind11::array_t<float> c) {
    // Accès aux données depuis les objets numpy
    auto buf_a = a.request();
    auto buf_b = b.request();
    auto buf_c = c.request();

    // Taille du tableau
    int size = buf_a.size;

    // Obtenir les pointeurs vers les données
    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_c = static_cast<float*>(buf_c.ptr);

    // Allocation sur le GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));

    // Copier les données de l'hôte vers le GPU
    cudaMemcpy(d_a, ptr_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, ptr_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Lancer le kernel CUDA
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    //addition_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, size);
    addition_kernel<<<1, 1>>>(d_a, d_b, d_c, size);

    // Copier le résultat du GPU vers l'hôte
    cudaMemcpy(ptr_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Libération de la mémoire GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


