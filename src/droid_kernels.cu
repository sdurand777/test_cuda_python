#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>


__global__ void addition_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> c)
{
    // Calcul de l'index global du thread en 2D (x, y)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Ligne du tenseur
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Colonne du tenseur

    // Vérification pour s'assurer que le thread est dans les limites du tenseur
    if (row < a.size(0) && col < a.size(1)) {
        // Addition des deux tenseurs élément par élément
        c[row][col] = a[row][col] + b[row][col];
    }
}


// Fonction pour imprimer un tenseur
void print_tensor(torch::Tensor tensor) {
    // Copier le tenseur du GPU vers le CPU
    auto cpu_tensor = tensor.to(torch::kCPU);
    auto data_ptr = cpu_tensor.data_ptr<float>();
    auto size = cpu_tensor.numel();
    
    std::cout << "Tensor values:\n";
    for (int i = 0; i < size; ++i) {
        if (i % cpu_tensor.size(1) == 0) std::cout << "\n";  // Nouvelle ligne pour chaque ligne du tenseur
        std::cout << data_ptr[i] << " ";
    }
    std::cout << std::endl;
}



torch::Tensor depth_filter_cuda(
        torch::Tensor a,
        torch::Tensor b)
{
    // Vérification que les tenseurs sont sur le GPU et ont le bon type de données
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Tensor a must be of type float");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "Tensor b must be of type float");

    // Vérification que les dimensions des tenseurs correspondent
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors a and b must have the same shape");

    // Créer un tenseur de sortie sur le GPU avec la même forme que a et b
    auto c = torch::zeros_like(a);

    //printf("Tensor a");
    //print_tensor(a);

    // Si vous avez besoin de transférer les tenseurs sur le CPU :
    torch::Tensor a_cpu = a.cpu();
    torch::Tensor b_cpu = b.cpu();

    // Obtenir des PackedTensorAccessor pour accéder aux données dans CUDA
    auto a_accessor = a.packed_accessor32<float,2,torch::RestrictPtrTraits>();
    auto b_accessor = b.packed_accessor32<float,2,torch::RestrictPtrTraits>();
    auto c_accessor = c.packed_accessor32<float,2,torch::RestrictPtrTraits>();

    // Configuration des dimensions des blocs et de la grille
    dim3 threadsPerBlock(16, 16);  // 16x16 threads par bloc
    dim3 numBlocks((a.size(1) + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (a.size(0) + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lancement du kernel CUDA
    addition_kernel<<<numBlocks, threadsPerBlock>>>(a_accessor, b_accessor, c_accessor);
    return c;
}


