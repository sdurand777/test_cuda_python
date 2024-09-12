
#include <torch/extension.h>
#include <cuda_runtime.h>

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
torch::Tensor addition_cuda(torch::Tensor a, torch::Tensor b) {
    // Vérifier que les tenseurs sont sur le GPU et de type float
    TORCH_CHECK(a.is_cuda(), "Le tenseur a doit être sur le GPU");
    TORCH_CHECK(b.is_cuda(), "Le tenseur b doit être sur le GPU");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Le tenseur a doit être de type float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Le tenseur b doit être de type float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Les deux tenseurs doivent avoir la même taille");

    // Créer un tenseur de sortie pour le résultat sur le GPU
    torch::Tensor c = torch::zeros_like(a);

    // Taille du tableau
    int size = a.numel();

    // Obtenir les pointeurs vers les données sur le GPU
    const float* d_a = a.data_ptr<float>();
    const float* d_b = b.data_ptr<float>();
    float* d_c = c.data_ptr<float>();

    // Lancer le kernel CUDA
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addition_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, size);

    // Retourner le tenseur résultat
    return c;
}


