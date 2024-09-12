#include <torch/extension.h>
#include <vector>

torch::Tensor addition_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor addition(  torch::Tensor a, 
                torch::Tensor b)
{
    int tmp = 0;
    int tmp2 = 10;

    return addition_cuda(a, b);
}

PYBIND11_MODULE(cuda_add, m) {
    m.def("addition", &addition, "A function that adds two arrays on the GPU");
}
