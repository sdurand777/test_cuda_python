#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/core/Formatting.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

torch::Tensor depth_filter_cuda(
    torch::Tensor a,
    torch::Tensor b);

torch::Tensor depth_filter_cuda_d(
    torch::Tensor a,
    torch::Tensor b);

torch::Tensor depth_filter(
    torch::Tensor a,
    torch::Tensor b) {


    torch::Tensor test = torch::ones(10);

    std::vector<int> raw_vec(10,1);
    std::cout << "Premier élément (avec []) : " << raw_vec[0] << std::endl;

    // Assure-toi que le tensor est sur CPU
    torch::Tensor tensor = test.to(torch::kCPU);

    // Obtiens un pointeur vers les données du tensor
    float* data_ptr = tensor.data_ptr<float>();

    // Crée un std::vector et copie les données du tensor
    std::vector<float> vector(data_ptr, data_ptr + tensor.numel());

    //auto test_accessor = test.packed_accessor32<float,2,torch::RestrictPtrTraits>();

    std::cout << "test : \n" << test << std::endl;

    return depth_filter_cuda(a, b);
}


torch::Tensor depth_filter_d(
    torch::Tensor a,
    torch::Tensor b) {

    return depth_filter_cuda_d(a, b);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depth_filter", &depth_filter, "depth_filter");
    m.def("depth_filter_d", &depth_filter_d, "depth_filter_d");

}
