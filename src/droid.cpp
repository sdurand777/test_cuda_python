#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

torch::Tensor depth_filter_cuda(
    torch::Tensor a,
    torch::Tensor b);

torch::Tensor depth_filter(
    torch::Tensor a,
    torch::Tensor b) {


    torch::Tensor test = torch::ones(10);
    //auto test_accessor = test.packed_accessor32<float,2,torch::RestrictPtrTraits>();

    std::cout << "test : \n" << test << std::endl;

    return depth_filter_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depth_filter", &depth_filter, "depth_filter");
}
