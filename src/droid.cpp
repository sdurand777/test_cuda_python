#include <torch/extension.h>
#include <vector>

torch::Tensor depth_filter_cuda(
    torch::Tensor a,
    torch::Tensor b);

torch::Tensor depth_filter(
    torch::Tensor a,
    torch::Tensor b) {
    return depth_filter_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depth_filter", &depth_filter, "depth_filter");
}
