#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void addition_cuda(pybind11::array_t<float> a, pybind11::array_t<float> b, pybind11::array_t<float> c);

void addition(pybind11::array_t<float> a, pybind11::array_t<float> b, pybind11::array_t<float> c)
{
    int tmp = 0;
    int tmp2 = 10;

    return addition_cuda(a, b, c);
}

PYBIND11_MODULE(cuda_add, m) {
    m.def("addition", &addition, "A function that adds two arrays on the GPU");
}
