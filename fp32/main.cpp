#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, 
                                                                torch::Tensor l, torch::Tensor m);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(torch::Tensor q, torch::Tensor k, torch::Tensor v, 
                                                                 torch::Tensor O, torch::Tensor dO, 
                                                                 torch::Tensor l, torch::Tensor m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
    m.def("backward", torch::wrap_pybind_function(backward), "backward");
}
