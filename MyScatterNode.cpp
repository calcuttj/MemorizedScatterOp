#include <torch/extension.h>
#include <vector>

using namespace torch::autograd;

// The Autograd Node: Handles the backward pass
struct MyScatterNode : public Node {
    at::Tensor persistent_index; 
    std::vector<int64_t> input_shape;

    variable_list apply(variable_list&& grads) override {
        auto grad_output = grads[0];

        // Check if the incoming gradient is defined. 
        if (!grad_output.defined()) {
            return {torch::Tensor()}; // Propagate the "None" gradient back
        }

        // Zero-copy expansion in backward
        // auto expanded_index = persistent_index.unsqueeze(1).expand(input_shape);
        auto expanded_index = persistent_index;
        for (size_t i = 0; i < (input_shape.size()-1); ++i) { //All but last dim
            expanded_index = expanded_index.unsqueeze(0);
        }
        expanded_index = expanded_index.expand(input_shape);
        auto grad_src = grad_output.gather(-1, expanded_index);
        return {grad_src};
    }
};


// The Stateful Class: Holds the persistent 1D index
struct MyScatterOp : public torch::CustomClassHolder {
public:
    at::Tensor index_1d;
    MyScatterOp(at::Tensor index) : index_1d(index) {
        //TODO -- check ndim of index_1d is == 1
    }

    torch::Tensor forward(torch::Tensor src, std::vector<int64_t> results_shape) {
        // Zero-copy expansion in forward
        //Copy of index
        auto expanded_index = index_1d;
        // std::cout << "Expanded index is on " << expanded_index.device() << std::endl;

        //TODO -- make this smarter and for a given dimension.
        for (size_t i = 0; i < (src.sizes().size()-1); ++i) { //All but last dim
            expanded_index = expanded_index.unsqueeze(0);
        }
        expanded_index = expanded_index.expand_as(src);
        auto result = at::zeros(results_shape, src.options());
        {
            at::NoGradGuard no_grad;
            result.scatter_reduce_(-1, expanded_index, src, "sum", false);
        }

        if (GradMode::is_enabled() && src.requires_grad()) {
            auto grad_fn = std::make_shared<MyScatterNode>();
            grad_fn->persistent_index = index_1d;
            grad_fn->input_shape = src.sizes().vec();
            grad_fn->set_next_edges(collect_next_edges(src));
            create_gradient_edge(result, grad_fn);
        }
        return result;
    }

    void to(torch::Device device) {
        index_1d = index_1d.to(device);
    }
    
    torch::Device device() {
        return index_1d.device();
    }
};

TORCH_LIBRARY(my_ops, m) {
    m.class_<MyScatterOp>("MyScatterOp")
        .def(torch::init<at::Tensor>())
        .def("to", &MyScatterOp::to)
        .def("device", &MyScatterOp::device)
        .def("forward", &MyScatterOp::forward);
}