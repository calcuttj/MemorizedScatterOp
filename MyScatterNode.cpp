#include <torch/extension.h>
// #include <ATen/at_indexing.h>
// #include <torch/c10/excep
#include <vector>

using namespace torch::autograd;
using namespace torch::indexing;
// The Autograd Node: Handles the backward pass
struct MyScatterNode : public Node {
    at::Tensor persistent_index;
    torch::Tensor src, result;
    std::vector<TensorIndex> where_src_is_result;
    std::vector<int64_t> input_shape;
    std::string operation;

    std::vector<std::string> implemented = {
        "sum" //,
        // "amax",
        // "amin"
    };

    variable_list apply(variable_list&& grads) override {

        auto locate_op = std::find(implemented.begin(), implemented.end(), operation);
        TORCH_CHECK(
            (locate_op != implemented.end()),
            "Only sum are implemented. User requested ",
            operation
        );

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
        
        torch::Tensor grad_src;
        if (operation == "sum") {
            grad_src = grad_output.gather(-1, expanded_index);
        }
        else if (operation == "amax" || operation == "amin") {
            torch::Tensor value = result.gather(-1/*dim*/, expanded_index);
            torch::Tensor test_src_is_result = torch::zeros_like(value);
            test_src_is_result.index_put_(where_src_is_result, 1.);
            
            torch::Tensor N_to_distribute = at::zeros_like(result).scatter_add(-1/*dim*/, expanded_index, test_src_is_result);
            torch::Tensor grad_distributed = grad_output / N_to_distribute;
            grad_src = test_src_is_result * grad_distributed.gather(-1/*dim*/, expanded_index);
        }
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


    torch::Tensor forward(torch::Tensor src, std::vector<int64_t> results_shape, std::string_view operation) {

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
        if (operation == "sum") {
            {
                at::NoGradGuard no_grad;
                result.scatter_reduce_(-1, expanded_index, src, "sum", false);
            }
            if (GradMode::is_enabled() && src.requires_grad()) {
                auto grad_fn = std::make_shared<MyScatterNode>();
                grad_fn->persistent_index = index_1d;
                grad_fn->input_shape = src.sizes().vec();
                grad_fn->operation = operation;
                grad_fn->set_next_edges(collect_next_edges(src));
                create_gradient_edge(result, grad_fn);
            }
        }
        else {
            TORCH_CHECK(
            false,
            "Only sum is implemented. User requested ",
            operation
        );
        }
        // else  if (operation == "amax" || operation == "amin") {

            // Using torch_scatter might be easier for this. Figure out how to build against torch_scatter...
        //     if (GradMode::is_enabled() && src.requires_grad()) {
        //         auto grad_fn = std::make_shared<MyScatterNode>();
        //         grad_fn->persistent_index = index_1d;
        //         grad_fn->input_shape = src.sizes().vec();
        //         grad_fn->operation = operation;
               
        //         grad_fn->result = result;
        //         auto value = result.gather(-1/*dim*/, expanded_index);
        //         torch::Tensor src_is_result = (src == value);
        //         auto where_src_is_result = src_is_result.nonzero();
        //         std::vector<TensorIndex> as_array;
        //         for (const auto & c : where_src_is_result.t().unbind(0))
        //             as_array.push_back(c);

        //         grad_fn->where_src_is_result = as_array;
        //         grad_fn->set_next_edges(collect_next_edges(src));
        //         create_gradient_edge(result, grad_fn);
        //     }
        // }

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
        .def(
            "forward",
            &MyScatterOp::forward
        );
}