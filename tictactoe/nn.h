#ifndef NN_H
#define NN_H

#include <torch/torch.h>
#include <vector> 
#include <cassert> 
#include <string>
using namespace std; 

struct SimpleNN: torch::nn::Module {
    vector<torch::nn::Linear> layers; 

    SimpleNN(int input_dim, int num_layers=2,int hidden_dims=4,int output_dims=1,torch::Device device=torch::kCPU) {
        assert(num_layers >= 2 && "Number of layers must be at least 2"); 
        for (int i = 0; i < num_layers; i++){
            int in_features = (i == 0) ? input_dim : hidden_dims; 
            int out_features = (i == num_layers - 1) ? output_dims : hidden_dims;
            auto layer = register_module("layer" + to_string(i), torch::nn::Linear(in_features, out_features)); 
            layer->to(device); 
            layers.push_back(layer); 
        }
    }

    torch::Tensor forward(torch::Tensor x,torch::Tensor action_mask=torch::Tensor()){
        for (int i = 0; i < layers.size()-1; i++){
            x = torch::leaky_relu(layers[i]->forward(x)); 
        }
        torch::Tensor final = torch::softmax(layers.back()->forward(x), 0);
        // mask out the invalid moves and readjust the probabilities, 1 in action_mask is valid, 0 is invalid
        if (action_mask.defined()){
            final = final*action_mask; 
        }
        final = final/torch::sum(final); 
        return final; 
    }
}; 

#endif