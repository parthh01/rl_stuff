#ifndef NN_H
#define NN_H

#include <torch/torch.h>
#include <vector> 
#include <cassert> 
#include <string>
using namespace std; 

struct SimpleNN: torch::nn::Module {
    vector<torch::nn::Linear> layers; 

    SimpleNN(int input_dim=10, int num_layers=2,int hidden_dims=4,int output_dims=1,torch::Device device=torch::kCPU,string filename="") {
        torch::manual_seed(42);
        if (filename != ""){
            try {
                torch::serialize::InputArchive archive;
                archive.load_from(filename);
                this->load(archive);
                // Reconstruct the layers vector based on the loaded parameters
                layers.push_back(register_module("input_layer", torch::nn::Linear(input_dim, hidden_dims)));
                for (int i = 1; i < num_layers; i++) {
                    layers.push_back(register_module("layer" + to_string(i), torch::nn::Linear(hidden_dims, hidden_dims)));
                }
                layers.push_back(register_module("output_layer", torch::nn::Linear(hidden_dims, output_dims)));
                
                // Move layers to the specified device
                for (auto& layer : layers) {
                    layer->to(device);
                }
            } catch (const c10::Error& e) {
                std::cerr << "Error loading the model: " << e.msg() << std::endl;
            }
        } else {
            assert(num_layers >= 1 && "Number of layers must be at least 1");
            auto input_layer = register_module("input_layer", torch::nn::Linear(input_dim, hidden_dims)); 
            input_layer->to(device); 
            layers.push_back(input_layer); 
            for (int i = 1; i < num_layers; i++){
                auto layer = register_module("layer" + to_string(i), torch::nn::Linear(hidden_dims, hidden_dims)); 
                layer->to(device); 
                layers.push_back(layer); 
            }
            auto output_layer = register_module("output_layer", torch::nn::Linear(hidden_dims, output_dims)); 
            output_layer->to(device); 
            layers.push_back(output_layer); 
        }
        
        // Initialize weights properly
        for (auto& layer : layers) {
            torch::nn::init::xavier_uniform_(layer->weight);
            torch::nn::init::constant_(layer->bias, 0.0);
        }
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor action_mask=torch::Tensor()) {
        x = x.to(torch::kFloat32);
        
        for (int i = 0; i < layers.size()-1; i++){
            x = torch::tanh(layers[i]->forward(x));
        }
        torch::Tensor logits = layers.back()->forward(x);
        
        if (action_mask.defined()){
            action_mask = action_mask.to(torch::kFloat32);
            
            // Apply mask before softmax
            auto masked_logits = logits.masked_fill(action_mask == 0, -1e9);
            
            // Apply temperature scaling to prevent extreme values
            masked_logits = masked_logits / 1.0;  // Temperature parameter
            
            // Compute log probabilities with numerical stability
            auto log_probs = torch::log_softmax(masked_logits, 0);
            
            // Clamp values to prevent extreme numbers
            log_probs = torch::clamp(log_probs, -20.0, 0.0);
            
            return log_probs;
        }
        return logits;

    }
    void clone_model(const SimpleNN& model){
        // Create a no_grad guard to prevent tracking operations for autograd
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < model.layers.size(); i++) {
            // Use copy_ which is an in-place operation, but safe with NoGradGuard
            layers[i]->weight.copy_(model.layers[i]->weight);
            layers[i]->bias.copy_(model.layers[i]->bias);
        }
    }

    void save_model(string filename) {
        try {
            torch::serialize::OutputArchive archive;
            this->save(archive);
            archive.save_to(filename + ".pt");
        } catch (const c10::Error& e) {
            std::cerr << "Error saving the model: " << e.msg() << std::endl;
        }
    }

}; 

long long count_parameters(const SimpleNN& nn,bool trainable=true){
    long long count = 0; 
    for (auto& param : nn.parameters()){
        if (trainable && param.requires_grad()) count += param.numel(); 
        else if (!trainable) count += param.numel(); 
    }
    return count; 
}



#endif