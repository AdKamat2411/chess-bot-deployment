#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "chess.hpp"
#include <string>
#include <vector>
#include <map>
#include <torch/script.h>

using namespace chess;

/**
 * Neural Network wrapper for AlphaZero model
 * Loads TorchScript model and provides inference interface
 */
class NeuralNetwork {
private:
    torch::jit::script::Module model_;
    bool loaded_;
    
    // Move mapping: policy index -> chess::Move
    // Policy is 4096-dim: from_square * 64 + to_square (no promotions in this encoding)
    std::map<int, Move> policy_to_move_;
    std::map<Move, int> move_to_policy_;
    
    // Pre-allocated buffers for performance (thread-local to avoid race conditions)
    static thread_local std::vector<float> encoding_buffer_;
    static thread_local torch::Tensor input_tensor_buffer_;
    static thread_local torch::Tensor batch_tensor_buffer_;  // For batch inference
    static bool buffers_initialized_;
    
    void initialize_move_mapping();
    int move_to_policy_index(const Move& move) const;
    void initialize_buffers();
    
public:
    NeuralNetwork();
    ~NeuralNetwork();
    
    /**
     * Load TorchScript model from file
     * @param model_path Path to aznet_traced.pt
     * @return true if successful
     */
    bool load_model(const std::string& model_path);
    
    /**
     * Encode chess board to 18-channel tensor
     * @param board Chess board
     * @return Vector of floats (18 * 8 * 8 = 1152 elements)
     */
    std::vector<float> encode_board(const Board& board) const;
    
    /**
     * Encode chess board directly into pre-allocated buffer (optimized version)
     * @param board Chess board
     * @param buffer Output buffer (must be at least 18 * 8 * 8 floats)
     */
    void encode_board_into_buffer(const Board& board, float* buffer) const;
    
    /**
     * Run inference on board position
     * @param board Chess board
     * @param policy_out Output: policy probabilities for each legal move (normalized), keyed by UCI string
     * @param value_out Output: position value in [0, 1] from white's perspective
     * @return true if successful
     */
    bool predict(const Board& board,
                std::map<std::string, double>& policy_out,
                double& value_out,
                double& raw_value_out);  // Raw value from model (same as value_out, model already outputs tanh'd values)
    
    /**
     * Batch inference for multiple positions (optimized)
     * @param boards Vector of chess boards to evaluate
     * @param policies_out Output: vector of policy maps (one per board)
     * @param values_out Output: vector of values (one per board)
     * @param raw_values_out Output: vector of raw values (one per board)
     * @return true if successful
     */
    bool predict_batch(const std::vector<Board>& boards,
                      std::vector<std::map<std::string, double>>& policies_out,
                      std::vector<double>& values_out,
                      std::vector<double>& raw_values_out);
    
    bool is_loaded() const { return loaded_; }
};

#endif // NEURAL_NETWORK_H

