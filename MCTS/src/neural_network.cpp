#include "../include/neural_network.h"
#include "../include/chess.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <torch/script.h>
#include <torch/torch.h>

// Test mode: Set to true to use deterministic stub NN for testing MCTS
static constexpr bool USE_NN_STUB = false;

namespace {
constexpr bool kEnableNNLogging = false;
}

// Static thread-local buffer initialization
thread_local std::vector<float> NeuralNetwork::encoding_buffer_(18 * 8 * 8, 0.0f);
thread_local torch::Tensor NeuralNetwork::input_tensor_buffer_;
thread_local torch::Tensor NeuralNetwork::batch_tensor_buffer_;
bool NeuralNetwork::buffers_initialized_ = false;

NeuralNetwork::NeuralNetwork() : loaded_(false) {
    initialize_move_mapping();
    initialize_buffers();
}

void NeuralNetwork::initialize_buffers() {
    if (!buffers_initialized_) {
        // Pre-allocate input tensor buffer (shape: 1, 18, 8, 8)
        input_tensor_buffer_ = torch::empty({1, 18, 8, 8}, torch::kFloat32);
        // Pre-allocate batch tensor buffer (shape: max_batch_size, 18, 8, 8)
        const int MAX_BATCH_SIZE = 32;  // Match MCTS batch size
        batch_tensor_buffer_ = torch::empty({MAX_BATCH_SIZE, 18, 8, 8}, torch::kFloat32);
        buffers_initialized_ = true;
    }
}

NeuralNetwork::~NeuralNetwork() {
    // torch::jit::script::Module handles its own cleanup
}

void NeuralNetwork::initialize_move_mapping() {
    // Create mapping from policy index to Move
    // Policy is 64*64 = 4096 dimensions (from_square * 64 + to_square)
    // We'll build this mapping on first use
    policy_to_move_.clear();
    move_to_policy_.clear();
}

int NeuralNetwork::move_to_policy_index(const Move& move) const {
    Square from = move.from();
    Square to = move.to();
    
    // MATCH PYTHON: move_to_index(move) = move.from_square * 64 + move.to_square
    // where from_square/to_square are python-chess indices (0-63)
    // 
    // chess.hpp Square::index() matches python-chess exactly:
    // - A1 = 0, B1 = 1, ..., H1 = 7
    // - A2 = 8, ..., H8 = 63
    // So we can use index() directly, but let's be explicit with rank/file to match Python's row/col
    
    // Python: row = square // 8, col = square % 8
    // chess.hpp: rank = index() >> 3, file = index() & 7
    // So: idx = rank * 8 + file = index() (they're equivalent)
    int from_idx = from.index();  // This matches python-chess square index
    int to_idx = to.index();      // This matches python-chess square index
    
    // Policy index = from_square * 64 + to_square (matches Python exactly)
    // Note: Promotions are NOT included in the 4096-dim policy
    // They will need to be handled separately or the model needs to be retrained
    int policy_idx = from_idx * 64 + to_idx;
    
    return policy_idx;
}

std::vector<float> NeuralNetwork::encode_board(const Board& board) const {
    // Encode board as 18-channel 8x8 tensor
    // Channels 0-11: Piece planes (same as before)
    //   [white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king,
    //    black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king]
    // Channel 12: Side to move (1.0 for white, 0.0 for black) - full 8x8 plane
    // Channel 13: White kingside castling rights (1.0 if available, 0.0 otherwise) - full 8x8 plane
    // Channel 14: White queenside castling rights - full 8x8 plane
    // Channel 15: Black kingside castling rights - full 8x8 plane
    // Channel 16: Black queenside castling rights - full 8x8 plane
    // Channel 17: En passant target square (1.0 at the square, 0.0 elsewhere) - full 8x8 plane
    // 
    // IMPORTANT: This must match Python's board_to_matrix() exactly:
    // - Python: row = square // 8, col = square % 8, mat[plane, row, col] = 1
    // - No vertical flipping! rank 0 = rank 1 (white's first rank)
    
    std::vector<float> tensor(18 * 8 * 8, 0.0f);
    
    // Planes 0-11: Pieces
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            Square sq = Square(Rank(rank), File(file));
            Piece piece = board.at(sq);
            
            if (piece == Piece::NONE) continue;
            
            // Determine channel based on piece type and color
            int channel = 0;
            PieceType pt = piece.type();
            Color color = piece.color();
            
            // Piece type index (0-5: pawn, knight, bishop, rook, queen, king)
            int piece_idx = 0;
            if (pt == PieceType::PAWN) piece_idx = 0;
            else if (pt == PieceType::KNIGHT) piece_idx = 1;
            else if (pt == PieceType::BISHOP) piece_idx = 2;
            else if (pt == PieceType::ROOK) piece_idx = 3;
            else if (pt == PieceType::QUEEN) piece_idx = 4;
            else if (pt == PieceType::KING) piece_idx = 5;
            
            // Channel = color_offset + piece_idx
            // White: channels 0-5, Black: channels 6-11
            channel = (color == Color::WHITE) ? piece_idx : (piece_idx + 6);
            
            // MATCH PYTHON: row = rank, col = file (NO FLIPPING!)
            // Python: idx64 = row * 8 + col where row = square // 8, col = square % 8
            // chess.hpp: Square::index() = file + rank * 8, which matches python-chess
            int idx64 = rank * 8 + file;
            int tensor_idx = channel * 64 + idx64;
            tensor[tensor_idx] = 1.0f;
        }
    }
    
    // Plane 12: Side to move (1.0 for white, 0.0 for black) - full 8x8 plane
    float side_to_move_value = (board.sideToMove() == Color::WHITE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 12 * 64 + idx64;
            tensor[tensor_idx] = side_to_move_value;
        }
    }
    
    // Planes 13-16: Castling rights
    Board::CastlingRights cr = board.castlingRights();
    
    // Plane 13: White kingside castling
    float white_kingside = cr.has(Color::WHITE, Board::CastlingRights::Side::KING_SIDE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 13 * 64 + idx64;
            tensor[tensor_idx] = white_kingside;
        }
    }
    
    // Plane 14: White queenside castling
    float white_queenside = cr.has(Color::WHITE, Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 14 * 64 + idx64;
            tensor[tensor_idx] = white_queenside;
        }
    }
    
    // Plane 15: Black kingside castling
    float black_kingside = cr.has(Color::BLACK, Board::CastlingRights::Side::KING_SIDE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 15 * 64 + idx64;
            tensor[tensor_idx] = black_kingside;
        }
    }
    
    // Plane 16: Black queenside castling
    float black_queenside = cr.has(Color::BLACK, Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 16 * 64 + idx64;
            tensor[tensor_idx] = black_queenside;
        }
    }
    
    // Plane 17: En passant target square
    Square ep_sq = board.enpassantSq();
    if (ep_sq != Square::NO_SQ && ep_sq.is_valid()) {
        // Convert square to rank and file
        int ep_rank = ep_sq.rank();
        int ep_file = ep_sq.file();
        int idx64 = ep_rank * 8 + ep_file;
        int tensor_idx = 17 * 64 + idx64;
        tensor[tensor_idx] = 1.0f;
    }
    // Otherwise, plane 17 remains all zeros (already initialized)
    
    return tensor;
}

void NeuralNetwork::encode_board_into_buffer(const Board& board, float* buffer) const {
    // Same logic as encode_board but writes directly to buffer
    // Clear buffer first (caller should do this, but be safe)
    std::fill(buffer, buffer + (18 * 8 * 8), 0.0f);
    
    // Planes 0-11: Pieces
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            Square sq = Square(Rank(rank), File(file));
            Piece piece = board.at(sq);
            
            if (piece == Piece::NONE) continue;
            
            int channel = 0;
            PieceType pt = piece.type();
            Color color = piece.color();
            
            int piece_idx = 0;
            if (pt == PieceType::PAWN) piece_idx = 0;
            else if (pt == PieceType::KNIGHT) piece_idx = 1;
            else if (pt == PieceType::BISHOP) piece_idx = 2;
            else if (pt == PieceType::ROOK) piece_idx = 3;
            else if (pt == PieceType::QUEEN) piece_idx = 4;
            else if (pt == PieceType::KING) piece_idx = 5;
            
            channel = (color == Color::WHITE) ? piece_idx : (piece_idx + 6);
            
            int idx64 = rank * 8 + file;
            int tensor_idx = channel * 64 + idx64;
            buffer[tensor_idx] = 1.0f;
        }
    }
    
    // Plane 12: Side to move
    float side_to_move_value = (board.sideToMove() == Color::WHITE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 12 * 64 + idx64;
            buffer[tensor_idx] = side_to_move_value;
        }
    }
    
    // Planes 13-16: Castling rights
    Board::CastlingRights cr = board.castlingRights();
    
    float white_kingside = cr.has(Color::WHITE, Board::CastlingRights::Side::KING_SIDE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 13 * 64 + idx64;
            buffer[tensor_idx] = white_kingside;
        }
    }
    
    float white_queenside = cr.has(Color::WHITE, Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 14 * 64 + idx64;
            buffer[tensor_idx] = white_queenside;
        }
    }
    
    float black_kingside = cr.has(Color::BLACK, Board::CastlingRights::Side::KING_SIDE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 15 * 64 + idx64;
            buffer[tensor_idx] = black_kingside;
        }
    }
    
    float black_queenside = cr.has(Color::BLACK, Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0f : 0.0f;
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int idx64 = rank * 8 + file;
            int tensor_idx = 16 * 64 + idx64;
            buffer[tensor_idx] = black_queenside;
        }
    }
    
    // Plane 17: En passant target square
    Square ep_sq = board.enpassantSq();
    if (ep_sq != Square::NO_SQ && ep_sq.is_valid()) {
        int ep_rank = ep_sq.rank();
        int ep_file = ep_sq.file();
        int idx64 = ep_rank * 8 + ep_file;
        int tensor_idx = 17 * 64 + idx64;
        buffer[tensor_idx] = 1.0f;
    }
}

bool NeuralNetwork::load_model(const std::string& model_path) {
    // In stub mode, always succeed (don't actually load model)
    if constexpr (USE_NN_STUB) {
        loaded_ = true;
        std::cout << "[NN STUB] Stub mode enabled - model loading skipped" << std::endl;
        return true;
    }
    
    try {
        // Load TorchScript model
        model_ = torch::jit::load(model_path);
        model_.eval();  // Set to evaluation mode
        
        // Try to move model to GPU if available
        if (torch::cuda::is_available()) {
            model_.to(torch::kCUDA);
            std::cerr << "[NN] Model moved to GPU (CUDA available)" << std::endl;
        } else {
            std::cerr << "[NN] Model running on CPU (CUDA not available)" << std::endl;
        }
        
        // Verify model loaded successfully
        loaded_ = true;
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

bool NeuralNetwork::predict_batch(const std::vector<Board>& boards,
                                  std::vector<std::map<std::string, double>>& policies_out,
                                  std::vector<double>& values_out,
                                  std::vector<double>& raw_values_out) {
    if (!loaded_) {
        std::cerr << "ERROR: Neural network model not loaded. Cannot predict." << std::endl;
        return false;
    }
    
    if (boards.empty()) {
        return true;
    }
    
    const size_t batch_size = boards.size();
    policies_out.clear();
    values_out.clear();
    raw_values_out.clear();
    policies_out.resize(batch_size);
    values_out.resize(batch_size);
    raw_values_out.resize(batch_size);
    
    // Profiling timers
    auto total_start = std::chrono::high_resolution_clock::now();
    auto encode_start = total_start;
    auto inference_start = total_start;
    auto postprocess_start = total_start;
    
    try {
        // Get batch tensor buffer (resize if needed)
        auto buffer_start = std::chrono::high_resolution_clock::now();
        if (batch_tensor_buffer_.size(0) < static_cast<int64_t>(batch_size)) {
            batch_tensor_buffer_ = torch::empty({static_cast<int64_t>(batch_size), 18, 8, 8}, torch::kFloat32);
        }
        
        // Create a view of the buffer for this batch
        auto batch_tensor = batch_tensor_buffer_.slice(0, 0, static_cast<int64_t>(batch_size));
        float* batch_data = batch_tensor.data_ptr<float>();
        auto buffer_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - buffer_start).count();
        
        // Encode all boards into batch tensor (on CPU)
        encode_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; i++) {
            float* board_data = batch_data + i * (18 * 8 * 8);
            std::fill(board_data, board_data + (18 * 8 * 8), 0.0f);
            encode_board_into_buffer(boards[i], board_data);
        }
        auto encode_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - encode_start).count();
        
        // Move tensor to GPU if available (after encoding)
        if (torch::cuda::is_available()) {
            batch_tensor = batch_tensor.to(torch::kCUDA);
        }
        
        // Run batch inference
        inference_start = std::chrono::high_resolution_clock::now();
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(batch_tensor);
        
        auto outputs = model_.forward(inputs).toTuple();
        torch::Tensor policy_logits_batch = outputs->elements()[0].toTensor();  // Shape: (batch_size, 4096)
        torch::Tensor value_tensor_batch = outputs->elements()[1].toTensor();   // Shape: (batch_size,) or (batch_size, 1)
        
        // Squeeze value tensor if it has an extra dimension
        if (value_tensor_batch.dim() == 2 && value_tensor_batch.size(1) == 1) {
            value_tensor_batch = value_tensor_batch.squeeze(1);  // Remove dimension 1
        }
        
        // Apply softmax to policy logits
        torch::Tensor policy_probs_batch = torch::softmax(policy_logits_batch, 1);  // Shape: (batch_size, 4096)
        auto inference_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - inference_start).count();
        
        // Extract results for each board
        postprocess_start = std::chrono::high_resolution_clock::now();
        auto policy_accessor = policy_probs_batch.accessor<float, 2>();
        auto value_accessor = value_tensor_batch.accessor<float, 1>();
        
        for (size_t i = 0; i < batch_size; i++) {
            // Extract value
            float model_value = value_accessor[i];
            raw_values_out[i] = static_cast<double>(model_value);
            values_out[i] = static_cast<double>(model_value);
            
            // Extract policy for this board
            Movelist legal_moves;
            movegen::legalmoves(legal_moves, boards[i]);
            
            if (legal_moves.empty()) {
                std::cerr << "ERROR: No legal moves in position " << i << ". Cannot predict policy." << std::endl;
                continue;
            }
            
            // Sum probabilities for legal moves (for normalization)
            double total_prob = 0.0;
            for (const Move& move : legal_moves) {
                int policy_idx = move_to_policy_index(move);
                if (policy_idx >= 0 && policy_idx < 4096) {
                    total_prob += policy_accessor[i][policy_idx];
                }
            }
            
            // Normalize and store policy
            if (total_prob > 0.0) {
                for (const Move& move : legal_moves) {
                    int policy_idx = move_to_policy_index(move);
                    if (policy_idx >= 0 && policy_idx < 4096) {
                        std::string move_uci = static_cast<std::string>(move.from()) + static_cast<std::string>(move.to());
                        if (move.typeOf() == Move::PROMOTION) {
                            PieceType pt = move.promotionType();
                            char promo = 'q';
                            if (pt == PieceType::KNIGHT) promo = 'n';
                            else if (pt == PieceType::BISHOP) promo = 'b';
                            else if (pt == PieceType::ROOK) promo = 'r';
                            move_uci += promo;
                        }
                        double prob = policy_accessor[i][policy_idx] / total_prob;
                        policies_out[i][move_uci] = prob;
                    }
                }
            } else {
                // Fallback: uniform distribution
                double uniform_prob = 1.0 / legal_moves.size();
                for (const Move& move : legal_moves) {
                    std::string move_uci = static_cast<std::string>(move.from()) + static_cast<std::string>(move.to());
                    if (move.typeOf() == Move::PROMOTION) {
                        PieceType pt = move.promotionType();
                        char promo = 'q';
                        if (pt == PieceType::KNIGHT) promo = 'n';
                        else if (pt == PieceType::BISHOP) promo = 'b';
                        else if (pt == PieceType::ROOK) promo = 'r';
                        move_uci += promo;
                    }
                    policies_out[i][move_uci] = uniform_prob;
                }
            }
        }
        auto postprocess_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - postprocess_start).count();
        auto total_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_start).count();
        
        // Print profiling info
        std::cerr << "[NN PROFILE] Batch size: " << batch_size 
                  << " | Total: " << std::fixed << std::setprecision(3) << total_time * 1000 << "ms"
                  << " | Buffer: " << buffer_time * 1000 << "ms"
                  << " | Encode: " << encode_time * 1000 << "ms"
                  << " | Inference: " << inference_time * 1000 << "ms"
                  << " | Postprocess: " << postprocess_time * 1000 << "ms"
                  << " | Per position: " << (total_time / batch_size) * 1000 << "ms"
                  << std::endl;
        
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during batch inference: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during batch inference: " << e.what() << std::endl;
        return false;
    }
}

bool NeuralNetwork::predict(const Board& board,
                            std::map<std::string, double>& policy_out,
                            double& value_out,
                            double& raw_value_out) {
    // TEST MODE: Deterministic stub for testing MCTS
    if constexpr (USE_NN_STUB) {
        policy_out.clear();
        
        // Get legal moves
        Movelist legal_moves;
        movegen::legalmoves(legal_moves, board);
        
        if (legal_moves.empty()) {
            std::cerr << "[NN STUB] No legal moves!" << std::endl;
            return false;
        }
        
        // DEBUG: Always log all legal moves for root position (White to move, starting FEN)
        std::string fen = board.getFen();
        bool is_root = (fen.find("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR") == 0 && 
                        board.sideToMove() == Color::WHITE &&
                        board.fullMoveNumber() == 1);
        
        if (is_root) {
            std::cerr << "[NN STUB] ===== ROOT POSITION - ALL LEGAL MOVES =====" << std::endl;
            std::cerr << "[NN STUB] FEN: " << fen << std::endl;
            std::cerr << "[NN STUB] Legal moves count: " << legal_moves.size() << std::endl;
            std::cerr << "[NN STUB] All legal moves:" << std::endl;
            int move_idx = 0;
            for (const Move& move : legal_moves) {
                std::string move_uci = static_cast<std::string>(move.from()) + static_cast<std::string>(move.to());
                if (move.typeOf() == Move::PROMOTION) {
                    PieceType pt = move.promotionType();
                    char promo = 'q';
                    if (pt == PieceType::KNIGHT) promo = 'n';
                    else if (pt == PieceType::BISHOP) promo = 'b';
                    else if (pt == PieceType::ROOK) promo = 'r';
                    move_uci += promo;
                }
                std::cerr << "[NN STUB]   " << std::setw(3) << move_idx++ << ". " << move_uci;
                if (move_uci == "e2e4") {
                    std::cerr << " <-- TARGET";
                }
                std::cerr << std::endl;
            }
            std::cerr << "[NN STUB] ============================================" << std::endl;
        }
        
        // SIMPLIFIED: Only set policy for root position (starting position)
        // For other positions, return empty policy (MCTS will use uniform prior)
        if (is_root) {
            std::string target_move = "e2e4";  // Always try e2e4 first
            bool found_target = false;
            
            // Try to find e2e4
            for (const Move& move : legal_moves) {
                std::string move_uci = static_cast<std::string>(move.from()) + static_cast<std::string>(move.to());
                if (move.typeOf() == Move::PROMOTION) {
                    PieceType pt = move.promotionType();
                    char promo = 'q';
                    if (pt == PieceType::KNIGHT) promo = 'n';
                    else if (pt == PieceType::BISHOP) promo = 'b';
                    else if (pt == PieceType::ROOK) promo = 'r';
                    move_uci += promo;
                }
                
                if (move_uci == target_move) {
                    policy_out[move_uci] = 1.0;
                    found_target = true;
                    std::cerr << "[NN STUB] Root: Found " << target_move << ", setting policy to 1.0" << std::endl;
                    break;
                }
            }
            
            // If e2e4 not found, use first legal move (shouldn't happen in starting position)
            if (!found_target) {
                const Move& first_move = legal_moves[0];
                std::string move_uci = static_cast<std::string>(first_move.from()) + static_cast<std::string>(first_move.to());
                if (first_move.typeOf() == Move::PROMOTION) {
                    PieceType pt = first_move.promotionType();
                    char promo = 'q';
                    if (pt == PieceType::KNIGHT) promo = 'n';
                    else if (pt == PieceType::BISHOP) promo = 'b';
                    else if (pt == PieceType::ROOK) promo = 'r';
                    move_uci += promo;
                }
                policy_out[move_uci] = 1.0;
                std::cerr << "[NN STUB] Root: e2e4 not found! Using first move: " << move_uci << " with prob=1.0" << std::endl;
            }
        } else {
            // For non-root positions, return empty policy (MCTS will use uniform prior)
            // This is correct - child nodes don't need their own policy, they use the parent's
            // (The parent's policy is stored in the parent node's policy_priors)
        }
        
               // Value = +1.0 from current player's perspective (winning) in [-1, +1] range
               // Model already outputs values in [-1, +1] range
               raw_value_out = 1.0;  // Raw value from model (already in [-1, +1] range)
               value_out = 1.0;  // Same as raw (model already outputs tanh'd values)
        
        // Only log value for root node (starting position) to reduce spam
        if (fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") {
            std::cerr << "[NN STUB] Root node: Side to move: " << (board.sideToMove() == Color::WHITE ? "White" : "Black")
                      << ", value=" << value_out << " (model already outputs in [-1, +1] range)" << std::endl;
        }
        return true;
    }
    
    if (!loaded_) {
        std::cerr << "ERROR: Neural network model not loaded. Cannot predict." << std::endl;
        return false;
    }
    
    try {
        // 1. Encode board directly into pre-allocated tensor buffer (optimization)
        // Get direct pointer to tensor data
        float* tensor_data = input_tensor_buffer_.data_ptr<float>();
        
        // Clear buffer first
        std::fill(tensor_data, tensor_data + (18 * 8 * 8), 0.0f);
        
        // Encode board directly into tensor buffer (no intermediate copies)
        encode_board_into_buffer(board, tensor_data);
        
        // 3. Run model inference using pre-allocated buffer
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor_buffer_);
        
        auto outputs = model_.forward(inputs).toTuple();
        torch::Tensor policy_logits = outputs->elements()[0].toTensor();  // Shape: (1, 4096)
        torch::Tensor value_tensor = outputs->elements()[1].toTensor();   // Shape: (1,)
        
               // 4. Extract value (model already outputs values in [-1, +1] range)
               float model_value = value_tensor.item<float>();  // Value from model (already in [-1, +1] range)
               
               // Model already outputs tanh'd values, so raw and final are the same
               raw_value_out = static_cast<double>(model_value);
               value_out = static_cast<double>(model_value);
        
        // 5. Extract policy and map to legal moves
        policy_out.clear();
        
        // Get legal moves
        Movelist legal_moves;
        movegen::legalmoves(legal_moves, board);
        
        if (legal_moves.empty()) {
            std::cerr << "ERROR: No legal moves in position. Cannot predict policy." << std::endl;
            return false;
        }
        
        // Apply softmax to policy logits
        torch::Tensor policy_probs = torch::softmax(policy_logits, 1);  // Shape: (1, 4096)
        auto policy_accessor = policy_probs.accessor<float, 2>();
        
        // Sum probabilities for legal moves (for normalization)
        double total_prob = 0.0;
        for (const Move& move : legal_moves) {
            int policy_idx = move_to_policy_index(move);
            if (policy_idx >= 0 && policy_idx < 4096) {
                total_prob += policy_accessor[0][policy_idx];
            }
        }
        
        // Normalize and store policy for legal moves only
        // Convert Move to UCI string for map key
        if (total_prob > 0.0) {
            for (const Move& move : legal_moves) {
                int policy_idx = move_to_policy_index(move);
                if (policy_idx >= 0 && policy_idx < 4096) {
                    double prob = policy_accessor[0][policy_idx] / total_prob;
                    // Convert Move to UCI string for map key
                    std::string move_uci = static_cast<std::string>(move.from()) + static_cast<std::string>(move.to());
                    if (move.typeOf() == Move::PROMOTION) {
                        PieceType pt = move.promotionType();
                        char promo = 'q';
                        if (pt == PieceType::KNIGHT) promo = 'n';
                        else if (pt == PieceType::BISHOP) promo = 'b';
                        else if (pt == PieceType::ROOK) promo = 'r';
                        move_uci += promo;
                    }
                    policy_out[move_uci] = prob;
                }
            }
        } else {
            // Fallback: uniform distribution if no valid policy found
            double uniform_prob = 1.0 / legal_moves.size();
            for (const Move& move : legal_moves) {
                std::string move_uci = static_cast<std::string>(move.from()) + static_cast<std::string>(move.to());
                if (move.typeOf() == Move::PROMOTION) {
                    PieceType pt = move.promotionType();
                    char promo = 'q';
                    if (pt == PieceType::KNIGHT) promo = 'n';
                    else if (pt == PieceType::BISHOP) promo = 'b';
                    else if (pt == PieceType::ROOK) promo = 'r';
                    move_uci += promo;
                }
                policy_out[move_uci] = uniform_prob;
            }
        }
        
        if constexpr (kEnableNNLogging) {
            static std::ofstream log_file;
            static bool log_file_opened = false;

            if (!log_file_opened) {
                std::time_t now = std::time(nullptr);
                std::tm* local_time = std::localtime(&now);
                std::ostringstream filename;
                filename << "nn_inference_"
                         << std::setfill('0') << std::setw(4) << (1900 + local_time->tm_year)
                         << std::setw(2) << (local_time->tm_mon + 1)
                         << std::setw(2) << local_time->tm_mday << "_"
                         << std::setw(2) << local_time->tm_hour
                         << std::setw(2) << local_time->tm_min
                         << std::setw(2) << local_time->tm_sec << ".log";
                log_file.open(filename.str(), std::ios::app);
                log_file_opened = true;
                if (log_file.is_open()) {
                    log_file << "=== Neural Network Inference Log ===" << std::endl;
                    log_file << "Started: " << std::asctime(local_time) << std::endl;
                    log_file << "=====================================" << std::endl << std::endl;
                }
            }

            auto log_line = [&](const std::string& line) {
                std::cerr << line << std::endl;
                if (log_file.is_open()) {
                    log_file << line << std::endl;
                }
            };

            std::string fen = board.getFen();

            log_line("\n=== NN Inference Results ===");
            log_line("FEN: " + fen);
            log_line("Side to move: " + std::string(board.sideToMove() == Color::WHITE ? "White" : "Black"));

            std::ostringstream value_line1;
            value_line1 << "Value (from model, in [-1,+1]): " << std::fixed << std::setprecision(4) << value_out
                       << " (from " << (board.sideToMove() == Color::WHITE ? "White" : "Black") << "'s perspective)";
            log_line(value_line1.str());

            std::vector<std::pair<std::string, double>> policy_vec(policy_out.begin(), policy_out.end());
            std::sort(policy_vec.begin(), policy_vec.end(),
                      [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                          return a.second > b.second;
                      });

            log_line("Top 15 policy moves:");
            int top_n = std::min(15, static_cast<int>(policy_vec.size()));
            for (int i = 0; i < top_n; i++) {
                std::ostringstream move_line;
                move_line << "  " << std::setw(2) << (i+1) << ". " << std::setw(6) << policy_vec[i].first
                          << " : " << std::fixed << std::setprecision(4) << policy_vec[i].second
                          << " (" << std::fixed << std::setprecision(2) << (policy_vec[i].second * 100.0) << "%)";
                log_line(move_line.str());
            }

            if (!policy_vec.empty()) {
                double max_prob = policy_vec[0].second;
                double sum_top3 = 0.0;
                for (int i = 0; i < std::min(3, static_cast<int>(policy_vec.size())); i++) {
                    sum_top3 += policy_vec[i].second;
                }
                std::ostringstream stats_line;
                stats_line << "Policy stats: max=" << std::fixed << std::setprecision(4) << max_prob
                           << ", top3_sum=" << std::fixed << std::setprecision(4) << sum_top3
                           << ", legal_moves=" << legal_moves.size();
                log_line(stats_line.str());
            }

            log_line("============================");

            if (log_file.is_open()) {
                log_file.flush();
            }
        }
        
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "ERROR: Neural network inference failed (PyTorch error): " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: Neural network inference failed (exception): " << e.what() << std::endl;
        return false;
    }
}

