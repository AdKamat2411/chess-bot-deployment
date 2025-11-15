/**
 * Simple neural network inference test
 * Usage: ./nn_test <model_path> [fen_string]
 * 
 * If FEN is not provided, uses starting position
 * Outputs: value and top policy moves
 */

#include "Chess.h"
#include "../include/neural_network.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;
using namespace chess;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <model_path> [fen_string]" << endl;
        cerr << "Example: " << argv[0] << " ../aznet_traced.pt" << endl;
        cerr << "Example: " << argv[0] << " ../aznet_traced.pt \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\"" << endl;
        return 1;
    }
    
    string model_path = argv[1];
    string fen;
    
    if (argc >= 3) {
        fen = argv[2];
    } else {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        cout << "No FEN provided, using starting position" << endl;
    }
    
    cout << "=== Neural Network Inference Test ===" << endl;
    cout << "Model: " << model_path << endl;
    cout << "FEN: " << fen << endl;
    cout << "=====================================" << endl << endl;
    
    // Load model
    NeuralNetwork* nn = new NeuralNetwork();
    if (!nn->load_model(model_path)) {
        cerr << "ERROR: Failed to load model from: " << model_path << endl;
        delete nn;
        return 1;
    }
    
    // Create board from FEN
    Board board(fen);
    cout << "Board:" << endl;
    cout << board << endl;
    cout << "Side to move: " << (board.sideToMove() == Color::WHITE ? "White" : "Black") << endl;
    cout << endl;
    
    // Run inference
    map<string, double> policy;
    double value;
    double raw_value;
    
    if (!nn->predict(board, policy, value, raw_value)) {
        cerr << "ERROR: Neural network prediction failed!" << endl;
        delete nn;
        return 1;
    }
    
    // Output value
    cout << "=== VALUE HEAD ===" << endl;
    cout << "Value (from model, [-1, +1]): " << fixed << setprecision(6) << value << endl;
    cout << "Raw value (same as value, model already outputs tanh'd): " << fixed << setprecision(6) << raw_value << endl;
    cout << "Value (from current player's perspective): " << fixed << setprecision(6) << value << endl;
    if (board.sideToMove() == Color::WHITE) {
        cout << "  White's perspective: " << value << " (1.0 = white winning, -1.0 = black winning)" << endl;
    } else {
        cout << "  Black's perspective: " << value << " (1.0 = black winning, -1.0 = white winning)" << endl;
    }
    cout << endl;
    
    // Output policy
    cout << "=== POLICY HEAD ===" << endl;
    cout << "Total legal moves: " << policy.size() << endl;
    cout << endl;
    
    // Sort policy by probability
    vector<pair<string, double>> policy_vec(policy.begin(), policy.end());
    sort(policy_vec.begin(), policy_vec.end(),
         [](const pair<string, double>& a, const pair<string, double>& b) {
             return a.second > b.second;
         });
    
    cout << "Top 20 moves:" << endl;
    cout << setw(6) << "Move" << " | " << setw(10) << "Probability" << " | " << setw(8) << "Percent" << endl;
    cout << "------|------------|----------" << endl;
    
    int top_n = min(20, static_cast<int>(policy_vec.size()));
    for (int i = 0; i < top_n; i++) {
        cout << setw(6) << policy_vec[i].first << " | "
             << setw(10) << fixed << setprecision(6) << policy_vec[i].second << " | "
             << setw(7) << fixed << setprecision(2) << (policy_vec[i].second * 100.0) << "%" << endl;
    }
    
    // Policy statistics
    if (!policy_vec.empty()) {
        double sum_all = 0.0;
        double sum_top3 = 0.0;
        double sum_top5 = 0.0;
        for (size_t i = 0; i < policy_vec.size(); i++) {
            sum_all += policy_vec[i].second;
            if (i < 3) sum_top3 += policy_vec[i].second;
            if (i < 5) sum_top5 += policy_vec[i].second;
        }
        
        cout << endl;
        cout << "Policy statistics:" << endl;
        cout << "  Max probability: " << fixed << setprecision(6) << policy_vec[0].second << endl;
        cout << "  Top 3 sum: " << fixed << setprecision(6) << sum_top3 << endl;
        cout << "  Top 5 sum: " << fixed << setprecision(6) << sum_top5 << endl;
        cout << "  Total sum: " << fixed << setprecision(6) << sum_all << " (should be ~1.0)" << endl;
    }
    
    // Output full policy vector (optional, can be commented out for large outputs)
    cout << endl;
    cout << "=== FULL POLICY VECTOR ===" << endl;
    cout << "Move | Probability" << endl;
    cout << "-----|------------" << endl;
    for (const auto& [move, prob] : policy_vec) {
        cout << setw(5) << move << " | " << fixed << setprecision(6) << prob << endl;
    }
    
    delete nn;
    return 0;
}

