/**
 * Simple neural network value output
 * Usage: ./nn_value <model_path> <fen>
 * Outputs: Only the value (from current player's perspective, [-1, +1])
 */

#include "Chess.h"
#include "../include/neural_network.h"
#include "../include/chess.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace chess;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <model_path> <fen>" << endl;
        cerr << "Example: " << argv[0] << " ../aznet_traced.pt \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\"" << endl;
        return 1;
    }
    
    string model_path = argv[1];
    string fen = argv[2];
    
    // Redirect all output to stderr except the value
    streambuf* cout_backup = cout.rdbuf();
    cout.rdbuf(cerr.rdbuf());
    
    // Load model
    NeuralNetwork* nn = new NeuralNetwork();
    if (!nn->load_model(model_path)) {
        cerr << "ERROR: Failed to load model from: " << model_path << endl;
        delete nn;
        return 1;
    }
    
    // Create board from FEN
    Board board(fen);
    
    // Run inference
    map<string, double> policy;
    double value;
    double raw_value;
    
    if (!nn->predict(board, policy, value, raw_value)) {
        cerr << "ERROR: Neural network prediction failed!" << endl;
        delete nn;
        return 1;
    }
    
    // Restore cout and output ONLY the value to stdout
    cout.rdbuf(cout_backup);
    cout << fixed << value << endl;
    
    delete nn;
    return 0;
}

