/**
 * MCTS Bridge: Simple program to get move from FEN
 * Usage: ./mcts_bridge <model_path> <fen> [max_iterations] [max_seconds] [cpuct]
 * Outputs: UCI move string (e.g., "e2e4")
 */

#include "Chess.h"
#include "../include/mcts.h"
#include "../include/neural_network.h"
#include "../include/chess.hpp"
#include <iostream>
#include <string>
#include <fstream>

using namespace std;
using namespace chess;

int main(int argc, char* argv[]) {
    // Redirect cout to cerr so debug output doesn't interfere with move output
    // We'll restore it before outputting the move
    streambuf* cout_backup = cout.rdbuf();
    cout.rdbuf(cerr.rdbuf());
    // Parse arguments
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <model_path> <fen> [max_iterations] [max_seconds] [cpuct]" << endl;
        cerr << "Example: " << argv[0] << " ../aznet_traced.pt \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\"" << endl;
        return 1;
    }
    
    string model_path = argv[1];
    string fen = argv[2];
    
    // Default MCTS parameters
    int max_iterations = 15000;
    int max_seconds = 5;
    double cpuct = 2.0;
    
    if (argc >= 4) {
        max_iterations = stoi(argv[3]);
    }
    if (argc >= 5) {
        max_seconds = stoi(argv[4]);
    }
    if (argc >= 6) {
        cpuct = stod(argv[5]);
    }
    
    // Redirect all debug/info output to stderr, only move goes to stdout
    // This allows Python to cleanly capture just the move
    
    // Load neural network
    NeuralNetwork* nn = new NeuralNetwork();
    bool nn_loaded = false;
    if (!model_path.empty() && model_path != "none") {
        nn_loaded = nn->load_model(model_path);
        if (!nn_loaded) {
            cerr << "ERROR: Failed to load neural network from: " << model_path << endl;
            delete nn;
            return 1;
        }
        // Model loaded (removed debug output for performance)
    }
    
    // Create initial state from FEN
    Chess_state* initial_state = new Chess_state(fen);
    
    // Create engine agent
    MCTS_agent* engine = new MCTS_agent(
        initial_state,
        max_iterations,
        max_seconds,
        nn_loaded ? nn : nullptr,
        cpuct
    );
    
    // Generate move
    const MCTS_move* engine_move = engine->genmove(nullptr);
    
    if (!engine_move) {
        cerr << "ERROR: Engine returned no move!" << endl;
        delete engine;
        delete nn;
        return 1;
    }
    
    // Restore cout to stdout for the move output
    cout.rdbuf(cout_backup);
    
    // Output move in UCI format (ONLY this goes to stdout)
    const Chess_move* chess_move = static_cast<const Chess_move*>(engine_move);
    string move_str = chess_move->sprint();
    
    // Output to stdout (this is what Python will read)
    cout << move_str << endl;
    
    // Cleanup
    delete engine;
    delete nn;
    
    return 0;
}

