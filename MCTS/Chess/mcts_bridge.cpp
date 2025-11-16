/**
 * MCTS Bridge: Simple program to get move from FEN
 * Usage (one-shot): ./mcts_bridge <model_path> <fen> [max_iterations] [max_seconds] [cpuct]
 * Usage (persistent): ./mcts_bridge --persistent <model_path> [max_iterations] [max_seconds] [cpuct]
 *   Then send FEN lines to stdin, receive moves from stdout
 * Outputs: UCI move string (e.g., "e2e4")
 */

#include "Chess.h"
#include "../include/mcts.h"
#include "../include/neural_network.h"
#include "../include/chess.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;
using namespace chess;

int process_move(NeuralNetwork* nn, bool nn_loaded, const string& fen, int max_iterations, int max_seconds, double cpuct) {
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
        return 1;
    }
    
    // Output move in UCI format (ONLY this goes to stdout)
    const Chess_move* chess_move = static_cast<const Chess_move*>(engine_move);
    string move_str = chess_move->sprint();
    
    // Output to stdout (this is what Python will read)
    cout << move_str << endl;
    cout.flush();  // Ensure immediate output
    
    // Cleanup
    delete engine;
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Redirect cout to cerr so debug output doesn't interfere with move output
    // We'll restore it before outputting the move
    streambuf* cout_backup = cout.rdbuf();
    cout.rdbuf(cerr.rdbuf());
    
    bool persistent_mode = false;
    int arg_offset = 0;
    
    // Check for persistent mode flag
    if (argc >= 2 && string(argv[1]) == "--persistent") {
        persistent_mode = true;
        arg_offset = 1;
    }
    
    // Parse arguments
    if (argc < 3 + arg_offset) {
        cerr << "Usage (one-shot): " << argv[0] << " <model_path> <fen> [max_iterations] [max_seconds] [cpuct]" << endl;
        cerr << "Usage (persistent): " << argv[0] << " --persistent <model_path> [max_iterations] [max_seconds] [cpuct]" << endl;
        cerr << "Example: " << argv[0] << " ../aznet_traced.pt \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\"" << endl;
        return 1;
    }
    
    string model_path = argv[1 + arg_offset];
    
    // Default MCTS parameters
    int max_iterations = 15000;
    int max_seconds = 5;
    double cpuct = 2.0;
    
    if (argc >= 4 + arg_offset) {
        max_iterations = stoi(argv[3 + arg_offset]);
    }
    if (argc >= 5 + arg_offset) {
        max_seconds = stoi(argv[4 + arg_offset]);
    }
    if (argc >= 6 + arg_offset) {
        cpuct = stod(argv[5 + arg_offset]);
    }
    
    // Load neural network ONCE (reused in persistent mode)
    NeuralNetwork* nn = new NeuralNetwork();
    bool nn_loaded = false;
    if (!model_path.empty() && model_path != "none") {
        cerr << "[BRIDGE] Loading model (this happens once)..." << endl;
        nn_loaded = nn->load_model(model_path);
        if (!nn_loaded) {
            cerr << "ERROR: Failed to load neural network from: " << model_path << endl;
            delete nn;
            return 1;
        }
        cerr << "[BRIDGE] Model loaded successfully!" << endl;
    }
    
    // Restore cout to stdout for move output
    cout.rdbuf(cout_backup);
    
    if (persistent_mode) {
        // Persistent mode: read FEN from stdin, output moves to stdout
        cerr << "[BRIDGE] Persistent mode: ready to process FEN positions" << endl;
        string fen;
        while (getline(cin, fen)) {
            // Trim whitespace
            fen.erase(0, fen.find_first_not_of(" \t\n\r"));
            fen.erase(fen.find_last_not_of(" \t\n\r") + 1);
            
            // Check for quit command
            if (fen == "quit" || fen == "exit") {
                break;
            }
            
            if (fen.empty()) {
                continue;
            }
            
            int result = process_move(nn, nn_loaded, fen, max_iterations, max_seconds, cpuct);
            if (result != 0) {
                cerr << "[BRIDGE] Error processing move" << endl;
            }
        }
    } else {
        // One-shot mode: process single FEN from command line
        string fen = argv[2 + arg_offset];
        int result = process_move(nn, nn_loaded, fen, max_iterations, max_seconds, cpuct);
        if (result != 0) {
            delete nn;
            return result;
        }
    }
    
    // Cleanup
    delete nn;
    
    return 0;
}

