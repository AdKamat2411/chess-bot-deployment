/**
 * Interactive chess game: Play against MCTS engine
 * Input moves via terminal, engine responds with MCTS search
 * Shows move statistics, FEN, and PGN export
 */

#include "Chess.h"
#include "../include/mcts.h"
#include "../include/neural_network.h"
#include "../include/chess.hpp"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace chess;

struct MoveStats {
    string move;
    string san_move;
    unsigned int tree_size;
    double time_seconds;
    unsigned int iterations;
    string player;
};

// Convert UCI string to chess::Move
Move parse_uci_move(const Board& board, const string& uci_str) {
    if (uci_str.length() < 4 || uci_str.length() > 5) {
        return Move::NO_MOVE;
    }
    
    // Parse from and to squares
    string from_str = uci_str.substr(0, 2);
    string to_str = uci_str.substr(2, 2);
    
    try {
        Square from(from_str);
        Square to(to_str);
        
        // Check if it's a promotion
        if (uci_str.length() == 5) {
            char promo_char = tolower(uci_str[4]);
            PieceType promo_type = PieceType::QUEEN;
            if (promo_char == 'n') promo_type = PieceType::KNIGHT;
            else if (promo_char == 'b') promo_type = PieceType::BISHOP;
            else if (promo_char == 'r') promo_type = PieceType::ROOK;
            
            return Move::make<Move::PROMOTION>(from, to, promo_type);
        }
        
        // Check if it's castling
        Movelist legal_moves;
        movegen::legalmoves(legal_moves, board);
        
        for (const Move& m : legal_moves) {
            if (m.from() == from && m.to() == to) {
                return m;
            }
        }
        
        return Move::NO_MOVE;
    } catch (...) {
        return Move::NO_MOVE;
    }
}

// Get user input move
string get_user_move(const Board& board) {
    string input;
    cout << "\nYour move (UCI format, e.g., 'e2e4' or 'quit' to exit): ";
    getline(cin, input);
    
    // Trim whitespace
    input.erase(0, input.find_first_not_of(" \t\n\r"));
    input.erase(input.find_last_not_of(" \t\n\r") + 1);
    
    // Convert to lowercase
    transform(input.begin(), input.end(), input.begin(), ::tolower);
    
    return input;
}

// Note: This function is no longer used, but kept for reference
// Move validation is now done inline in the main loop

int main(int argc, char* argv[]) {
    // Configuration
    const int MAX_ITERATIONS = 5000;
    const int MAX_SECONDS = 2;
    const int MAX_MOVES = 1000;
    const double CPUCT = 1.5;
    
    // Neural network model path
    string model_path = "../aznet_traced.pt";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    cout << "=== MCTS Chess Engine - Interactive Mode ===" << endl;
    cout << "Config: " << MAX_ITERATIONS << " iterations or " << MAX_SECONDS << " seconds per move" << endl;
    cout << "CPUCT: " << CPUCT << endl;
    cout << "=============================================" << endl << endl;
    
    // Load neural network
    NeuralNetwork* nn = new NeuralNetwork();
    bool nn_loaded = false;
    if (!model_path.empty()) {
        cout << "Loading neural network from: " << model_path << endl;
        nn_loaded = nn->load_model(model_path);
        if (!nn_loaded) {
            cout << "Warning: Failed to load neural network. Falling back to heuristic rollouts." << endl;
        } else {
            cout << "Neural network loaded successfully!" << endl;
        }
    }
    cout << endl;
    
    // Ask for starting position (FEN) or use default
    cout << "Enter starting FEN (or press Enter for default starting position): ";
    string fen_input;
    getline(cin, fen_input);
    
    // Trim whitespace
    fen_input.erase(0, fen_input.find_first_not_of(" \t\n\r"));
    fen_input.erase(fen_input.find_last_not_of(" \t\n\r") + 1);
    
    string starting_fen;
    if (fen_input.empty()) {
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        cout << "Using default starting position." << endl;
    } else {
        starting_fen = fen_input;
        cout << "Using FEN: " << starting_fen << endl;
    }
    
    // Ask user which color they want to play
    cout << "\nChoose your color:" << endl;
    cout << "1. White (you move first)" << endl;
    cout << "2. Black (engine moves first)" << endl;
    cout << "Enter choice (1 or 2): ";
    
    string choice;
    getline(cin, choice);
    bool user_plays_white = (choice == "1" || choice == "w" || choice == "white");
    
    // Create initial state from FEN
    Chess_state* initial_state = new Chess_state(starting_fen);
    Board current_board(initial_state->get_board().getFen());
    
    // Create engine agent
    Chess_state* engine_state = new Chess_state(current_board.getFen());
    MCTS_agent* engine = new MCTS_agent(
        engine_state,
        MAX_ITERATIONS,
        MAX_SECONDS,
        nn_loaded ? nn : nullptr,
        CPUCT
    );
    
    vector<MoveStats> move_history;
    int move_number = 1;
    bool white_turn = true;
    
    cout << "\nMove | Player | Move    | Tree Size | Time (s) | Positions/s" << endl;
    cout << "-----|--------|---------|-----------|----------|------------" << endl;
    
    // Store initial FEN
    string initial_fen = current_board.getFen();
    cout << "Initial FEN: " << initial_fen << endl << endl;
    
    // If user plays black, engine moves first
    if (!user_plays_white) {
        cout << "Engine (White) is thinking..." << endl;
        
        auto start_time = chrono::high_resolution_clock::now();
        const MCTS_move* engine_move = engine->genmove(nullptr);
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end_time - start_time).count();
        
        if (!engine_move) {
            cout << "Error: Engine returned no move!" << endl;
            delete engine;
            delete nn;
            return 1;
        }
        
        const Chess_move* chess_move = static_cast<const Chess_move*>(engine_move);
        string move_str = chess_move->sprint();
        string san_move = chess::uci::moveToSan(current_board, chess_move->move);
        
        // Apply move to board
        current_board.makeMove(chess_move->move);
        
        // Update engine state
        Chess_state* new_state = new Chess_state(current_board.getFen());
        delete engine;
        engine = new MCTS_agent(new_state, MAX_ITERATIONS, MAX_SECONDS, 
                               nn_loaded ? nn : nullptr, CPUCT);
        
        MoveStats stats;
        stats.move = move_str;
        stats.san_move = san_move;
        stats.tree_size = MAX_ITERATIONS;  // Approximate
        stats.time_seconds = elapsed;
        stats.iterations = MAX_ITERATIONS;
        stats.player = "Engine";
        move_history.push_back(stats);
        
        cout << setw(4) << move_number << " | "
             << setw(6) << "Engine" << " | "
             << setw(7) << move_str << " | "
             << setw(9) << MAX_ITERATIONS << " | "
             << setw(8) << fixed << setprecision(3) << elapsed << " | "
             << setw(10) << fixed << setprecision(0) << (MAX_ITERATIONS / elapsed) << endl;
        
        cout << "     FEN: " << current_board.getFen() << endl;
        cout << "     Last move: " << san_move << " (" << move_str << ")" << endl;
        
        white_turn = false;
        move_number++;
    }
    
    // Main game loop
    while (move_number <= MAX_MOVES) {
        // Check if game is over
        auto [reason, result] = current_board.isGameOver();
        if (result != GameResult::NONE) {
            cout << "\n=== Game Over ===" << endl;
            if (result == GameResult::DRAW) {
                cout << "Result: Draw" << endl;
            } else if (result == GameResult::WIN) {
                cout << "Result: " << (current_board.sideToMove() == Color::WHITE ? "White" : "Black") << " wins!" << endl;
            } else {
                cout << "Result: " << (current_board.sideToMove() == Color::WHITE ? "Black" : "White") << " wins!" << endl;
            }
            break;
        }
        
        // Determine whose turn it is based on actual board state, not just white_turn flag
        // This is more reliable since the board state is the source of truth
        bool is_user_turn = (current_board.sideToMove() == Color::WHITE && user_plays_white) ||
                           (current_board.sideToMove() == Color::BLACK && !user_plays_white);
        
        if (is_user_turn) {
            // User's turn
            cout << "\n--- Your turn ---" << endl;
            cout << current_board << endl;
            cout << "FEN: " << current_board.getFen() << endl;
            
            string user_input;
            Move user_move_obj;
            bool valid_move = false;
            
            while (!valid_move) {
                user_input = get_user_move(current_board);
                if (user_input == "quit" || user_input == "q" || user_input == "exit") {
                    cout << "Game ended by user." << endl;
                    goto game_end;
                }
                
                // Parse move first to validate, but don't apply yet
                Move temp_move = parse_uci_move(current_board, user_input);
                if (temp_move == Move::NO_MOVE) {
                    cout << "Invalid move format. Use UCI notation (e.g., 'e2e4', 'e7e8q' for promotion)" << endl;
                    continue;
                }
                
                // Check if move is legal
                Movelist legal_moves;
                movegen::legalmoves(legal_moves, current_board);
                bool is_legal = false;
                for (const Move& m : legal_moves) {
                    if (m == temp_move) {
                        is_legal = true;
                        user_move_obj = temp_move;
                        break;
                    }
                }
                
                if (!is_legal) {
                    cout << "Illegal move! Please try again." << endl;
                    continue;
                }
                
                valid_move = true;
            }
            
            // Convert to SAN BEFORE applying the move (board state needed for SAN conversion)
            string san_move;
            try {
                san_move = chess::uci::moveToSan(current_board, user_move_obj);
            } catch (...) {
                san_move = user_input;  // Fallback to UCI if SAN conversion fails
            }
            
            // Convert to Chess_move for engine (before applying move to board)
            Chess_move* chess_move = new Chess_move(user_move_obj);
            string move_str = chess_move->sprint();
            
            // Update engine with user's move (engine will advance its tree internally)
            const MCTS_move* engine_move = engine->genmove(chess_move);
            delete chess_move;
            
            // Now sync current_board with engine's state (engine has already applied the move)
            const MCTS_state* engine_state = engine->get_current_state();
            const Chess_state* engine_chess_state = static_cast<const Chess_state*>(engine_state);
            current_board = engine_chess_state->get_board();  // Sync board with engine
            
            MoveStats stats;
            stats.move = move_str;
            stats.san_move = san_move;
            stats.tree_size = 0;  // User move, no tree search
            stats.time_seconds = 0.0;
            stats.iterations = 0;
            stats.player = "You";
            move_history.push_back(stats);
            
            cout << setw(4) << move_number << " | "
                 << setw(6) << "You" << " | "
                 << setw(7) << move_str << " | "
                 << setw(9) << "-" << " | "
                 << setw(8) << "-" << " | "
                 << setw(10) << "-" << endl;
            
            cout << "     FEN: " << current_board.getFen() << endl;
            cout << "     Last move: " << san_move << " (" << move_str << ")" << endl;
            
        } else {
            // Engine's turn
            cout << "\n--- Engine thinking... ---" << endl;
            
            auto start_time = chrono::high_resolution_clock::now();
            const MCTS_move* engine_move = engine->genmove(nullptr);
            auto end_time = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(end_time - start_time).count();
            
            if (!engine_move) {
                cout << "Error: Engine returned no move!" << endl;
                break;
            }
            
            // Debug: Print root children information
            MCTS_tree* tree = engine->get_tree();
            MCTS_node* root = tree->get_root();
            if (root && root->get_children()) {
                cout << "\n=== Root Children Debug ===" << endl;
                for (auto *c : *root->get_children()) {
                    string uci = c->get_move() ? c->get_move()->sprint() : "NULL";
                    const Chess_state* chess_state = dynamic_cast<const Chess_state*>(c->get_current_state());
                    bool is_capture = false;
                    int captured_value = 0;
                    if (chess_state) {
                        is_capture = chess_state->was_capture();
                        captured_value = chess_state->captured_piece_value();
                    }
                    bool is_terminal = c->is_terminal();
                    double raw_nn = c->get_raw_nn_value();
                    double node_v = c->get_nn_value();
                    double Q = (c->get_number_of_simulations() > 0) ? c->get_score() / (double)c->get_number_of_simulations() : 0.0;
                    double P = root->get_prior(c->get_move());
                    cout << uci << "  cap=" << is_capture << " cap_val=" << captured_value 
                         << " term=" << is_terminal << " raw_nn=" << raw_nn << " nn_val=" << node_v
                         << " P=" << P << " Q=" << Q << " N=" << c->get_number_of_simulations() << endl;
                }
                cout << "===========================" << endl;
            }
            
            const Chess_move* chess_move = static_cast<const Chess_move*>(engine_move);
            string move_str = chess_move->sprint();
            
            // Convert to SAN BEFORE engine advances tree (need current board state)
            string san_move;
            try {
                san_move = chess::uci::moveToSan(current_board, chess_move->move);
            } catch (...) {
                san_move = move_str;  // Fallback to UCI
            }
            
            // Note: Engine has already advanced its tree (genmove calls advance_tree internally)
            // We'll sync current_board after printing stats
            
            unsigned int estimated_positions = (elapsed >= MAX_SECONDS) ?
                static_cast<unsigned int>(MAX_ITERATIONS * (elapsed / MAX_SECONDS)) :
                MAX_ITERATIONS;
            double positions_per_sec = (elapsed > 0) ? estimated_positions / elapsed : 0;
            
            MoveStats stats;
            stats.move = move_str;
            stats.san_move = san_move;
            stats.tree_size = estimated_positions;
            stats.time_seconds = elapsed;
            stats.iterations = estimated_positions;
            stats.player = "Engine";
            move_history.push_back(stats);
            
            cout << setw(4) << move_number << " | "
                 << setw(6) << "Engine" << " | "
                 << setw(7) << move_str << " | "
                 << setw(9) << estimated_positions << " | "
                 << setw(8) << fixed << setprecision(3) << elapsed << " | "
                 << setw(10) << fixed << setprecision(0) << positions_per_sec << endl;
            
            // Sync board AFTER engine move (engine has already advanced its tree)
            const MCTS_state* engine_state_after = engine->get_current_state();
            const Chess_state* engine_chess_state_after = static_cast<const Chess_state*>(engine_state_after);
            current_board = engine_chess_state_after->get_board();
            
            cout << "     FEN: " << current_board.getFen() << endl;
            cout << "     Last move: " << san_move << " (" << move_str << ")" << endl;
        }
        
        white_turn = !white_turn;
        move_number++;
    }
    
    if (move_number > MAX_MOVES) {
        cout << "\n=== Game stopped: Maximum moves reached ===" << endl;
    }
    
game_end:
    // Print summary statistics
    cout << "\n=== Game Summary ===" << endl;
    cout << "Total moves: " << move_history.size() << endl;
    
    double total_time = 0;
    unsigned long long total_positions = 0;
    for (const auto& stats : move_history) {
        total_time += stats.time_seconds;
        total_positions += stats.iterations;
    }
    
    cout << "Total time: " << fixed << setprecision(2) << total_time << " seconds" << endl;
    cout << "Total positions evaluated: " << total_positions << endl;
    if (total_time > 0) {
        cout << "Average positions/second: " << fixed << setprecision(0) << (total_positions / total_time) << endl;
    }
    if (!move_history.empty()) {
        cout << "Average time per move: " << fixed << setprecision(3) << (total_time / move_history.size()) << " seconds" << endl;
    }
    
    // Generate and output PGN
    cout << "\n=== PGN ===" << endl;
    cout << "[Event \"Human vs MCTS Engine\"]" << endl;
    cout << "[Site \"Computer\"]" << endl;
    
    time_t now = time(0);
    tm* local_time = localtime(&now);
    char date_str[11];
    strftime(date_str, sizeof(date_str), "%Y.%m.%d", local_time);
    cout << "[Date \"" << date_str << "\"]" << endl;
    
    cout << "[Round \"1\"]" << endl;
    cout << "[White \"" << (user_plays_white ? "Human" : "MCTS Engine") << "\"]" << endl;
    cout << "[Black \"" << (user_plays_white ? "MCTS Engine" : "Human") << "\"]" << endl;
    
    // Determine result
    string result = "*";
    auto [reason, game_result] = current_board.isGameOver();
    if (game_result != GameResult::NONE) {
        if (game_result == GameResult::DRAW) {
            result = "1/2-1/2";
        } else if (game_result == GameResult::WIN) {
            result = (current_board.sideToMove() == Color::WHITE) ? "0-1" : "1-0";
        } else if (game_result == GameResult::LOSE) {
            result = (current_board.sideToMove() == Color::WHITE) ? "1-0" : "0-1";
        }
    }
    cout << "[Result \"" << result << "\"]" << endl;
    cout << "[FEN \"" << initial_fen << "\"]" << endl;
    cout << endl;
    
    // Output moves in PGN format
    Board pgn_board(initial_fen);
    int move_count = 1;
    string pgn_moves;
    
    for (size_t i = 0; i < move_history.size(); i++) {
        const auto& stats = move_history[i];
        
        // Convert UCI move to Move object and then to SAN
        Movelist movelist;
        movegen::legalmoves(movelist, pgn_board);
        Move actual_move = Move::NO_MOVE;
        
        for (const Move& m : movelist) {
            string m_uci = static_cast<string>(m.from()) + static_cast<string>(m.to());
            if (m.typeOf() == Move::PROMOTION) {
                PieceType pt = m.promotionType();
                char promo = 'q';
                if (pt == PieceType::KNIGHT) promo = 'n';
                else if (pt == PieceType::BISHOP) promo = 'b';
                else if (pt == PieceType::ROOK) promo = 'r';
                m_uci += promo;
            }
            if (m_uci == stats.move) {
                actual_move = m;
                break;
            }
        }
        
        if (actual_move != Move::NO_MOVE) {
            string san_move;
            try {
                san_move = chess::uci::moveToSan(pgn_board, actual_move);
            } catch (...) {
                san_move = stats.move;
            }
            
            bool is_white_move = (i == 0 && user_plays_white) || 
                                (i > 0 && move_history[i-1].player == "Engine" && user_plays_white) ||
                                (i > 0 && move_history[i-1].player == "You" && !user_plays_white);
            
            if (is_white_move) {
                pgn_moves += to_string(move_count) + ". " + san_move + " ";
            } else {
                pgn_moves += san_move + " ";
                move_count++;
            }
            
            pgn_board.makeMove(actual_move);
        } else {
            // Fallback
            bool is_white_move = (i == 0 && user_plays_white);
            if (is_white_move) {
                pgn_moves += to_string(move_count) + ". " + stats.move + " ";
            } else {
                pgn_moves += stats.move + " ";
                move_count++;
            }
        }
        
        if ((i + 1) % 20 == 0) {
            pgn_moves += "\n";
        }
    }
    
    cout << pgn_moves << result << endl;
    
    // Cleanup
    delete engine;
    delete nn;
    
    return 0;
}

