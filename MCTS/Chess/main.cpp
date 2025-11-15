/**
 * Two MCTS agents playing chess against each other
 * Shows MCTS statistics: tree size, positions evaluated, time per move, etc.
 */

#include "Chess.h"
#include "../include/mcts.h"
#include "../include/neural_network.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <string>

using namespace std;
using namespace chess;

struct MoveStats {
    string move;
    string san_move;  // SAN notation for PGN
    unsigned int tree_size;
    double time_seconds;
    int iterations;
    string player;
};

int main(int argc, char* argv[]) {
    // Configuration
    const int MAX_ITERATIONS = 5000;  // Reduced for faster games
    const int MAX_SECONDS = 2;         // Reduced for faster games
    const int MAX_MOVES = 1000;  // Safety limit (game will stop naturally at checkmate/draw before this)
    const double CPUCT = 1.0;  // PUCT exploration constant
    
    // Neural network model path (default or from command line)
    string model_path = "../aznet_traced.pt";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    cout << "=== MCTS Chess Engine Self-Play ===" << endl;
    cout << "Config: " << MAX_ITERATIONS << " iterations or " << MAX_SECONDS << " seconds per move" << endl;
    cout << "CPUCT: " << CPUCT << endl;
    cout << "====================================" << endl << endl;
    
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
    } else {
        cout << "No model path provided. Using heuristic rollouts." << endl;
    }
    cout << endl;
    
    // Create initial state
    Chess_state* initial_state = new Chess_state();
    
    // Create first agent (white) - pass NN if loaded
    MCTS_agent* white_agent = new MCTS_agent(
        initial_state, 
        MAX_ITERATIONS, 
        MAX_SECONDS,
        nn_loaded ? nn : nullptr,
        CPUCT
    );
    
    // For black, we'll create it after white's first move
    MCTS_agent* black_agent = nullptr;
    
    vector<MoveStats> move_history;
    int move_number = 1;
    bool white_turn = true;
    
    cout << "Move | Player | Move    | Tree Size | Time (s) | Positions/s" << endl;
    cout << "-----|--------|---------|-----------|----------|------------" << endl;
    
    // Store initial FEN as string (board will be modified during game)
    const MCTS_state* start_state = white_agent->get_current_state();
    const Chess_state* initial_chess_state = static_cast<const Chess_state*>(start_state);
    string initial_fen = initial_chess_state->get_board().getFen();
    
    // Output initial FEN
    cout << "Initial FEN: " << initial_fen << endl << endl;
    
    const MCTS_move* last_move = nullptr;
    
    while (move_number <= MAX_MOVES) {
        MCTS_agent* current_agent = white_turn ? white_agent : black_agent;
        string player_name = white_turn ? "White" : "Black";
        
        // Create black agent after white's first move
        bool is_black_first_move = false;
            if (!black_agent && !white_turn && last_move) {
                // Create black agent from the position after white's move
                const MCTS_state* current_state = white_agent->get_current_state();
                Chess_state* black_start = new Chess_state(*static_cast<const Chess_state*>(current_state));
                black_agent = new MCTS_agent(
                    black_start, 
                    MAX_ITERATIONS, 
                    MAX_SECONDS,
                    nn_loaded ? nn : nullptr,
                    CPUCT
                );
                current_agent = black_agent;
                is_black_first_move = true;
            }
        
        // Get current state to check if game is over
        const MCTS_state* current_state = current_agent->get_current_state();
        const Chess_state* chess_state = static_cast<const Chess_state*>(current_state);
        
        if (chess_state->is_terminal()) {
            auto [reason, result] = chess_state->get_board().isGameOver();
            cout << "\n=== Game Over ===" << endl;
            if (result == GameResult::DRAW) {
                cout << "Result: Draw" << endl;
            } else if (result == GameResult::WIN) {
                cout << "Result: " << (chess_state->get_board().sideToMove() == Color::WHITE ? "White" : "Black") << " wins!" << endl;
            } else {
                cout << "Result: " << (chess_state->get_board().sideToMove() == Color::WHITE ? "Black" : "White") << " wins!" << endl;
            }
            break;
        }
        
        // Measure time for this move
        auto start_time = chrono::high_resolution_clock::now();
        
        // Check if position is terminal BEFORE calling genmove
        // (genmove will return NULL if terminal, but we want to handle it properly)
        const MCTS_state* pre_move_state = current_agent->get_current_state();
        const Chess_state* pre_move_chess_state = static_cast<const Chess_state*>(pre_move_state);
        if (pre_move_chess_state->is_terminal()) {
            auto [reason, result] = pre_move_chess_state->get_board().isGameOver();
            cout << "\n=== Game Over ===" << endl;
            if (result == GameResult::DRAW) {
                cout << "Result: Draw" << endl;
            } else if (result == GameResult::WIN) {
                cout << "Result: " << (pre_move_chess_state->get_board().sideToMove() == Color::WHITE ? "White" : "Black") << " wins!" << endl;
            } else {
                cout << "Result: " << (pre_move_chess_state->get_board().sideToMove() == Color::WHITE ? "Black" : "White") << " wins!" << endl;
            }
            cout << "Final FEN: " << pre_move_chess_state->get_board().getFen() << endl;
            break;
        }
        
        // Get move from agent
        // Don't pass last_move on black's first turn (tree is already at correct position)
        const MCTS_move* move = current_agent->genmove(is_black_first_move ? nullptr : last_move);
        
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end_time - start_time).count();
        
        if (!move) {
            cout << "\nError: No move returned (unexpected terminal state)!" << endl;
            // Debug: Check why
            const MCTS_state* debug_state = current_agent->get_current_state();
            const Chess_state* debug_chess_state = static_cast<const Chess_state*>(debug_state);
            auto [reason, result] = debug_chess_state->get_board().isGameOver();
            cout << "  Position FEN: " << debug_chess_state->get_board().getFen() << endl;
            cout << "  Is terminal: " << debug_chess_state->is_terminal() << endl;
            cout << "  Game result: " << (int)result << " (0=NONE, 1=WIN, 2=LOSE, 3=DRAW)" << endl;
            cout << "  Reason: " << (int)reason << endl;
            break;
        }
        
        const Chess_move* chess_move = static_cast<const Chess_move*>(move);
        string move_str = chess_move->sprint();
        
        // Get SAN notation for PGN
        // Note: We'll compute SAN during PGN generation to avoid issues
        string san_move = move_str;  // Temporary: use UCI, will convert to SAN later
        
        // Estimate positions evaluated (iterations done)
        // The agent runs for MAX_SECONDS or until MAX_ITERATIONS
        unsigned int estimated_positions;
        if (elapsed >= MAX_SECONDS) {
            // Hit time limit - estimate based on actual time
            estimated_positions = static_cast<unsigned int>(MAX_ITERATIONS * (elapsed / MAX_SECONDS));
        } else {
            // Hit iteration limit - use max iterations
            estimated_positions = MAX_ITERATIONS;
        }
        
        double positions_per_sec = (elapsed > 0) ? estimated_positions / elapsed : 0;
        
        // Store stats
        MoveStats stats;
        stats.move = move_str;
        stats.san_move = san_move;
        stats.tree_size = estimated_positions;  // Approximate
        stats.time_seconds = elapsed;
        stats.iterations = estimated_positions;
        stats.player = player_name;
        move_history.push_back(stats);
        
        // Print move info
        cout << setw(4) << move_number << " | " 
             << setw(6) << player_name << " | "
             << setw(7) << move_str << " | "
             << setw(9) << estimated_positions << " | "
             << setw(8) << fixed << setprecision(3) << elapsed << " | "
             << setw(10) << fixed << setprecision(0) << positions_per_sec << endl;
        
        // Get FEN after this move (from the agent's updated state)
        const MCTS_state* updated_state = current_agent->get_current_state();
        const Chess_state* updated_chess_state = static_cast<const Chess_state*>(updated_state);
        string fen = updated_chess_state->get_board().getFen();
        cout << "     FEN: " << fen << endl;
        
        // Store this move for the next player
        last_move = chess_move;
        white_turn = !white_turn;
        move_number++;
        
        // Check if game ended after this move (before next iteration)
        const MCTS_state* post_move_state = current_agent->get_current_state();
        const Chess_state* post_move_chess_state = static_cast<const Chess_state*>(post_move_state);
        if (post_move_chess_state->is_terminal()) {
            auto [reason, result] = post_move_chess_state->get_board().isGameOver();
            cout << "\n=== Game Over ===" << endl;
            if (result == GameResult::DRAW) {
                cout << "Result: Draw" << endl;
            } else if (result == GameResult::WIN) {
                cout << "Result: " << (post_move_chess_state->get_board().sideToMove() == Color::WHITE ? "White" : "Black") << " wins!" << endl;
            } else {
                cout << "Result: " << (post_move_chess_state->get_board().sideToMove() == Color::WHITE ? "Black" : "White") << " wins!" << endl;
            }
            break;
        }
    }
    
    if (move_number > MAX_MOVES) {
        cout << "\n=== Game stopped: Maximum moves reached (safety limit) ===" << endl;
        cout << "This is unusual - most chess games end before 1000 moves." << endl;
    }
    
    // Print summary statistics
    cout << "\n=== Game Summary ===" << endl;
    cout << "Total moves: " << move_history.size() << endl;
    
    double total_time = 0;
    unsigned int total_positions = 0;
    for (const auto& stats : move_history) {
        total_time += stats.time_seconds;
        total_positions += stats.iterations;
    }
    
    cout << "Total time: " << fixed << setprecision(2) << total_time << " seconds" << endl;
    cout << "Total positions evaluated: " << total_positions << endl;
    cout << "Average positions/second: " << fixed << setprecision(0) << (total_positions / total_time) << endl;
    cout << "Average time per move: " << fixed << setprecision(3) << (total_time / move_history.size()) << " seconds" << endl;
    
    // Generate and output PGN
    cout << "\n=== PGN ===" << endl;
    cout << "[Event \"MCTS Self-Play\"]" << endl;
    cout << "[Site \"Computer\"]" << endl;
    
    // Get current date
    time_t now = time(0);
    tm* local_time = localtime(&now);
    char date_str[11];
    strftime(date_str, sizeof(date_str), "%Y.%m.%d", local_time);
    cout << "[Date \"" << date_str << "\"]" << endl;
    
    cout << "[Round \"1\"]" << endl;
    cout << "[White \"MCTS Agent\"]" << endl;
    cout << "[Black \"MCTS Agent\"]" << endl;
    
    // Determine result
    string result = "*";  // Unknown/ongoing
    if (move_number <= MAX_MOVES) {
        const MCTS_state* final_state = white_agent->get_current_state();
        const Chess_state* final_chess_state = static_cast<const Chess_state*>(final_state);
        if (final_chess_state->is_terminal()) {
            auto [reason, game_result] = final_chess_state->get_board().isGameOver();
            if (game_result == GameResult::DRAW) {
                result = "1/2-1/2";
            } else if (game_result == GameResult::WIN) {
                result = (final_chess_state->get_board().sideToMove() == Color::WHITE) ? "0-1" : "1-0";
            } else if (game_result == GameResult::LOSE) {
                result = (final_chess_state->get_board().sideToMove() == Color::WHITE) ? "1-0" : "0-1";
            }
        }
    }
    cout << "[Result \"" << result << "\"]" << endl;
    cout << "[FEN \"" << initial_fen << "\"]" << endl;
    cout << endl;
    
    // Output moves in PGN format
    // Reconstruct board and convert moves to SAN
    Board pgn_board(initial_fen);
    int move_count = 1;
    string pgn_moves;
    
    for (size_t i = 0; i < move_history.size(); i++) {
        const auto& stats = move_history[i];
        
        // Convert UCI move to Move object and then to SAN
        Movelist movelist;
        movegen::legalmoves(movelist, pgn_board);
        Move actual_move = Move::NO_MOVE;
        
        // Find the move by comparing UCI strings
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
            // Convert to SAN
            string san_move;
            try {
                san_move = chess::uci::moveToSan(pgn_board, actual_move);
            } catch (...) {
                san_move = stats.move;  // Fallback to UCI
            }
            
            if (stats.player == "White") {
                pgn_moves += to_string(move_count) + ". " + san_move + " ";
            } else {
                pgn_moves += san_move + " ";
                move_count++;
            }
            
            // Apply move to board
            pgn_board.makeMove(actual_move);
        } else {
            // Fallback: use stored move string
            if (stats.player == "White") {
                pgn_moves += to_string(move_count) + ". " + stats.move + " ";
            } else {
                pgn_moves += stats.move + " ";
                move_count++;
            }
        }
        
        // Line breaks every 10 moves (20 plies) for readability
        if ((i + 1) % 20 == 0) {
            pgn_moves += "\n";
        }
    }
    
    cout << pgn_moves << result << endl;
    
    // Cleanup
    delete white_agent;
    if (black_agent) delete black_agent;
    delete nn;
    
    return 0;
}

