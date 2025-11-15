#include "Chess.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;
using namespace chess;

// Constructor: Start from initial position
Chess_state::Chess_state() : MCTS_state(), board_(), last_move_(Move::NO_MOVE), 
    captured_piece_(Piece::NONE), was_capture_(false) {
    // Board is initialized to starting position by default
}

// Constructor: Start from FEN string
Chess_state::Chess_state(const string& fen) : MCTS_state(), board_(fen), 
    last_move_(Move::NO_MOVE), captured_piece_(Piece::NONE), was_capture_(false) {
}

// Copy constructor
Chess_state::Chess_state(const Chess_state& other) 
    : MCTS_state(other), board_(other.board_), last_move_(other.last_move_),
      captured_piece_(other.captured_piece_), was_capture_(other.was_capture_) {
    // chess::Board copy constructor handles everything
}

// Check if game is over
bool Chess_state::is_terminal() const {
    auto [reason, result] = board_.isGameOver();
    return result != GameResult::NONE;
}

// Create new state after applying a move
MCTS_state* Chess_state::next_state(const MCTS_move* move) const {
    const Chess_move* m = static_cast<const Chess_move*>(move);
    
    // Create new state by copying current board
    Chess_state* new_state = new Chess_state(*this);
    
    // Check if this is a capture before applying the move
    bool is_capture = board_.isCapture(m->move);
    Piece captured_piece = Piece::NONE;
    if (is_capture) {
        captured_piece = board_.at(m->move.to());
    }
    
    // Apply the move
    new_state->board_.makeMove(m->move);
    
    // Store move and capture information
    new_state->last_move_ = m->move;
    new_state->was_capture_ = is_capture;
    new_state->captured_piece_ = captured_piece;
    
    return new_state;
}

// Check if the move that led to this state was a capture
bool Chess_state::was_capture() const {
    return was_capture_;
}

int Chess_state::captured_piece_value() const {
    if (!was_capture_ || captured_piece_ == Piece::NONE) {
        return 0;
    }
    
    PieceType pt = captured_piece_.type();
    if (pt == PieceType::QUEEN) return 9;
    if (pt == PieceType::ROOK) return 5;
    if (pt == PieceType::BISHOP || pt == PieceType::KNIGHT) return 3;
    if (pt == PieceType::PAWN) return 1;
    return 0;
}

// Get all legal moves
queue<MCTS_move*>* Chess_state::actions_to_try() const {
    queue<MCTS_move*>* Q = new queue<MCTS_move*>();
    
    // Generate all legal moves
    Movelist movelist;
    movegen::legalmoves(movelist, board_);
    
    // Add each move to the queue
    for (const Move& move : movelist) {
        Q->push(new Chess_move(move));
    }
    
    return Q;
}

// Check if it's player 1's turn (white)
bool Chess_state::player1_turn() const {
    return board_.sideToMove() == Color::WHITE;
}

// Random rollout simulation
double Chess_state::rollout() const {
    // If already terminal, return result from white's perspective
    if (is_terminal()) {
        auto [reason, result] = board_.isGameOver();
        if (result == GameResult::DRAW) {
            return 0.5;
        }
        // WIN/LOSE are from side-to-move's perspective
        // We need white's perspective (player1)
        Color white = Color::WHITE;
        Color side_to_move = board_.sideToMove();
        
        if (result == GameResult::WIN) {
            // Side to move won
            return (side_to_move == white) ? 1.0 : 0.0;
        } else if (result == GameResult::LOSE) {
            // Side to move lost
            return (side_to_move == white) ? 0.0 : 1.0;
        }
        return 0.5;
    }
    
    // Create a copy for simulation (we'll modify it)
    Board sim_board = board_;
    
    // Random number generator (seed with current time)
    static mt19937 rng(static_cast<unsigned>(
        chrono::steady_clock::now().time_since_epoch().count()));
    
    Color white = Color::WHITE;
    
    // Simulate random game until terminal
    int max_moves = 500;  // Safety limit to prevent infinite loops
    int move_count = 0;
    
    while (move_count < max_moves) {
        // Check if game is over
        auto [reason, result] = sim_board.isGameOver();
        if (result != GameResult::NONE) {
            // Game ended - convert to white's perspective
            if (result == GameResult::DRAW) {
                return 0.5;
            }
            
            // WIN/LOSE are from current side-to-move's perspective
            // We need to convert to white's perspective
            Color current_side = sim_board.sideToMove();
            
            if (result == GameResult::WIN) {
                // Current side won
                return (current_side == white) ? 1.0 : 0.0;
            } else if (result == GameResult::LOSE) {
                // Current side lost (opponent won)
                return (current_side == white) ? 0.0 : 1.0;
            }
            return 0.5;
        }
        
        // Get legal moves
        Movelist movelist;
        movegen::legalmoves(movelist, sim_board);
        
        if (movelist.empty()) {
            // No legal moves - should be terminal, but handle gracefully
            return 0.5;
        }
        
        // Pick random move
        uniform_int_distribution<size_t> dist(0, movelist.size() - 1);
        size_t random_idx = dist(rng);
        Move random_move = movelist[random_idx];
        
        // Apply move
        sim_board.makeMove(random_move);
        move_count++;
    }
    
    // Reached max moves - return neutral value (draw)
    // We don't use material-based heuristics - rely on NN evaluation instead
    return 0.5;  // Draw/neutral position
}

// Print the board
void Chess_state::print() const {
    cout << board_ << endl;
    cout << "FEN: " << board_.getFen() << endl;
    cout << "Side to move: " << (board_.sideToMove() == Color::WHITE ? "White" : "Black") << endl;
    
    auto [reason, result] = board_.isGameOver();
    if (result != GameResult::NONE) {
        cout << "Game over: ";
        if (result == GameResult::WIN) cout << "Win";
        else if (result == GameResult::LOSE) cout << "Loss";
        else if (result == GameResult::DRAW) cout << "Draw";
        cout << " (" << static_cast<int>(reason) << ")" << endl;
    }
}

