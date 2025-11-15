#ifndef MCTS_CHESS_H
#define MCTS_CHESS_H

#include "../include/state.h"
#include "../include/chess.hpp"
#include <queue>

using namespace std;
using namespace chess;

/**
 * Chess_move - Wraps chess::Move for MCTS
 */
struct Chess_move : public MCTS_move {
    Move move;
    
    Chess_move(const Move& m) : move(m) {}
    Chess_move(const Chess_move& other) : move(other.move) {}
    
    bool operator==(const MCTS_move& other) const override {
        const Chess_move& o = static_cast<const Chess_move&>(other);
        return move == o.move;
    }
    
    string sprint() const override {
        // Use the same format as neural_network.cpp to ensure exact string matching
        // static_cast<string>(Square) produces "e2", "e4", etc.
        string result = static_cast<string>(move.from()) + static_cast<string>(move.to());
    
        if (move.typeOf() == Move::PROMOTION) {
            PieceType pt = move.promotionType();
            char promo = 'q';
            if (pt == PieceType::KNIGHT) promo = 'n';
            else if (pt == PieceType::BISHOP) promo = 'b';
            else if (pt == PieceType::ROOK) promo = 'r';
            result += promo;
        }
    
        return result;
    }
    
};

/**
 * Chess_state - Implements MCTS_state for chess using chess::Board
 */
class Chess_state : public MCTS_state {
private:
    Board board_;
    Move last_move_;           // Last move that led to this state
    Piece captured_piece_;     // Piece that was captured (if any)
    bool was_capture_;         // Whether last move was a capture
    
public:
    // Constructors
    Chess_state();
    Chess_state(const string& fen);
    Chess_state(const Chess_state& other);
    
    // MCTS_state interface implementation
    bool is_terminal() const override;
    MCTS_state* next_state(const MCTS_move* move) const override;
    queue<MCTS_move*>* actions_to_try() const override;
    double rollout() const override;
    bool player1_turn() const override;
    void print() const override;
    
    // Access to underlying board (useful for debugging)
    const Board& get_board() const { return board_; }
    Board& get_board() { return board_; }
    
    // Check if the move that led to this state was a capture
    bool was_capture() const;
    
    // Get the value of the captured piece (if any)
    // Returns: 9=queen, 5=rook, 3=bishop/knight, 1=pawn, 0=no capture
    // Note: This is only for debug output, not used in evaluation
    int captured_piece_value() const;
};

#endif // MCTS_CHESS_H

