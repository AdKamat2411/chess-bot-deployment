/**
 * Example usage of Chess_state with MCTS
 * 
 * This demonstrates how to use the Chess_state implementation
 * with the MCTS algorithm.
 */

#include "Chess.h"
#include "../include/mcts.h"
#include <iostream>

using namespace std;

int main() {
    // Create a chess state (starts from initial position)
    Chess_state* initial_state = new Chess_state();
    
    cout << "Initial position:" << endl;
    initial_state->print();
    cout << endl;
    
    // Create MCTS agent
    MCTS_agent* agent = new MCTS_agent(initial_state, 10000, 3);  // 10k iterations or 5 seconds
    
    cout << "Running MCTS search..." << endl;
    
    // Get a move (no enemy move for first move)
    const MCTS_move* move = agent->genmove(NULL);
    
    if (move) {
        const Chess_move* chess_move = static_cast<const Chess_move*>(move);
        cout << "Best move: " << chess_move->sprint() << endl;
    } else {
        cout << "No move available (game over?)" << endl;
    }
    
    // Print current state
    const MCTS_state* current = agent->get_current_state();
    const Chess_state* chess_state = static_cast<const Chess_state*>(current);
    cout << "\nAfter move:" << endl;
    chess_state->print();
    
    // Cleanup
    delete agent;
    
    return 0;
}

