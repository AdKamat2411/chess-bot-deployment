#include "Chess.h"
#include <iostream>
using namespace std;
using namespace chess;

int main() {
    string fen = "rnb1kbnr/pp1p1ppp/8/2p1p2Q/4P3/3B4/PPP1NPPP/RNB1K2R b KQkq - 0 6";
    Chess_state state(fen);
    auto [reason, result] = state.get_board().isGameOver();
    cout << "Is terminal: " << state.is_terminal() << endl;
    cout << "Result: " << (int)result << " (0=NONE, 1=WIN, 2=LOSE, 3=DRAW)" << endl;
    cout << "Reason: " << (int)reason << endl;
    cout << "FEN: " << state.get_board().getFen() << endl;
    return 0;
}
