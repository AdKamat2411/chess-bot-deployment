/**
 * Lichess Bot Client
 * Connects to Lichess Bot API and plays games using MCTS engine
 * 
 * Usage: ./lichess_bot <api_token> [model_path]
 * 
 * API Token: Get from https://lichess.org/account/oauth/token
 *            Create a token with "bot:play" scope
 */

#include "Chess.h"
#include "../include/mcts.h"
#include "../include/neural_network.h"
#include "../include/chess.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <vector>
#include <curl/curl.h>
#include <regex>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace chess;

// Global state
atomic<bool> running(true);
mutex game_mutex;
map<string, Board> active_games;  // gameId -> board state
map<string, MCTS_agent*> game_agents;  // gameId -> MCTS agent
NeuralNetwork* global_nn = nullptr;  // Shared neural network across games

// libcurl write callback for streaming
struct WriteData {
    string data;
};

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    WriteData* wd = static_cast<WriteData*>(userp);
    wd->data.append(static_cast<char*>(contents), total_size);
    return total_size;
}

// libcurl write callback for streaming (line by line)
size_t StreamCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    string* buffer = static_cast<string*>(userp);
    buffer->append(static_cast<char*>(contents), total_size);
    
    // Process complete lines (ND-JSON format)
    size_t pos = 0;
    while ((pos = buffer->find('\n')) != string::npos) {
        string line = buffer->substr(0, pos);
        buffer->erase(0, pos + 1);
        
        if (!line.empty()) {
            // Process JSON line here
            // This will be handled in the streaming thread
        }
    }
    
    return total_size;
}

// Make HTTP GET request
string http_get(const string& url, const string& token) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        cerr << "ERROR: Failed to initialize curl" << endl;
        return "";
    }
    
    WriteData wd;
    struct curl_slist* headers = nullptr;
    
    string auth_header = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &wd);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        cerr << "ERROR: curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
        return "";
    }
    
    return wd.data;
}

// Make HTTP POST request
string http_post(const string& url, const string& token, const string& data = "", const string& content_type = "application/json") {
    CURL* curl = curl_easy_init();
    if (!curl) {
        cerr << "ERROR: Failed to initialize curl" << endl;
        return "";
    }
    
    WriteData wd;
    struct curl_slist* headers = nullptr;
    
    string auth_header = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, ("Content-Type: " + content_type).c_str());
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &wd);
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        cerr << "ERROR: curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
        return "";
    }
    
    return wd.data;
}

// Simple JSON parsing helpers (basic, for simple cases)
string extract_json_string(const string& json, const string& key) {
    regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    smatch match;
    if (regex_search(json, match, pattern)) {
        return match[1].str();
    }
    return "";
}

// Forward declarations
void stream_game(const string& game_id, const string& token);
void make_move(const string& game_id, const string& token, MCTS_agent* agent, Board& board);
vector<string> get_online_bots(const string& token);
bool challenge_bot(const string& token, const string& bot_username, bool rated = false, int time_limit = 300, int increment = 0);

// Stream events from Lichess (challenges, game starts, etc.)
void stream_events(const string& token) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        cerr << "ERROR: Failed to initialize curl for event stream" << endl;
        return;
    }
    
    string buffer;
    struct curl_slist* headers = nullptr;
    string auth_header = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    
    curl_easy_setopt(curl, CURLOPT_URL, "https://lichess.org/api/bot/stream/event");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);  // No timeout for streaming
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    
    cout << "Listening for events..." << endl;
    
    // For streaming, we need to process the buffer continuously
    while (running) {
        // Process buffer for complete JSON lines
        size_t pos = 0;
        while ((pos = buffer.find('\n')) != string::npos) {
            string line = buffer.substr(0, pos);
            buffer.erase(0, pos + 1);
            
            if (line.empty()) continue;
            
            // Debug: print all events to see what we're receiving
            cout << "[Event] Received: " << line.substr(0, min(100, (int)line.length())) << "..." << endl;
            
            // Parse JSON event (simple string matching)
            string type = extract_json_string(line, "type");
            
            if (type == "challenge") {
                // Extract challenge ID (simplified - look for "id" in challenge object)
                regex id_pattern("\"challenge\"\\s*:\\s*\\{[^}]*\"id\"\\s*:\\s*\"([^\"]+)\"");
                smatch match;
                if (regex_search(line, match, id_pattern)) {
                    string challenge_id = match[1].str();
                    cout << "[Event] Received challenge: " << challenge_id << endl;
                    
                    // Auto-accept challenges
                    string accept_url = "https://lichess.org/api/challenge/" + challenge_id + "/accept";
                    string response = http_post(accept_url, token);
                    cout << "[Event] Challenge accepted: " << response << endl;
                }
            } else if (type == "gameStart") {
                // Extract game ID
                regex id_pattern("\"game\"\\s*:\\s*\\{[^}]*\"id\"\\s*:\\s*\"([^\"]+)\"");
                smatch match;
                if (regex_search(line, match, id_pattern)) {
                    string game_id = match[1].str();
                    cout << "[Event] *** GAME STARTED: " << game_id << " ***" << endl;
                    
                    // Start game stream in separate thread
                    thread game_thread(stream_game, game_id, token);
                    game_thread.detach();
                }
            } else if (!type.empty()) {
                cout << "[Event] Unknown event type: " << type << endl;
            }
        }
        
        // Perform the curl operation (this will block and stream data)
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK && res != CURLE_OPERATION_TIMEDOUT) {
            cerr << "[Event] ERROR: Event stream failed: " << curl_easy_strerror(res) << endl;
            this_thread::sleep_for(chrono::seconds(5));
            // Reconnect
            buffer.clear();
            continue;
        }
        
        // Small sleep to prevent CPU spinning
        this_thread::sleep_for(chrono::milliseconds(50));
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}

string extract_json_value(const string& json, const string& key) {
    // Try string first
    string str_val = extract_json_string(json, key);
    if (!str_val.empty()) return str_val;
    
    // Try number or boolean
    regex pattern("\"" + key + "\"\\s*:\\s*([^,\\}]+)");
    smatch match;
    if (regex_search(json, match, pattern)) {
        string val = match[1].str();
        // Remove quotes and whitespace
        val.erase(remove(val.begin(), val.end(), '"'), val.end());
        val.erase(remove(val.begin(), val.end(), ' '), val.end());
        return val;
    }
    return "";
}

// Stream game state and make moves
void stream_game(const string& game_id, const string& token) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        cerr << "ERROR: Failed to initialize curl for game stream" << endl;
        return;
    }
    
    string buffer;
    struct curl_slist* headers = nullptr;
    string auth_header = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    
    string url = "https://lichess.org/api/bot/game/stream/" + game_id;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);  // No timeout for streaming
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    
    Board game_board;
    MCTS_agent* agent = nullptr;
    bool is_white = false;
    bool game_initialized = false;
    static map<string, int> last_move_count;  // Track move count per game
    
    cout << "[Game " << game_id << "] Streaming game..." << endl;
    
    while (running) {
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            cerr << "[Game " << game_id << "] ERROR: Game stream failed: " << curl_easy_strerror(res) << endl;
            break;
        }
        
        // Process buffer for complete JSON lines
        size_t pos = 0;
        while ((pos = buffer.find('\n')) != string::npos) {
            string line = buffer.substr(0, pos);
            buffer.erase(0, pos + 1);
            
            if (line.empty()) continue;
            
            // Debug: print game state updates
            cout << "[Game " << game_id << "] Update: " << line.substr(0, min(80, (int)line.length())) << "..." << endl;
            
            // Parse JSON game state (simplified parsing)
            string type = extract_json_string(line, "type");
            
            if (type == "gameFull") {
                // Extract FEN from state object
                regex fen_pattern("\"state\"\\s*:\\s*\\{[^}]*\"fen\"\\s*:\\s*\"([^\"]+)\"");
                smatch match;
                string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
                if (regex_search(line, match, fen_pattern)) {
                    fen = match[1].str();
                }
                
                game_board = Board(fen);
                
                // Determine if we're white (simplified - check if white to move at start)
                // In a real implementation, you'd get your user ID from the API
                is_white = (game_board.sideToMove() == Color::WHITE);
                
                // Use global neural network (shared across games)
                // Initialize MCTS agent
                Chess_state* initial_state = new Chess_state(fen);
                agent = new MCTS_agent(initial_state, 5000, 2, global_nn, 1.0);
                game_initialized = true;
                
                lock_guard<mutex> lock(game_mutex);
                active_games[game_id] = game_board;
                game_agents[game_id] = agent;
                
                cout << "[Game " << game_id << "] Game initialized. FEN: " << fen << endl;
                cout << "[Game " << game_id << "] We are: " << (is_white ? "White" : "Black") << endl;
                
                // If it's our turn, make a move
                if (game_board.sideToMove() == (is_white ? Color::WHITE : Color::BLACK)) {
                    cout << "[Game " << game_id << "] It's our turn! Making move..." << endl;
                    make_move(game_id, token, agent, game_board);
                } else {
                    cout << "[Game " << game_id << "] Waiting for opponent's move..." << endl;
                }
            } else if (type == "gameState" && game_initialized) {
                // Extract moves string
                regex moves_pattern("\"moves\"\\s*:\\s*\"([^\"]+)\"");
                smatch match;
                if (regex_search(line, match, moves_pattern)) {
                    string moves_str = match[1].str();
                    
                    if (!moves_str.empty()) {
                        // Parse moves (space-separated UCI)
                        istringstream move_stream(moves_str);
                        string move_uci;
                        vector<string> all_moves;
                        while (move_stream >> move_uci) {
                            all_moves.push_back(move_uci);
                        }
                        
                        // Get current move count to only apply new moves
                        int current_move_count = all_moves.size();
                        int start_idx = last_move_count[game_id];
                        last_move_count[game_id] = current_move_count;
                        
                        // Apply only new moves
                        for (int i = start_idx; i < current_move_count; i++) {
                            const string& move_uci = all_moves[i];
                            if (move_uci.length() >= 4) {
                                try {
                                    Square from = Square(move_uci.substr(0, 2));
                                    Square to = Square(move_uci.substr(2, 2));
                                    
                                    Move move;
                                    if (move_uci.length() == 5) {
                                        // Promotion
                                        char promo = move_uci[4];
                                        PieceType pt = PieceType::QUEEN;
                                        if (promo == 'n') pt = PieceType::KNIGHT;
                                        else if (promo == 'b') pt = PieceType::BISHOP;
                                        else if (promo == 'r') pt = PieceType::ROOK;
                                        move = Move::make<Move::PROMOTION>(from, to, pt);
                                    } else {
                                        move = Move::make<Move::NORMAL>(from, to);
                                    }
                                    
                                    // Check if move is legal by generating legal moves
                                    Movelist legal_moves;
                                    movegen::legalmoves(legal_moves, game_board);
                                    bool is_legal = false;
                                    for (const Move& m : legal_moves) {
                                        if (m == move) {
                                            is_legal = true;
                                            break;
                                        }
                                    }
                                    
                                    if (is_legal) {
                                        game_board.makeMove(move);
                                        cout << "[Game " << game_id << "] Opponent played: " << move_uci << endl;
                                        
                                        // Update agent
                                        lock_guard<mutex> lock(game_mutex);
                                        if (game_agents.find(game_id) != game_agents.end()) {
                                            Chess_move* chess_move = new Chess_move(move);
                                            game_agents[game_id]->genmove(chess_move);
                                            delete chess_move;
                                        }
                                    }
                                } catch (...) {
                                    // Skip invalid moves
                                    continue;
                                }
                            }
                        }
                        
                        // Check if it's our turn
                        if (game_board.sideToMove() == (is_white ? Color::WHITE : Color::BLACK)) {
                            cout << "[Game " << game_id << "] It's our turn! Making move..." << endl;
                            lock_guard<mutex> lock(game_mutex);
                            if (game_agents.find(game_id) != game_agents.end()) {
                                make_move(game_id, token, game_agents[game_id], game_board);
                            }
                        } else {
                            cout << "[Game " << game_id << "] Waiting for opponent's move..." << endl;
                        }
                    }
                }
            }
        }
        
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    // Cleanup
    lock_guard<mutex> lock(game_mutex);
    if (game_agents.find(game_id) != game_agents.end()) {
        delete game_agents[game_id];
        game_agents.erase(game_id);
    }
    active_games.erase(game_id);
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}

// Make a move using MCTS and send it to Lichess
void make_move(const string& game_id, const string& token, MCTS_agent* agent, Board& board) {
    cout << "[Game " << game_id << "] Thinking..." << endl;
    
    // Get move from MCTS
    const MCTS_move* move = agent->genmove(nullptr);
    if (!move) {
        cerr << "[Game " << game_id << "] ERROR: No move returned from MCTS" << endl;
        return;
    }
    
    const Chess_move* chess_move = static_cast<const Chess_move*>(move);
    string move_uci = chess_move->sprint();
    
    cout << "[Game " << game_id << "] Playing move: " << move_uci << endl;
    
    // Send move to Lichess
    string url = "https://lichess.org/api/bot/game/" + game_id + "/move/" + move_uci;
    string response = http_post(url, token);
    
    if (response.find("ok") != string::npos) {
        cout << "[Game " << game_id << "] ✓ Move sent successfully: " << move_uci << endl;
    } else {
        cerr << "[Game " << game_id << "] ERROR: Failed to send move. Response: " << response << endl;
    }
}

// Get list of online bots
vector<string> get_online_bots(const string& token) {
    string url = "https://lichess.org/api/bot/online";
    string response = http_get(url, token);
    
    vector<string> bots;
    
    if (response.empty()) {
        cerr << "ERROR: Empty response from API" << endl;
        return bots;
    }
    
    // The API returns an array of user objects, each with an "id" or "username" field
    // Format: [{"id":"bot1",...}, {"id":"bot2",...}]
    // We need to extract the "id" or "username" field from each object
    
    // First, try to find if it's a simple array of strings: ["bot1", "bot2"]
    if (response.find('[') == 0 && response.find('{') == string::npos) {
        // Simple string array format
        regex bot_pattern("\"([^\"]+)\"");
        smatch match;
        string::const_iterator search_start(response.cbegin());
        
        while (regex_search(search_start, response.cend(), match, bot_pattern)) {
            bots.push_back(match[1].str());
            search_start = match.suffix().first;
        }
    } else {
        // Complex object array format - extract "id" or "username" fields
        // Look for patterns like "id":"username" or "username":"username"
        regex id_pattern("\"(?:id|username)\"\\s*:\\s*\"([^\"]+)\"");
        smatch match;
        string::const_iterator search_start(response.cbegin());
        
        while (regex_search(search_start, response.cend(), match, id_pattern)) {
            string username = match[1].str();
            // Avoid duplicates
            if (find(bots.begin(), bots.end(), username) == bots.end()) {
                bots.push_back(username);
            }
            search_start = match.suffix().first;
        }
    }
    
    return bots;
}

// Challenge a bot
bool challenge_bot(const string& token, const string& bot_username, 
                   bool rated, int time_limit, int increment) {
    string url = "https://lichess.org/api/challenge/" + bot_username;
    
    // Build form data
    ostringstream form_data;
    form_data << "rated=" << (rated ? "true" : "false");
    form_data << "&clock.limit=" << time_limit;
    form_data << "&clock.increment=" << increment;
    form_data << "&variant=standard";
    
    string response = http_post(url, token, form_data.str(), "application/x-www-form-urlencoded");
    
    if (response.find("ok") != string::npos || response.find("challenge") != string::npos) {
        cout << "Challenge sent to " << bot_username << " successfully!" << endl;
        return true;
    } else {
        cerr << "ERROR: Failed to challenge " << bot_username << ". Response: " << response << endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <api_token> [model_path] [command]" << endl;
        cerr << endl;
        cerr << "Commands:" << endl;
        cerr << "  (no command)        - Listen for challenges and play games" << endl;
        cerr << "  list                - List online bots" << endl;
        cerr << "  challenge <user>     - Challenge a specific bot" << endl;
        cerr << "  auto                - Auto-challenge random bots and play games" << endl;
        cerr << endl;
        cerr << "Get your API token from: https://lichess.org/account/oauth/token" << endl;
        cerr << "Create a token with 'bot:play' scope" << endl;
        return 1;
    }
    
    string token = argv[1];
    string model_path = (argc >= 3) ? argv[2] : "";
    string command = (argc >= 4) ? argv[3] : "";
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Handle commands
    if (command == "list") {
        cout << "=== Online Bots ===" << endl;
        vector<string> bots = get_online_bots(token);
        if (bots.empty()) {
            cout << "No bots online or failed to fetch list." << endl;
            cout << "Note: Make sure your API token has 'bot:play' scope" << endl;
        } else {
            cout << "Found " << bots.size() << " online bot(s):" << endl;
            for (const string& bot : bots) {
                cout << "  - " << bot << endl;
            }
        }
        curl_global_cleanup();
        return 0;
    } else if (command == "challenge") {
        if (argc < 5) {
            cerr << "Usage: " << argv[0] << " <api_token> [model_path] challenge <bot_username>" << endl;
            curl_global_cleanup();
            return 1;
        }
        string bot_username = argv[4];
        cout << "Challenging " << bot_username << "..." << endl;
        bool success = challenge_bot(token, bot_username, false, 300, 0);
        curl_global_cleanup();
        return success ? 0 : 1;
    } else if (command == "auto") {
        // Auto-challenge mode: periodically challenge random bots
        cout << "=== Auto-Challenge Mode ===" << endl;
        cout << "Will automatically challenge bots and play games" << endl;
        cout << "Press Ctrl+C to stop" << endl;
        cout << "===========================" << endl << endl;
        
        // Load neural network if provided
        if (!model_path.empty()) {
            global_nn = new NeuralNetwork();
            if (!global_nn->load_model(model_path)) {
                cerr << "WARNING: Failed to load neural network. Using heuristic rollouts." << endl;
                delete global_nn;
                global_nn = nullptr;
            } else {
                cout << "Neural network loaded: " << model_path << endl;
            }
        }
        
        // Start event streaming in background thread
        thread event_thread(stream_events, token);
        event_thread.detach();
        
        // Periodically challenge bots
        while (running) {
            this_thread::sleep_for(chrono::seconds(10));  // Wait 10 seconds between challenges
            
            if (!running) break;
            
            cout << "\n[Auto] Checking for online bots..." << endl;
            vector<string> bots = get_online_bots(token);
            
            if (bots.empty()) {
                cout << "[Auto] No bots online. Waiting..." << endl;
                continue;
            }
            
            // Pick a random bot to challenge
            srand(time(nullptr));
            int random_idx = rand() % bots.size();
            string target_bot = bots[random_idx];
            
            cout << "[Auto] Challenging " << target_bot << "..." << endl;
            challenge_bot(token, target_bot, false, 300, 0);
            
            // Wait a bit before next challenge
            this_thread::sleep_for(chrono::seconds(5));
        }
        
        // Cleanup
        running = false;
        this_thread::sleep_for(chrono::seconds(2));
        
        lock_guard<mutex> lock(game_mutex);
        for (auto& [id, agent] : game_agents) {
            delete agent;
        }
        game_agents.clear();
        active_games.clear();
        
        curl_global_cleanup();
        if (global_nn) delete global_nn;
        return 0;
    }
    
    // Load neural network if provided (shared across games)
    if (!model_path.empty()) {
        global_nn = new NeuralNetwork();
        if (!global_nn->load_model(model_path)) {
            cerr << "WARNING: Failed to load neural network. Using heuristic rollouts." << endl;
            delete global_nn;
            global_nn = nullptr;
        } else {
            cout << "Neural network loaded: " << model_path << endl;
        }
    }
    
    cout << "=== Lichess Bot Client ===" << endl;
    cout << "Connecting to Lichess..." << endl;
    cout << "Press Ctrl+C to stop" << endl;
    cout << "=========================" << endl << endl;
    
    // Start event streaming
    stream_events(token);
    
    // Cleanup
    running = false;
    this_thread::sleep_for(chrono::seconds(1));  // Give threads time to finish
    
    // Cleanup agents
    lock_guard<mutex> lock(game_mutex);
    for (auto& [id, agent] : game_agents) {
        delete agent;
    }
    game_agents.clear();
    active_games.clear();
    
    curl_global_cleanup();
    if (global_nn) delete global_nn;
    
    return 0;
}

