FLAGS = -O2 -g3 -pedantic -std=c++17 -pthread # -Wall -Wextra

CHESS_EXE = chess_mcts

COMMON_OBJ = JobScheduler.o mcts.o neural_network.o

# LibTorch paths (set these after downloading libtorch)
# Check common locations: Downloads, home directory, or custom path
ifdef LIBTORCH_PATH
  # User specified path
else ifneq ($(wildcard $(HOME)/Downloads/libtorch),)
  LIBTORCH_PATH = $(HOME)/Downloads/libtorch
else ifneq ($(wildcard $(HOME)/libtorch),)
  LIBTORCH_PATH = $(HOME)/libtorch
else
  LIBTORCH_PATH = $(HOME)/libtorch  # Default fallback
endif

TORCH_INCLUDE = $(LIBTORCH_PATH)/include
TORCH_LIB = $(LIBTORCH_PATH)/lib

# Check if libtorch is available
LIBTORCH_AVAILABLE = $(shell test -d $(LIBTORCH_PATH) && echo "yes" || echo "no")

all: Chess

mcts.o: src/mcts.cpp include/mcts.h include/state.h
ifeq ($(LIBTORCH_AVAILABLE),yes)
	g++ -c $(FLAGS) -I. -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include src/mcts.cpp -o mcts.o
else
	g++ -c $(FLAGS) -I. src/mcts.cpp -o mcts.o
endif

JobScheduler.o: src/JobScheduler.cpp include/JobScheduler.h
	g++ -c $(FLAGS) -I. src/JobScheduler.cpp -o JobScheduler.o

neural_network.o: src/neural_network.cpp include/neural_network.h include/chess.hpp
ifeq ($(LIBTORCH_AVAILABLE),yes)
	g++ -c $(FLAGS) -I. -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include src/neural_network.cpp -o neural_network.o
else
	@echo "Warning: libtorch not found at $(LIBTORCH_PATH)"
	@echo "Please download libtorch and set LIBTORCH_PATH"
	@exit 1
endif

Chess.o: Chess/Chess.cpp Chess/Chess.h include/state.h include/chess.hpp
	g++ -c $(FLAGS) -I. Chess/Chess.cpp -o Chess.o

Chess: mcts.o JobScheduler.o Chess.o neural_network.o Chess/main.cpp
ifeq ($(LIBTORCH_AVAILABLE),yes)
	g++ -o $(CHESS_EXE) $(FLAGS) -I. -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include \
		Chess/main.cpp Chess.o $(COMMON_OBJ) \
		-L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 \
		-Wl,-rpath,$(TORCH_LIB)
else
	@echo "Error: Cannot build without libtorch"
	@exit 1
endif

Example: mcts.o JobScheduler.o Chess.o Chess/example.cpp
	g++ -o chess_example $(FLAGS) -I. Chess/example.cpp Chess.o $(COMMON_OBJ)

Play: mcts.o JobScheduler.o Chess.o neural_network.o Chess/play.cpp
ifeq ($(LIBTORCH_AVAILABLE),yes)
	g++ -o chess_play $(FLAGS) -I. -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include \
		Chess/play.cpp Chess.o $(COMMON_OBJ) \
		-L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 \
		-Wl,-rpath,$(TORCH_LIB)
else
	@echo "Error: Cannot build without libtorch"
	@exit 1
endif

NNTest: Chess.o neural_network.o Chess/nn_test.cpp
ifeq ($(LIBTORCH_AVAILABLE),yes)
	g++ -o nn_test $(FLAGS) -I. -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include \
		Chess/nn_test.cpp Chess.o neural_network.o \
		-L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 \
		-Wl,-rpath,$(TORCH_LIB)
else
	@echo "Error: Cannot build without libtorch"
	@exit 1
endif

NNValue: Chess.o neural_network.o Chess/nn_value.cpp
ifeq ($(LIBTORCH_AVAILABLE),yes)
	g++ -o nn_value $(FLAGS) -I. -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include \
		Chess/nn_value.cpp Chess.o neural_network.o \
		-L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 \
		-Wl,-rpath,$(TORCH_LIB)
else
	@echo "Error: Cannot build without libtorch"
	@exit 1
endif

LichessBot: mcts.o JobScheduler.o Chess.o neural_network.o Chess/lichess_bot.cpp
ifeq ($(LIBTORCH_AVAILABLE),yes)
	g++ -o lichess_bot $(FLAGS) -I. -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include \
		Chess/lichess_bot.cpp Chess.o $(COMMON_OBJ) \
		-L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 \
		-lcurl \
		-Wl,-rpath,$(TORCH_LIB)
else
	@echo "Error: Cannot build without libtorch"
	@exit 1
endif

Bridge: mcts.o JobScheduler.o Chess.o neural_network.o Chess/mcts_bridge.cpp
ifeq ($(LIBTORCH_AVAILABLE),yes)
	g++ -o mcts_bridge $(FLAGS) -I. -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include \
		Chess/mcts_bridge.cpp Chess.o $(COMMON_OBJ) \
		-L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 \
		-Wl,-rpath,$(TORCH_LIB)
else
	@echo "Error: Cannot build without libtorch"
	@exit 1
endif

clean:
	rm -f *.o $(CHESS_EXE) chess_play chess_example nn_test nn_value lichess_bot mcts_bridge

.PHONY: all clean Chess

