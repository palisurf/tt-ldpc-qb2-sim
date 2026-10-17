# ==============================================================================
# Makefile for Tile Matrix 4-Codeword LDPC Engine (engines/tile_matrix_4cw)
# Architecture: Decoupled Access-Execute (DAE) with Native 32x32 Tensix Tiles
# Target: Tenstorrent QuietBox 2 (Blackhole / 440 Tensix Cores)
# ==============================================================================

TT_METAL_HOME ?= /home/ttuser/tt-metal

CXX          := g++
CXXFLAGS     := -std=c++20 -O3 -Wall -Wextra -pthread -fPIC
DEFINES      := -DARCH_BLACKHOLE -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DLDPC_USE_SFPU_LUT=1

INCLUDES     := -I. \
                -I$(TT_METAL_HOME) \
                -I$(TT_METAL_HOME)/build/include \
                -I$(TT_METAL_HOME)/build/include/tt-metalium \
                -I$(TT_METAL_HOME)/tt_metal/hw/inc

LIBDIRS      := -L$(TT_METAL_HOME)/build/lib -Wl,-rpath,$(TT_METAL_HOME)/build/lib
LIBS         := -ltt_metal -ltt_stl -lfmt -lspdlog -lpthread -ldl

TARGET       := ldpc_sim_tile_matrix
SRCS         := ldpc_sim_tile_matrix.cpp
OBJS         := $(SRCS:.cpp=.o)

TEST_LOCKSTEP := test_tensix_lockstep_debug
TEST_OBJS     := test_tensix_lockstep_debug.o

.PHONY: all clean run test_lockstep passthrough r_dataflow

all: $(TARGET)

passthrough: DEFINES += -DLDPC_TEST_PASSTHROUGH_MODE=1
passthrough: clean $(TARGET)
	@echo "[PASSTHROUGH BUILD] Compiled $(TARGET) with LDPC_TEST_PASSTHROUGH_MODE=1"

r_dataflow: DEFINES += -DLDPC_TEST_R_DATAFLOW_MODE=1
r_dataflow: clean $(TARGET)
	@echo "[R_DATAFLOW BUILD] Compiled $(TARGET) with LDPC_TEST_R_DATAFLOW_MODE=1"

amin_3p: DEFINES += -DLDPC_USE_3PIECE_AMIN_STAR=1
amin_3p: clean $(TARGET)
	@echo "[3-PIECE AMIN* BUILD] Compiled $(TARGET) with LDPC_USE_3PIECE_AMIN_STAR=1"

amin_lut: DEFINES += -DLDPC_USE_SFPU_LUT=1
amin_lut: clean $(TARGET)
	@echo "[SFPU LUT BUILD] Compiled $(TARGET) with LDPC_USE_SFPU_LUT=1"

$(TARGET): $(OBJS)
	@echo "[LINK] $@"
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LIBDIRS) $(LIBS)

$(TEST_LOCKSTEP): $(TEST_OBJS)
	@echo "[LINK] $@"
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) -o $@ $(LIBDIRS) $(LIBS)

test_lockstep: $(TEST_LOCKSTEP)

%.o: %.cpp
	@echo "[CXX] $<"
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

# Run 100 codewords (25 batches of 4) on single-core at 2.0 dB, 16 iterations
run: $(TARGET)
	@echo "[RUN] Executing $(TARGET) on QuietBox 2..."
	./$(TARGET) 2.0 25 16

clean:
	@echo "[CLEAN]"
	rm -f $(OBJS) $(TARGET) $(TEST_OBJS) $(TEST_LOCKSTEP)
