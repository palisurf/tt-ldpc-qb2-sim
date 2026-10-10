# ==============================================================================
# Makefile for 5-Worker Scalar LDPC Engine (engines/scalar_5w)
# Architecture: 5 RV32 Workers (BRISC, NCRISC, TRISC0..2) with 2-CW Packed Interleaving
# Target: Tenstorrent QuietBox 2 (Blackhole / 440 Tensix Cores)
# ==============================================================================

TT_METAL_HOME ?= /home/ttuser/tt-metal

CXX          := g++
CXXFLAGS     := -std=c++20 -O3 -Wall -Wextra -pthread -fPIC
DEFINES      := -DARCH_BLACKHOLE -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL

INCLUDES     := -I. \
                -I$(TT_METAL_HOME) \
                -I$(TT_METAL_HOME)/build/include \
                -I$(TT_METAL_HOME)/build/include/tt-metalium \
                -I$(TT_METAL_HOME)/tt_metal/hw/inc

LIBDIRS      := -L$(TT_METAL_HOME)/build/lib -Wl,-rpath,$(TT_METAL_HOME)/build/lib
LIBS         := -ltt_metal -ltt_stl -lfmt -lspdlog -lpthread -ldl

TARGET       := ldpc_sim_scalar
SRCS         := ldpc_sim_scalar.cpp
OBJS         := $(SRCS:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "[LINK] $@"
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LIBDIRS) $(LIBS)

%.o: %.cpp
	@echo "[CXX] $<"
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

# Run 10 batches of 2 codewords on single-core at 2.0 dB, 16 iterations
run: $(TARGET)
	@echo "[RUN] Executing $(TARGET) on QuietBox 2..."
	./$(TARGET) -c ../../matrices/AR4JA_r45_4c_128c_r12.chinn.out -e 2.0 -p 512 -t 50 -k 10 -i 16 -b 10

clean:
	@echo "[CLEAN]"
	rm -f $(OBJS) $(TARGET)
