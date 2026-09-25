# ==============================================================================
# Makefile for TT-Metalium LDPC Monte Carlo Simulator (QB2 / Blackhole)
# ==============================================================================

TT_METAL_HOME ?= /home/ttuser/tt-metal

CXX          := g++
CXXFLAGS     := -std=c++20 -O3 -Wall -Wextra -pthread -fPIC
DEFINES      := -DARCH_BLACKHOLE -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL

INCLUDES     := -I$(TT_METAL_HOME) \
                -I$(TT_METAL_HOME)/build/include \
                -I$(TT_METAL_HOME)/build/include/tt-metalium \
                -I$(TT_METAL_HOME)/tt_metal/hw/inc

LIBDIRS      := -L$(TT_METAL_HOME)/build/lib -Wl,-rpath,$(TT_METAL_HOME)/build/lib
LIBS         := -ltt_metal -ltt_stl -lfmt -lspdlog -lpthread -ldl

TARGET       := ldpc_sim
SRCS         := ldpc_sim.cpp
OBJS         := $(SRCS:.cpp=.o)

HOST_TEST_TARGET := test_kernel_host
HOST_TEST_SRCS   := tests/test_kernel_host.cpp

TTSIM_TEST_TARGET := test_kernel_ttsim
TTSIM_TEST_SRCS   := tests/test_kernel_ttsim.cpp

.PHONY: all clean test_host test_ttsim sim

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "[LINK] $@"
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LIBDIRS) $(LIBS)

# Standalone Host Unit Test (Pure C++, zero TT-Metal SDK dependency)
test_host: $(HOST_TEST_SRCS)
	@echo "[CXX] $< -> $(HOST_TEST_TARGET)"
	$(CXX) -std=c++17 -O3 -Wall -Wextra $(HOST_TEST_SRCS) -o $(HOST_TEST_TARGET)
	@echo "[RUN] Executing host unit tests..."
	./$(HOST_TEST_TARGET)

# TT-Metalium Single-Core Unit Test Harness (Compiles with TT-Metalium)
test_ttsim: $(TTSIM_TEST_SRCS)
	@echo "[CXX] $< -> $(TTSIM_TEST_TARGET)"
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) $< -o $(TTSIM_TEST_TARGET) $(LIBDIRS) $(LIBS)

test_unit_single_core: tests/test_unit_single_core.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) $< -o $@ $(LIBDIRS) $(LIBS)

# Standalone High-Speed Uncoded BPSK AWGN Simulator
sim_uncoded_bpsk: sim_uncoded_bpsk.cpp
	$(CXX) -std=c++20 -O3 -Wall -Wextra -pthread $< -o $@

# Standalone PRNG Jump Separation Test
test_xoshiro_jump: tests/test_xoshiro_jump.cpp
	$(CXX) -std=c++20 -O3 -Wall -Wextra $< -o $@


# Execute single-core unit test under Tenstorrent ttsim emulator
sim: test_ttsim
	@if [ -z "$$TT_METAL_SIMULATOR" ]; then \
		export TT_METAL_SIMULATOR=$(TT_METAL_HOME)/sim/libttsim_bh.so; \
	fi; \
	if [ -z "$$TT_METAL_SIMULATOR_HOME" ]; then \
		export TT_METAL_SIMULATOR_HOME=$(TT_METAL_HOME)/sim; \
	fi; \
	export TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME); \
	export TT_METAL_SLOW_DISPATCH_MODE=1; \
	echo "[SIMULATION] Launching $(TTSIM_TEST_TARGET) via ttsim ($$TT_METAL_SIMULATOR)..."; \
	./$(TTSIM_TEST_TARGET) $(ARGS)

%.o: %.cpp
	@echo "[CXX] $<"
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

clean:
	@echo "[CLEAN]"
	rm -f $(OBJS) $(TARGET) $(HOST_TEST_TARGET) $(TTSIM_TEST_TARGET)