# ==============================================================================
# Makefile for 4-Codeword Circulant DMA LDPC Engine (engines/circulant_dma_4cw)
# Architecture: Line-Rate 512-bit DMA Bursts with Zero-Transpose SFPU Math
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

TARGET       := ldpc_sim_circulant_dma
SRCS         := ldpc_sim_circulant_dma.cpp
OBJS         := $(SRCS:.cpp=.o)

.PHONY: all clean run passthrough r_dataflow lut

all: $(TARGET)

lut: DEFINES += -DLDPC_USE_SFPU_LUT=1
lut: clean $(TARGET)
	@echo "[LUT BUILD] Compiled $(TARGET) with hardware SFPU LUT (LDPC_USE_SFPU_LUT=1)"

passthrough: DEFINES += -DLDPC_TEST_PASSTHROUGH_MODE=1
passthrough: clean $(TARGET)
	@echo "[PASSTHROUGH BUILD] Compiled $(TARGET) with LDPC_TEST_PASSTHROUGH_MODE=1"

r_dataflow: DEFINES += -DLDPC_TEST_R_DATAFLOW_MODE=1
r_dataflow: clean $(TARGET)
	@echo "[R_DATAFLOW BUILD] Compiled $(TARGET) with LDPC_TEST_R_DATAFLOW_MODE=1"

$(TARGET): $(OBJS)
	@echo "[LINK] $@"
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LIBDIRS) $(LIBS)

%.o: %.cpp
	@echo "[CXX] $<"
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

run: $(TARGET)
	@echo "[RUN] Executing $(TARGET) on QuietBox 2..."
	./$(TARGET) 2.0 25 16

clean:
	@echo "[CLEAN]"
	rm -f $(OBJS) $(TARGET)
