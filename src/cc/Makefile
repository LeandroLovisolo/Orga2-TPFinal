MAIN            = nn
MAIN_SRC        = nn.cc matrix.cc mnist_loader.cc

TESTS           = tests
TESTS_SRC       = matrix.cc matrix_test.cc simd_matrix_test.cc network_test.cc

TEST_DEPS       = gtest-all.o gmock-all.o gtest_main.o

MAIN_JS         = ../ui/nn.js
MAIN_JS_DATA    = ../ui/nn.data
MAIN_JS_SRC     = nn_js.cc matrix.cc

BINARIES        = $(MAIN) $(TESTS) $(MAIN_JS)

################################################################################

LIB_DIR   = ../../lib
EIGEN_DIR = ../../lib/eigen
GTEST_DIR = ../../lib/googletest/googletest
GMOCK_DIR = ../../lib/googletest/googlemock

CXX        = clang++
CXXOFLAG   = -O3
CXXFLAGS  := $(CXXFLAGS) -std=c++14 \
             -I $(LIB_DIR) \
             -I $(EIGEN_DIR) \
             -isystem $(GTEST_DIR)/include \
             -isystem $(GMOCK_DIR)/include \
             -I $(GTEST_DIR) -I $(GMOCK_DIR) \
             $(CXXOFLAG)
NASM       = nasm
NASM_FLAGS = -f elf64 -g -F dwarf
LD         = $(CXX)
LDFLAGS    = $(CXXFLAGS)
EMXX       = /usr/lib/emscripten/em++
EMXXFLAGS  = -std=c++14

################################################################################

.PHONY: all clean

all: $(BINARIES)

$(MAIN): $(MAIN_SRC:.cc=.o) simd_matrix.o
	$(LD) $(LDFLAGS) -o $@ $^

$(TESTS): $(TESTS_SRC:.cc=.o) simd_matrix.o $(TEST_DEPS)
	$(LD) $(LDFLAGS) -pthread -o $@ $^

$(MAIN_JS): $(MAIN_JS_SRC)
	$(EMXX) $(EMXXFLAGS) -o $@ $^ --preload-file checkpoint

%.o: %.cc
	$(CXX) $(CXXFLAGS) -M -MF $(patsubst %.o, %.d, $@) $<
	$(CXX) $(CXXFLAGS) -c $<

simd_matrix.o: simd_matrix.asm
	$(NASM) $(NASM_FLAGS) -o $@ $<

gtest-all.o:
	$(CXX) $(CXXFLAGS) -c ${GTEST_DIR}/src/gtest-all.cc -o gtest-all.o

gmock-all.o:
	$(CXX) $(CXXFLAGS) -c ${GMOCK_DIR}/src/gmock-all.cc -o gmock-all.o

gtest_main.o:
	$(CXX) $(CXXFLAGS) -c ${GTEST_DIR}/src/gtest_main.cc -o gtest_main.o

-include $(DEP)

clean:
	rm -f $(MAIN) $(TESTS) *.o *.d

clean-ui:
	rm -f $(MAIN_JS) $(MAIN_JS_DATA)
