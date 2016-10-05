MAIN            = nn
MAIN_SRC        = nn.cc matrix.cc mnist_loader.cc

TESTS           = tests
TESTS_SRC       = matrix.cc matrix_test.cc simd_matrix_test.cc network_test.cc

TEST_DEPS       = gtest-all.o gmock-all.o gtest_main.o

BINARIES        = $(MAIN) $(TESTS)

################################################################################

LIB_DIR   = ../../lib
EIGEN_DIR = ../../lib/eigen
GTEST_DIR = ../../lib/googletest/googletest
GMOCK_DIR = ../../lib/googletest/googlemock

CXX        = clang++
CXXFLAGS  := $(CXXFLAGS) -std=c++14 \
             -I $(LIB_DIR) \
             -I $(EIGEN_DIR) \
             -isystem $(GTEST_DIR)/include \
             -isystem $(GMOCK_DIR)/include \
             -I $(GTEST_DIR) -I $(GMOCK_DIR) \
             -O3
NASM       = nasm
NASM_FLAGS = -f elf64 -g -F dwarf
LD         = $(CXX)
LDFLAGS    = $(CXXFLAGS)

################################################################################

TRAIN_IMAGES = ../../data/train-images-idx3-ubyte
TRAIN_LABELS = ../../data/train-labels-idx1-ubyte
TEST_IMAGES = ../../data/t10k-images-idx3-ubyte
TEST_LABELS = ../../data/t10k-labels-idx1-ubyte

################################################################################

.PHONY: all clean data

all: data $(BINARIES)

$(MAIN): $(MAIN_SRC:.cc=.o) simd_matrix.o
	$(LD) $(LDFLAGS) -o $@ $^

$(TESTS): $(TESTS_SRC:.cc=.o) simd_matrix.o $(TEST_DEPS)
	$(LD) $(LDFLAGS) -pthread -o $@ $^

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

data: $(TRAIN_IMAGES) $(TRAIN_LABELS) $(TEST_IMAGES) $(TEST_LABELS)

$(TRAIN_IMAGES): ${TRAIN_IMAGES}.gz
	gunzip --keep $^

$(TRAIN_LABELS): ${TRAIN_LABELS}.gz
	gunzip --keep $^

$(TEST_IMAGES): ${TEST_IMAGES}.gz
	gunzip --keep $^

$(TEST_LABELS): ${TEST_LABELS}.gz
	gunzip --keep $^

clean:
	rm -f $(BINARIES) *.o *.d \
        $(TRAIN_IMAGES) $(TRAIN_LABELS) \
        $(TEST_IMAGES) $(TEST_LABELS)
