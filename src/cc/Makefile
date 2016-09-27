MAIN            = nn
MAIN_SRC        = nn.cc matrix.cc mnist_loader.cc

NN_TEST         = network_test
NN_TEST_SRC     = network_test.cc matrix.cc

MATRIX_TEST     = matrix_test
MATRIX_TEST_SRC = matrix_test.cc matrix.cc

TEST_DEPS       = gtest-all.o gmock-all.o gtest_main.o

BINARIES        = $(MAIN) $(NN_TEST) $(MATRIX_TEST)

################################################################################

LIB_DIR   = ../../lib
EIGEN_DIR = ../../lib/eigen
GTEST_DIR = ../../lib/googletest/googletest
GMOCK_DIR = ../../lib/googletest/googlemock

CXX       = clang++
CXXFLAGS := $(CXXFLAGS) -std=c++14 \
            -I $(LIB_DIR) \
            -I $(EIGEN_DIR) \
            -isystem $(GTEST_DIR)/include \
            -isystem $(GMOCK_DIR)/include \
            -I $(GTEST_DIR) -I $(GMOCK_DIR) \
            -O3 -fopenmp
LD        = $(CXX)
LDFLAGS   = $(CXXFLAGS)

################################################################################

TRAIN_IMAGES = ../../data/train-images-idx3-ubyte
TRAIN_LABELS = ../../data/train-labels-idx1-ubyte
TEST_IMAGES = ../../data/t10k-images-idx3-ubyte
TEST_LABELS = ../../data/t10k-labels-idx1-ubyte

################################################################################

.PHONY: all clean data

all: data $(BINARIES)

$(MAIN): $(MAIN_SRC:.cc=.o)
	$(LD) $(LDFLAGS) -o $@ $^

$(NN_TEST): $(NN_TEST_SRC:.cc=.o) $(TEST_DEPS)
	$(LD) $(LDFLAGS) -pthread -o $@ $^

$(MATRIX_TEST): $(MATRIX_TEST_SRC:.cc=.o) $(TEST_DEPS)
	$(LD) $(LDFLAGS) -pthread -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -M -MF $(patsubst %.o, %.d, $@) $<
	$(CXX) $(CXXFLAGS) -c $<

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
