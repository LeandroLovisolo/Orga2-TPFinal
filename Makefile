TRAIN_IMAGES = data/train-images-idx3-ubyte
TRAIN_LABELS = data/train-labels-idx1-ubyte
TEST_IMAGES  = data/t10k-images-idx3-ubyte
TEST_LABELS  = data/t10k-labels-idx1-ubyte

NUM_EPOCHS   = 100

################################################################################

.PHONY: all data ui plots informe publish clean clean-data \
        binaries binaryO0 binaryO1 binaryO2 binaryO3 \
        experiments experiments-naive experiments-simd experiments-eigen \
        experiment-naive-O0 experiment-naive-O1 \
        experiment-naive-O2 experiment-naive-O3 \
        experiment-simd-O0 experiment-simd-O1 \
        experiment-simd-O2 experiment-simd-O3 \
        experiment-eigen-O0 experiment-eigen-O1 \
        experiment-eigen-O2 experiment-eigen-O3

all: binaries

ui:
	make -C src/cc ui

bundle: clean
	mkdir Orga2-TPFinal
	mkdir Orga2-TPFinal/data
	cp data/mnist.pkl.gz Orga2-TPFinal/data
	cp -r lib Makefile README.md src stats Orga2-TPFinal
	tar zcf Orga2-TPFinal.tar.gz Orga2-TPFinal
	rm -rf Orga2-TPFinal

publish:
	git subtree push --prefix src/ui origin gh-pages

clean:
	make -C src/cc clean
	make -C src/tex clean
	rm -rf $(TRAIN_IMAGES) $(TRAIN_LABELS) \
         $(TEST_IMAGES) $(TEST_LABELS) \
         src/cc/nnO0 src/cc/nnO1 src/cc/nnO2 src/cc/nnO3 \
         src/__pycache__ src/python/*.pyc src/python/plot/*.pyc

clean-ui:
	make -C src/cc clean-ui

################################################################################

binaries: data binaryO0 binaryO1 binaryO2 binaryO3 default-binary-and-ui

default-binary-and-ui:
	make -C src/cc clean all

binaryO0:
	make -C src/cc clean nnO0 MAIN=nnO0 CXXOFLAG=-O0

binaryO1:
	make -C src/cc clean nnO1 MAIN=nnO1 CXXOFLAG=-O1

binaryO2:
	make -C src/cc clean nnO2 MAIN=nnO2 CXXOFLAG=-O2

binaryO3:
	make -C src/cc clean nnO3 MAIN=nnO3 CXXOFLAG=-O3

################################################################################

data: $(TRAIN_IMAGES) $(TRAIN_LABELS) $(TEST_IMAGES) $(TEST_LABELS)

${TRAIN_IMAGES}.gz:
	cd data; wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

${TRAIN_LABELS}.gz:
	cd data; wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

${TEST_IMAGES}.gz:
	cd data; wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

${TEST_LABELS}.gz:
	cd data; wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

$(TRAIN_IMAGES): ${TRAIN_IMAGES}.gz
	gunzip $^

$(TRAIN_LABELS): ${TRAIN_LABELS}.gz
	gunzip $^

$(TEST_IMAGES): ${TEST_IMAGES}.gz
	gunzip $^

$(TEST_LABELS): ${TEST_LABELS}.gz
	gunzip $^

clean-data:
	rm -f $(TRAIN_IMAGES) $(TRAIN_LABELS) $(TEST_IMAGES) $(TEST_LABELS)

################################################################################

experiments: experiments-naive experiments-simd experiments-eigen

experiments-naive: experiment-naive-O0 \
                   experiment-naive-O1 \
                   experiment-naive-O2 \
                   experiment-naive-O3

experiment-naive-O0:
	src/cc/nnO0 -d data -m naive -n $(NUM_EPOCHS) -s stats/naive-O0.txt

experiment-naive-O1:
	src/cc/nnO1 -d data -m naive -n $(NUM_EPOCHS) -s stats/naive-O1.txt

experiment-naive-O2:
	src/cc/nnO2 -d data -m naive -n $(NUM_EPOCHS) -s stats/naive-O2.txt

experiment-naive-O3:
	src/cc/nnO3 -d data -m naive -n $(NUM_EPOCHS) -s stats/naive-O3.txt

experiments-simd: experiment-simd-O0 \
                  experiment-simd-O1 \
                  experiment-simd-O2 \
                  experiment-simd-O3

experiment-simd-O0:
	src/cc/nnO0 -d data -m simd -n $(NUM_EPOCHS) -s stats/simd-O0.txt

experiment-simd-O1:
	src/cc/nnO1 -d data -m simd -n $(NUM_EPOCHS) -s stats/simd-O1.txt

experiment-simd-O2:
	src/cc/nnO2 -d data -m simd -n $(NUM_EPOCHS) -s stats/simd-O2.txt

experiment-simd-O3:
	src/cc/nnO3 -d data -m simd -n $(NUM_EPOCHS) -s stats/simd-O3.txt

experiments-eigen: experiment-eigen-O0 \
                   experiment-eigen-O1 \
                   experiment-eigen-O2 \
                   experiment-eigen-O3

experiment-eigen-O0:
	src/cc/nnO0 -d data -m eigen -n $(NUM_EPOCHS) -s stats/eigen-O0.txt

experiment-eigen-O1:
	src/cc/nnO1 -d data -m eigen -n $(NUM_EPOCHS) -s stats/eigen-O1.txt

experiment-eigen-O2:
	src/cc/nnO2 -d data -m eigen -n $(NUM_EPOCHS) -s stats/eigen-O2.txt

experiment-eigen-O3:
	src/cc/nnO3 -d data -m eigen -n $(NUM_EPOCHS) -s stats/eigen-O3.txt

################################################################################

plots:
	src/python/plot/plot.py -p total-training-time -d stats \
	                        -o src/tex/total-training-time.pdf
	src/python/plot/plot.py -p avg-epoch-time -d stats \
	                        -o src/tex/avg-epoch-time.pdf

informe: plots
	make -C src/tex all
