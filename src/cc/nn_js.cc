#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <emscripten.h>

#include "network.h"
#include "matrix.h"

using namespace std;

Network<NaiveMatrix> *nn;

NaiveMatrix softmax(const NaiveMatrix& m) {
  NaiveMatrix res(m);
  float denominator = 0;
  for(int i = 0; i < m.size(); i++) denominator += exp(m(i));
  for(int i = 0; i < m.size(); i++) {
    res(i) = exp(m(i)) / denominator;
  }
  return res;;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void nn_init() {
  ifstream f("checkpoint");
  string checkpoint((istreambuf_iterator<char>(f)),
                     istreambuf_iterator<char>());
  nn = new Network<NaiveMatrix>(checkpoint);
}

EMSCRIPTEN_KEEPALIVE
void nn_eval(float *input, float *output) {
  NaiveMatrix inputv(784);
  for(int i = 0; i < 28 * 28; i++) inputv(i) = input[i];
  NaiveMatrix outputv = nn->FeedForward(inputv);
  outputv = softmax(outputv);
  for(int i = 0; i < 10; i++) output[i] = outputv(i);
}

}
