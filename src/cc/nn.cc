#include <iostream>
#include <Eigen/Dense>

#include "mnist_loader.h"
#include "network.h"

using namespace Eigen;
using namespace std;

labelled_data to_labelled_data(const labelled_mnist_data& data) {
  labelled_data data_;
  for(pair<vector<uchar>, uchar> p : data) {
    vector<double> input;
    for(uchar pixel : p.first) input.push_back((double) pixel);
    vector<double> output(10, 0);
    output[p.second] = 1.0;
    data_.push_back(make_pair(input, output));
  }
  return data_;
}

int main() {
  MnistLoader loader("../../data/train-images-idx3-ubyte",
                     "../../data/train-labels-idx1-ubyte",
                     "../../data/t10k-images-idx3-ubyte",
                     "../../data/t10k-labels-idx1-ubyte");

  cout << "Loading data..." << endl;
  labelled_data training_data = to_labelled_data(loader.train_data());
  labelled_data test_data = to_labelled_data(loader.test_data());

  Network nn(vector<int> { 784, 30, 10 });

  cout << "Starting training..." << endl;
  nn.SGD(training_data, test_data, 100, 50, 0.1);
  cout << "Training finished." << endl;

  return 0;
}
