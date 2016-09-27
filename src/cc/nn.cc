#include <iostream>
#include <Eigen/Dense>

#include "matrix.h"
#include "mnist_loader.h"
#include "network.h"

using namespace Eigen;
using namespace std;

template<class Matrix, class Vector=Matrix>
TrainingData<Vector> to_training_data(const LabelledMistData& mnist_data) {
  TrainingData<Vector> training_data;
  for(pair<vector<uchar>, uchar> p : mnist_data) {
    Vector input(p.first.size());
    for(uint i = 0; i < p.first.size(); i++) input(i) = (double) p.first[i];
    Vector output(10);
    output.Zeros();
    output(p.second) = 1.0;
    training_data.push_back(make_pair(input, output));
  }
  return training_data;
}

template<class Matrix, class Vector=Matrix>
void TrainNetwork() {
  MnistLoader loader("../../data/train-images-idx3-ubyte",
                     "../../data/train-labels-idx1-ubyte",
                     "../../data/t10k-images-idx3-ubyte",
                     "../../data/t10k-labels-idx1-ubyte");

  cout << "Loading data..." << endl;

  TrainingData<Vector> training_data =
      to_training_data<Matrix>(loader.train_data());
  TrainingData<Vector> test_data =
      to_training_data<Matrix>(loader.test_data());

  Network<Matrix> nn(vector<int> { 784, 30, 10 });

  cout << "Starting training..." << endl;
  nn.SGD(training_data, test_data, 100, 50, 0.1);
  cout << "Training finished." << endl;
}

int main() {
  TrainNetwork<EigenMatrix>();
  return 0;
}
