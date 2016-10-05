#include <iostream>

#include "matrix.h"
#include "mnist_loader.h"
#include "network.h"
#include "optparse.h"

constexpr char kTrainImagesPath[] = "../../data/train-images-idx3-ubyte";
constexpr char kTrainLabelsPath[] = "../../data/train-labels-idx1-ubyte";
constexpr char kTestImagesPath[] = "../../data/t10k-images-idx3-ubyte";
constexpr char kTestLabelsPath[] = "../../data/t10k-labels-idx1-ubyte";

using namespace std;

template<class Matrix, class Vector=Matrix>
TrainingData<Vector> to_training_data(const LabelledMistData& mnist_data) {
  TrainingData<Vector> training_data;
  for(pair<vector<uchar>, uchar> p : mnist_data) {
    Vector input(p.first.size());
    for(uint i = 0; i < p.first.size(); i++) input(i) = (float) p.first[i];
    Vector output(10);
    output.Zeros();
    output(p.second) = 1.0;
    training_data.push_back(make_pair(input, output));
  }
  return training_data;
}

template<class Matrix, class Vector=Matrix>
void TrainNetwork(const string& matrix_impl_name) {
  MnistLoader loader(kTrainImagesPath, kTrainLabelsPath,
                     kTestImagesPath, kTestLabelsPath);

  TrainingData<Vector> training_data =
      to_training_data<Matrix>(loader.train_data());
  TrainingData<Vector> test_data =
      to_training_data<Matrix>(loader.test_data());

  cout << "Training with matrix implementation: " << matrix_impl_name << endl;
  Network<Matrix> nn(vector<int> { 784, 30, 10 });
  nn.SGD(training_data, test_data, 100, 50, 0.1);
  cout << "Training finished." << endl;
}

int main(int argc, char **argv) {
  optparse::OptionParser parser =
    optparse::OptionParser().description("Neural network training launcher");

  parser.add_option("-m", "--matrix").dest("matrix")
    .help("Matrix implementation to be used. Possible values: "
          "naive, simd, eigen (default).")
    .set_default("eigen")
    .metavar("MATRIX_IMPL");

  const optparse::Values options = parser.parse_args(argc, argv);
  string impl = options["matrix"];

  if(impl == "naive") {
    TrainNetwork<NaiveMatrix>(impl);
  } else if(impl == "simd") {
    TrainNetwork<SimdMatrix>(impl);
  } else if(impl == "eigen") {
    TrainNetwork<EigenMatrix>(impl);
  } else {
    cerr << "Invalid MATRIX_IMPL. Run with --help to see valid options."
         << endl;
  }

  return 0;
}
