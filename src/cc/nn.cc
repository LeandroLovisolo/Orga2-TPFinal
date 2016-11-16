#include <iostream>
#include <fstream>
#include <string>

#include "matrix.h"
#include "mnist_loader.h"
#include "network.h"
#include "optparse.h"

constexpr char kDefaultMatrixImpl[] = "simd";
constexpr int kDefaultNumEpochs = 10;

constexpr char kDefaultDataBasePath[] = "data/";
constexpr char kTrainImagesPath[] = "train-images-idx3-ubyte";
constexpr char kTrainLabelsPath[] = "train-labels-idx1-ubyte";
constexpr char kTestImagesPath[] = "t10k-images-idx3-ubyte";
constexpr char kTestLabelsPath[] = "t10k-labels-idx1-ubyte";

constexpr int kHiddenLayerSize = 30;

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

string join_path(const string& directory, const string& filename) {
  return directory + filename;
}

void TestCheckpointAccuracy(const string& data_base_path,
                            const string& checkpoint_path) {
  MnistLoader loader(join_path(data_base_path, kTrainImagesPath),
                     join_path(data_base_path, kTrainLabelsPath),
                     join_path(data_base_path, kTestImagesPath),
                     join_path(data_base_path, kTestLabelsPath));
  ifstream f(checkpoint_path);
  string checkpoint((istreambuf_iterator<char>(f)),
                     istreambuf_iterator<char>());
  Network<NaiveMatrix> network(checkpoint);
  TrainingData<NaiveMatrix> test_data =
      to_training_data<NaiveMatrix, NaiveMatrix>(loader.test_data());
  int total = network.Evaluate_(test_data);
  float accuracy = (float) total / test_data.size() * 100;
  cout << accuracy << "\% accuracy on test data." << endl
       << "(" << total << " out of " << test_data.size()
       << " digits correctly classified.)" << endl;
}

void ExportTrainingImage(const string& data_base_path, int index) {
  MnistLoader loader(join_path(data_base_path, kTrainImagesPath),
                     join_path(data_base_path, kTrainLabelsPath),
                     join_path(data_base_path, kTestImagesPath),
                     join_path(data_base_path, kTestLabelsPath));
  cout << loader.ImageToPpm(loader.train_data(), index);
}

template<class Matrix, class Vector=Matrix>
void TrainNetwork(const string& matrix_impl_name, int epochs,
                  const string& data_base_path,
                  const string& stats_file,
                  const string& checkpoint_file) {
  // Load training data
  MnistLoader loader(join_path(data_base_path, kTrainImagesPath),
                     join_path(data_base_path, kTrainLabelsPath),
                     join_path(data_base_path, kTestImagesPath),
                     join_path(data_base_path, kTestLabelsPath));
  TrainingData<Vector> training_data =
      to_training_data<Matrix>(loader.train_data());
  TrainingData<Vector> test_data =
      to_training_data<Matrix>(loader.test_data());

  // Perform training
  cout << "Training with matrix implementation: " << matrix_impl_name << endl;
  Network<Matrix> nn(vector<int> { 784, kHiddenLayerSize, 10 });
  nn.SGD(training_data, test_data, stats_file, epochs, 50, 0.1);
  cout << "Training finished." << endl;

  // Save checkpoint
  if(!checkpoint_file.empty()) {
    ofstream os(checkpoint_file);
    os << nn.SaveCheckpoint();
    os.close();
    cout << "Checkpoint saved to: " << checkpoint_file << endl;
  }
}

int main(int argc, char **argv) {
  optparse::OptionParser parser =
    optparse::OptionParser().description("Neural network training launcher");

  parser.add_option("-m", "--matrix").dest("matrix")
    .help("Matrix implementation to be used. Possible values: "
          "naive, simd (default), eigen.")
    .set_default(kDefaultMatrixImpl)
    .metavar("MATRIX_IMPL");
  parser.add_option("-n" "--num-epochs").dest("epochs")
    .help("Number of training epochs (default: 100)")
    .set_default(kDefaultNumEpochs)
    .metavar("EPOCHS");
  parser.add_option("-d", "--data").dest("data_base_path")
    .help("Directory with data files")
    .set_default(kDefaultDataBasePath)
    .metavar("DATA_DIR");
  parser.add_option("-s", "--stats").dest("stats_file")
    .help("File where to write out stats.")
    .set_default("")
    .metavar("STATS_PATH");
  parser.add_option("-o", "--output").dest("checkpoint_file")
    .help("File where to save training checkpoint")
    .set_default("")
    .metavar("OUTPUT_PATH");
  parser.add_option("-t", "--test-checkpoint").dest("test_checkpoint_file")
    .help("Load checkpoint from file, print out accuracy and exit.")
    .set_default("")
    .metavar("CHECKPOINT_PATH");
  parser.add_option("--export-training-image").dest("export_training_image")
    .help("Print out training image in PPM format and exit immediately.")
    .set_default("")
    .metavar("INDEX");

  const optparse::Values options = parser.parse_args(argc, argv);
  string impl = options["matrix"];
  int epochs = stoi(options["epochs"]);
  string data_base_path = options["data_base_path"];
  string stats_file = options["stats_file"];
  string checkpoint_file = options["checkpoint_file"];
  string test_checkpoint_file = options["test_checkpoint_file"];
  string export_training_image = options["export_training_image"];

  if(!test_checkpoint_file.empty()) {
    TestCheckpointAccuracy(data_base_path, test_checkpoint_file);
    return 0;
  }

  if(!export_training_image.empty()) {
    ExportTrainingImage(data_base_path, stoi(export_training_image));
    return 0;
  }

  if(impl == "naive") {
    TrainNetwork<NaiveMatrix>(impl, epochs, data_base_path,
                              stats_file, checkpoint_file);
  } else if(impl == "simd") {
    TrainNetwork<SimdMatrix>(impl, epochs, data_base_path,
                             stats_file, checkpoint_file);
  } else if(impl == "eigen") {
    TrainNetwork<EigenMatrix>(impl, epochs, data_base_path,
                              stats_file, checkpoint_file);
  } else {
    cerr << "Invalid MATRIX_IMPL. Run with --help to see valid options."
         << endl;
  }

  return 0;
}
