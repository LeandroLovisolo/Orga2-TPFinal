#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>

template<class Vector>
using TrainingData = std::vector<std::pair<Vector, Vector>>;

template<class Matrix, class Vector=Matrix>
class Network {
 public:
  Network(const std::vector<int> &sizes);
  Network(const std::string& checkpoint);
  Vector FeedForward(const Vector& input);
  void SGD(const TrainingData<Vector>& training_data,
           int epochs, int mini_batch_size, float eta);
  void SGD(const TrainingData<Vector>& training_data,
           const TrainingData<Vector>& test_data,
           const std::string& stats_file,
           int epochs, int mini_batch_size, float eta);
  std::string SaveCheckpoint();

  int num_layers;
  std::vector<int> sizes;
  std::vector<Matrix> weights;
  std::vector<Vector> biases;

  // NOT PART OF PUBLIC API - METHODS BELOW MARKED PUBLIC FOR TESTING PURPOSES
  void LoadCheckpoint_(const std::string& checkpoint);
  Vector Sigmoid_(const Vector& input);
  Vector SigmoidPrime_(const Vector& input);
  std::pair<std::vector<Matrix>, std::vector<Vector>> Backpropagation_(
      const Vector &input, const Vector &expected);
  void UpdateMiniBatch_(const TrainingData<Vector>& minibatch, float eta);
  TrainingData<Vector> GetMiniBatch_(const TrainingData<Vector>& training_data,
                                     int mini_batch_size, int n);
  virtual void Shuffle_(TrainingData<Vector>& data);
  int Evaluate_(const TrainingData<Vector>& test_data);
};

template<class Matrix, class Vector>
Network<Matrix, Vector>::Network(const std::vector<int> &sizes)
    : num_layers(sizes.size()), sizes(sizes) {
  // We need at least two layers (input and output)
  assert(sizes.size() >= 2);

  // Initialize bias vectors
  for(int i = 1; i < num_layers; i++) {
    biases.push_back(Vector(sizes[i]));
    biases.back().Random();
  }

  // Initialize weight matrices
  for(int i = 0; i < num_layers - 1; i++) {
    weights.push_back(Matrix(sizes[i+1], sizes[i]));
    weights.back().Random();
  }
}

template<class Matrix, class Vector>
Network<Matrix, Vector>::Network(const std::string& checkpoint) {
  LoadCheckpoint_(checkpoint);
}

template<class Matrix, class Vector>
void Network<Matrix, Vector>::LoadCheckpoint_(const std::string& checkpoint) {
  std::stringstream ss(checkpoint);

  // Number of layers
  ss >> num_layers;

  // Layer sizes
  sizes = std::vector<int>(num_layers);
  for(int i = 0; i < num_layers; i++) ss >> sizes[i];

  // Weights and biases
  weights.clear();
  biases.clear();
  for(int l = 0; l < num_layers - 1; l++) {
    // Weights
    weights.push_back(Matrix(sizes[l + 1], sizes[l]));
    for(int i = 0; i < sizes[l + 1]; i++) {
      for(int j = 0; j < sizes[l]; j++) {
        ss >> weights[l](i, j);
      }
    }
    // Biases
    biases.push_back(Vector(sizes[l + 1]));
    for(int i = 0; i < sizes[l + 1]; i++) {
      ss >>  biases[l](i);
    }
  }
}

template<class Matrix, class Vector>
std::string Network<Matrix, Vector>::SaveCheckpoint() {
  std::stringstream checkpoint;

  // Number of layers
  checkpoint << num_layers;

  // Layer sizes
  for(int size : sizes) checkpoint << " " << size;

  for(int l = 0; l < num_layers - 1; l++) {
    // Weights
    for(int i = 0; i < sizes[l + 1]; i++) {
      for(int j = 0; j < sizes[l]; j++) {
        checkpoint << " " << weights[l](i, j);
      }
    }

    // Biases
    for(int i = 0; i < sizes[l + 1]; i++) {
      checkpoint << " " << biases[l](i);
    }
  }

  return checkpoint.str();
}

template<class Matrix, class Vector>
Vector Network<Matrix, Vector>::FeedForward(const Vector &input) {
  Vector a(input);
  for(int i = 0; i < num_layers - 1; i++) {
    a = Sigmoid_((weights[i] * a) + biases[i]);
  }
  return a;
}

float sigmoid_unary(float x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template<class Matrix, class Vector>
Vector Network<Matrix, Vector>::Sigmoid_(const Vector& input) {
  return input.ApplyFn(std::ptr_fun(sigmoid_unary));
}

template<class Matrix, class Vector>
Vector Network<Matrix, Vector>::SigmoidPrime_(const Vector &input) {
  return Sigmoid_(input).CoeffWiseProduct(
      Vector(input.size()).Ones() - Sigmoid_(input));
}

template<class Matrix, class Vector>
std::pair<std::vector<Matrix>, std::vector<Vector>>
Network<Matrix, Vector>::Backpropagation_(const Vector& input,
                                          const Vector& expected) {
  // Initialize weights gradient with zeroes
  std::vector<Matrix> nabla_w;
  for(Matrix w : weights) {
    nabla_w.push_back(Matrix(w.rows(), w.cols()).Zeros());
  }

  // Initialize biases gradient with zeroes
  std::vector<Vector> nabla_b;
  for(Vector b : biases) {
    nabla_b.push_back(Vector(b.size()).Zeros());
  }

  // Current activation (i.e. the raw input)
  Vector activation = input;

  // Vector of activations, layer by layer
  std::vector<Vector> activations { activation };

  // Vector of z vectors, layer by layer
  std::vector<Vector> zs;

  // Feedforward
  for(int i = 0; i < num_layers - 1; i++) {
    Vector z = (weights[i] * activation) + biases[i];
    zs.push_back(z);
    activation = Sigmoid_(z);
    activations.push_back(activation);
  }

  // Backward pass
  Vector cost_derivative = activations.back() - expected;
  Vector delta = cost_derivative.CoeffWiseProduct(SigmoidPrime_(zs.back()));
  nabla_b.back() = delta;
  nabla_w.back() = delta * activations[activations.size() - 2].Transpose();

  for(int l = 2; l < num_layers; l++) {
    Vector z = zs[zs.size() - l];
    Vector sp = SigmoidPrime_(z);
    delta = (weights[weights.size() - l + 1].Transpose() * delta)
        .CoeffWiseProduct(sp);
    nabla_b[nabla_b.size() - l] = delta;
    nabla_w[nabla_w.size() - l] =
        delta * activations[activations.size() - l - 1].Transpose();
  }

  return make_pair(nabla_w, nabla_b);
}

template<class Matrix, class Vector>
void Network<Matrix, Vector>::UpdateMiniBatch_(
    const TrainingData<Vector>& minibatch, float eta) {
  // Initialize weights gradient with zeroes
  std::vector<Matrix> nabla_w;
  for(const Matrix& w : weights) {
    nabla_w.push_back(Matrix(w.rows(), w.cols()).Zeros());
  }

  // Initialize biases gradient with zeroes
  std::vector<Vector> nabla_b;
  for(const Vector b : biases) {
    nabla_b.push_back(Vector(b.size()).Zeros());
  }

  // Accumulate gradients obtained from all input/output pairs
  for(auto p : minibatch) {
    Vector &x = p.first;
    Vector &y = p.second;
    auto delta_nablas = Backpropagation_(x, y);
    for(int i = 0; i < num_layers - 1; i++) {
      nabla_w[i] += delta_nablas.first[i];
      nabla_b[i] += delta_nablas.second[i];
    }
  }

  // Update weights
  for(int i = 0; i < num_layers - 1; i++) {
    weights[i] -= nabla_w[i] * (eta / minibatch.size());
  }

  // Update biases
  for(int i = 0; i < num_layers - 1; i++) {
    biases[i] -= nabla_b[i] * (eta / minibatch.size());
  }
}

template<class Matrix, class Vector>
TrainingData<Vector> Network<Matrix, Vector>::GetMiniBatch_(
    const TrainingData<Vector>& training_data,
    int mini_batch_size, int n) {
  if(n * mini_batch_size >= training_data.size()) {
    return TrainingData<Vector>();
  }
  return TrainingData<Vector>(
          training_data.begin() + n * mini_batch_size,
          training_data.begin() + std::min((n + 1) * mini_batch_size,
                                           (int) training_data.size()));
}

template<class Matrix, class Vector>
void Network<Matrix, Vector>::Shuffle_(TrainingData<Vector>& data) {
  random_shuffle(data.begin(), data.end());
}

template<class Matrix, class Vector>
void Network<Matrix, Vector>::SGD(const TrainingData<Vector>& training_data,
                                  int epochs, int mini_batch_size, float eta) {
  SGD(training_data, TrainingData<Vector>(), "", epochs, mini_batch_size, eta);
}

template<class Matrix, class Vector>
void Network<Matrix, Vector>::SGD(const TrainingData<Vector>& training_data,
                                  const TrainingData<Vector>& test_data,
                                  const std::string& stats_file,
                                  int epochs, int mini_batch_size, float eta) {
  using std::chrono::steady_clock;
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  using std::chrono::seconds;

  // Local copy of training data
  TrainingData<Vector> shuffled_training_data(training_data);

  // Statistics
  steady_clock::time_point total_t0 = steady_clock::now();

  for(int i = 0; i < epochs; i++) {
    // Shuffle local copy of training data
    Shuffle_(shuffled_training_data);

    // Iterate over all minibatches
    steady_clock::time_point t0 = steady_clock::now();
    for(int j = 0; j < shuffled_training_data.size(); j += mini_batch_size) {
      steady_clock::time_point t = steady_clock::now();

      // Report progress
      if(!test_data.empty()) {
        auto dt = duration_cast<microseconds>(t - t0).count();
        if(dt > 10000) {
          std::cout << "\r[Epoch " << (i + 1) << "] Minibatch "
                    << (j / mini_batch_size + 1) << " / "
                    << (shuffled_training_data.size() / mini_batch_size);
          t0 = steady_clock::now();
        }
      }

      // Get current minibatch
      TrainingData<Vector> mini_batch(
          shuffled_training_data.begin() + j,
          shuffled_training_data.begin() +
              std::min(j + mini_batch_size,
                       (int) shuffled_training_data.size()));

      // Run gradient descent on current minibatch
      UpdateMiniBatch_(mini_batch, eta);
    }

    // Report progress
    if(!test_data.empty()) {
      float accuracy = (float) Evaluate_(test_data) / test_data.size() * 100;
      std::cout << "\r[Epoch " << (i + 1) << "] "
                << accuracy << "\% accuracy on test data." << std::endl;
    }
  }

  // Statistics
  auto total_t = duration_cast<seconds>(steady_clock::now() - total_t0).count();
  auto epoch_avg = (double) total_t / epochs;
  std::cout << "Total training time: " << total_t << std::endl
            << "Average epoch time: " << epoch_avg << std::endl;
  if(!stats_file.empty()) {
    std::ofstream file(stats_file);
    file << "total_training_time: " << total_t << std::endl
         << "avg_epoch_time: " << epoch_avg << std::endl;
    file.close();
  }
}

template<class Matrix, class Vector>
int Network<Matrix, Vector>::Evaluate_(const TrainingData<Vector>& test_data) {
  int total = 0;

  for(auto& p : test_data) {
    Vector prediction = FeedForward(p.first);

    int predicted = 0;
    for(int i = 0; i < prediction.size(); i++) {
      if(prediction(i) > prediction(predicted)) predicted = i;
    }

    int expected = 0;
    for(int i = 0; i < p.second.size(); i++) {
      if(p.second(i) > p.second(expected)) expected = i;
    }

    if(predicted == expected) total++;
  }

  return total;
}

#endif  // __NETWORK_H__
