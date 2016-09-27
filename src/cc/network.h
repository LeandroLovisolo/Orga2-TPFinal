#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>

template<class Matrix>
class Network {
 public:
  typedef Matrix Vector;
  typedef std::vector<std::pair<Vector, Vector>> TrainingData;

  Network(const std::vector<int> &sizes);
  Vector FeedForward(const Vector& input);
  void SGD(const TrainingData& training_data, int epochs, int mini_batch_size,
           double eta);
  void SGD(const TrainingData& training_data, const TrainingData& test_data,
           int epochs, int mini_batch_size, double eta);

  int num_layers;
  std::vector<int> sizes;
  std::vector<Matrix> weights;
  std::vector<Vector> biases;

  // NOT PART OF PUBLIC API - METHODS BELOW MARKED PUBLIC FOR TESTING PURPOSES
  Vector Sigmoid_(const Vector& input);
  Vector SigmoidPrime_(const Vector& input);
  std::pair<std::vector<Matrix>,
            std::vector<Network<Matrix>::Vector>> Backpropagation_(
                const Vector &input, const Vector &expected);
  void UpdateMiniBatch_(const TrainingData& minibatch, double eta);
  TrainingData GetMiniBatch_(const TrainingData& training_data,
                             int mini_batch_size, int n);
  virtual void Shuffle_(TrainingData& data);
  int Evaluate_(const TrainingData& test_data);
};

template<class Matrix>
Network<Matrix>::Network(const std::vector<int> &sizes)
    : num_layers(sizes.size()), sizes(sizes) {
  // We need at least two layers (input and output)
  assert(sizes.size() >= 2);

  // Initialize bias vectors
  for(int i = 1; i < num_layers; i++) {
    biases.push_back(Vector(sizes[i], 1));
    biases.back().Random();
  }

  // Initialize weight matrices
  for(int i = 0; i < num_layers - 1; i++) {
    weights.push_back(Matrix(sizes[i+1], sizes[i]));
    weights.back().Random();
  }
}

template<class Matrix>
typename Network<Matrix>::Vector Network<Matrix>::FeedForward(
    const Network<Matrix>::Vector &input) {
  Vector a(input);
  for(int i = 0; i < num_layers - 1; i++) {
    a = Sigmoid_((weights[i] * a) + biases[i]);
  }
  return a;
}

double sigmoid_unary(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template<class Matrix>
typename Network<Matrix>::Vector Network<Matrix>::Sigmoid_(
    const Network<Matrix>::Vector& input) {
  return input.ApplyFn(std::ptr_fun(sigmoid_unary));
}

template<class Matrix>
typename Network<Matrix>::Vector Network<Matrix>::SigmoidPrime_(
    const Network<Matrix>::Vector &input) {
  return Sigmoid_(input).CoeffWiseProduct(
      Network<Matrix>::Vector(input.size()).Ones() - Sigmoid_(input));
}

template<class Matrix>
std::pair<std::vector<Matrix>, std::vector<typename Network<Matrix>::Vector>>
Network<Matrix>::Backpropagation_(const Network<Matrix>::Vector& input,
                                  const Network<Matrix>::Vector& expected) {
  // Initialize weights gradient with zeroes
  std::vector<Matrix> nabla_w;
  for(Matrix w : weights) {
    nabla_w.push_back(Matrix(w.rows(), w.cols()).Zeros());
  }

  // Initialize biases gradient with zeroes
  std::vector<Network<Matrix>::Vector> nabla_b;
  for(Network<Matrix>::Vector b : biases) {
    nabla_b.push_back(Network<Matrix>::Vector(b.size()).Zeros());
  }

  // Current activation (i.e. the raw input)
  Network<Matrix>::Vector activation = input;

  // Vector of activations, layer by layer
  std::vector<Network<Matrix>::Vector> activations { activation };

  // Vector of z vectors, layer by layer
  std::vector<Network<Matrix>::Vector> zs;

  // Feedforward
  for(int i = 0; i < num_layers - 1; i++) {
    Network<Matrix>::Vector z = (weights[i] * activation) + biases[i];
    zs.push_back(z);
    activation = Sigmoid_(z);
    activations.push_back(activation);
  }

  // Backward pass
  Network<Matrix>::Vector cost_derivative = activations.back() - expected;
  Network<Matrix>::Vector delta = cost_derivative.CoeffWiseProduct(
      SigmoidPrime_(zs.back()));
  nabla_b.back() = delta;
  nabla_w.back() = delta * activations[activations.size() - 2].Transpose();

  for(int l = 2; l < num_layers; l++) {
    Network<Matrix>::Vector z = zs[zs.size() - l];
    Network<Matrix>::Vector sp = SigmoidPrime_(z);
    delta = (weights[weights.size() - l + 1].Transpose() * delta)
        .CoeffWiseProduct(sp);
    nabla_b[nabla_b.size() - l] = delta;
    nabla_w[nabla_w.size() - l] =
        delta * activations[activations.size() - l - 1].Transpose();
  }

  return make_pair(nabla_w, nabla_b);
}

template<class Matrix>
void Network<Matrix>::UpdateMiniBatch_(
    const Network<Matrix>::TrainingData& minibatch, double eta) {
  // Initialize weights gradient with zeroes
  std::vector<Matrix> nabla_w;
  for(const Matrix& w : weights) {
    nabla_w.push_back(Matrix(w.rows(), w.cols()).Zeros());
  }

  // Initialize biases gradient with zeroes
  std::vector<Network<Matrix>::Vector> nabla_b;
  for(const Network<Matrix>::Vector b : biases) {
    nabla_b.push_back(Network<Matrix>::Vector(b.size()).Zeros());
  }

  // Accumulate gradients obtained from all input/output pairs
  for(auto p : minibatch) {
    Network<Matrix>::Vector &x = p.first;
    Network<Matrix>::Vector &y = p.second;
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

template<class Matrix>
typename Network<Matrix>::TrainingData Network<Matrix>::GetMiniBatch_(
    const Network<Matrix>::TrainingData& training_data,
    int mini_batch_size, int n) {
  if(n * mini_batch_size >= training_data.size()) {
    return Network<Matrix>::TrainingData();
  }
  return Network<Matrix>::TrainingData(
          training_data.begin() + n * mini_batch_size,
          training_data.begin() + std::min((n + 1) * mini_batch_size,
                                           (int) training_data.size()));
}

template<class Matrix>
void Network<Matrix>::Shuffle_(Network<Matrix>::TrainingData& data) {
  random_shuffle(data.begin(), data.end());
}

using namespace std;

template<class Matrix>
void Network<Matrix>::SGD(const Network<Matrix>::TrainingData& training_data,
                          int epochs, int mini_batch_size, double eta) {
  SGD(training_data, Network<Matrix>::TrainingData(),
      epochs, mini_batch_size, eta);
}

template<class Matrix>
void Network<Matrix>::SGD(const Network<Matrix>::TrainingData& training_data,
                          const Network<Matrix>::TrainingData& test_data,
                          int epochs, int mini_batch_size, double eta) {
  // Local copy of training data
  Network<Matrix>::TrainingData shuffled_training_data(training_data);

  for(int i = 0; i < epochs; i++) {
    // Shuffle local copy of training data
    Shuffle_(shuffled_training_data);

    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;

    // Iterate over all minibatches
    steady_clock::time_point t0 = steady_clock::now();
    for(int j = 0; j < shuffled_training_data.size(); j += mini_batch_size) {
      steady_clock::time_point t = steady_clock::now();

      // Report progress
      if(!test_data.empty()) {
        auto dt = duration_cast<microseconds>(t - t0).count();
        if(dt > 10000) {
          cout << "\r[Epoch " << (i + 1) << "] Minibatch "
               << (j / mini_batch_size + 1) << " / "
               << (shuffled_training_data.size() / mini_batch_size);
          t0 = steady_clock::now();
        }
      }

      // Get current minibatch
      Network<Matrix>::TrainingData mini_batch(
          shuffled_training_data.begin() + j,
          shuffled_training_data.begin() +
              min(j + mini_batch_size, (int) shuffled_training_data.size()));

      // Run gradient descent on current minibatch
      UpdateMiniBatch_(mini_batch, eta);
    }

    // Report progress
    if(!test_data.empty()) {
      double accuracy = (double) Evaluate_(test_data) / test_data.size() * 100;
      cout << "\r[Epoch " << (i + 1) << "] "
           << accuracy << "\% accuracy on test data." << endl;
    }
  }
}

template<class Matrix>
int Network<Matrix>::Evaluate_(const Network<Matrix>::TrainingData& test_data) {
  int total = 0;

  for(auto& p : test_data) {
    Network<Matrix>::Vector prediction = FeedForward(p.first);

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

#endif // __NETWORK_H__
