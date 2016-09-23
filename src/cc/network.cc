#include <algorithm>
#include <iostream>

#include "Eigen/Core"

#include "network.h"

using namespace std;
using namespace Eigen;

Network::Network(const vector<int> &sizes)
    : num_layers(sizes.size()),
      sizes(sizes) {
  // We need at least two layers (input and output)
  assert(sizes.size() >= 2);

  // Initialize bias vectors
  for(int i = 1; i < num_layers; i++) {
    biases.push_back(VectorXd(sizes[i]));
    biases.back().setRandom();
  }

  // Initialize weight matrices
  for(int i = 0; i < num_layers - 1; i++) {
    weights.push_back(MatrixXd(sizes[i+1], sizes[i]));
    weights.back().setRandom();
  }
}

VectorXd std2eigen(const vector<double> &v) {
  VectorXd v_(v.size());
  for(int i = 0; i < v.size(); i++) v_(i) = v[i];
  return v_;
}

vector<double> eigen2std(const VectorXd &v) {
  vector<double> v_(v.size());
  for(int i = 0; i < v.size(); i++) v_[i] = v(i);
  return v_;
}

vector<double> Network::FeedForward(const vector<double> &input) {
  VectorXd a = std2eigen(input);
  for(int i = 0; i < num_layers - 1; i++) {
    a = Sigmoid_((weights[i] * a) + biases[i]);
  }
  return eigen2std(a);
}

double sigmoid_unary(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

VectorXd Network::Sigmoid_(const VectorXd &input) {
  return input.unaryExpr(ptr_fun(sigmoid_unary));
}

VectorXd Network::SigmoidPrime_(const VectorXd &input) {
  return Sigmoid_(input).cwiseProduct(
      (VectorXd::Ones(input.size()) - Sigmoid_(input)));
}

pair<vector<MatrixXd>, vector<VectorXd>>
Network::Backpropagation_(const VectorXd &input, const VectorXd &expected) {
  // Initialize weights gradient with zeroes
  vector<MatrixXd> nabla_w;
  for(MatrixXd w : weights) {
    nabla_w.push_back(MatrixXd(w.rows(), w.cols()).setZero());
  }

  // Initialize biases gradient with zeroes
  vector<VectorXd> nabla_b;
  for(VectorXd b : biases) {
    nabla_b.push_back(VectorXd(b.size()).setZero());
  }

  // Current activation (i.e. the raw input)
  VectorXd activation = input;

  // Vector of activations, layer by layer
  vector<VectorXd> activations { activation };

  // Vector of z vectors, layer by layer
  vector<VectorXd> zs;

  // Feedforward
  for(int i = 0; i < num_layers - 1; i++) {
    VectorXd z = (weights[i] * activation) + biases[i];
    zs.push_back(z);
    activation = Sigmoid_(z);
    activations.push_back(activation);
  }

  // Backward pass
  VectorXd cost_derivative = activations.back() - expected;
  VectorXd delta = cost_derivative.cwiseProduct(SigmoidPrime_(zs.back()));
  nabla_b.back() = delta;
  nabla_w.back() = delta * activations[activations.size() - 2].transpose();

  for(int l = 2; l < num_layers; l++) {
    VectorXd z = zs[zs.size() - l];
    VectorXd sp = SigmoidPrime_(z);
    delta = (weights[weights.size() - l + 1].transpose() * delta)
            .cwiseProduct(sp);
    nabla_b[nabla_b.size() - l] = delta;
    nabla_w[nabla_w.size() - l] =
        delta * activations[activations.size() - l - 1].transpose();
  }

  return make_pair(nabla_w, nabla_b);
}

void Network::UpdateMiniBatch_(const labelled_eigen_data& minibatch,
                               double eta) {
  // Initialize weights gradient with zeroes
  vector<MatrixXd> nabla_w;
  for(MatrixXd w : weights) {
    nabla_w.push_back(MatrixXd(w.rows(), w.cols()).setZero());
  }

  // Initialize biases gradient with zeroes
  vector<VectorXd> nabla_b;
  for(VectorXd b : biases) {
    nabla_b.push_back(VectorXd(b.size()).setZero());
  }

  // Accumulate gradients obtained from all input/output pairs
  for(pair<VectorXd, VectorXd> p : minibatch) {
    VectorXd &x = p.first;
    VectorXd &y = p.second;
    pair<vector<MatrixXd>, vector<VectorXd>> delta_nablas =
        Backpropagation_(x, y);
    for(int i = 0; i < num_layers - 1; i++) {
      nabla_w[i] += delta_nablas.first[i];
      nabla_b[i] += delta_nablas.second[i];
    }
  }

  // Update weights
  for(int i = 0; i < num_layers - 1; i++) {
    weights[i] -= (eta / minibatch.size()) * nabla_w[i];
  }

  // Update biases
  for(int i = 0; i < num_layers - 1; i++) {
    biases[i] -= (eta / minibatch.size()) * nabla_b[i];
  }
}

void Network::SGD(const labelled_data& training_data, int epochs,
                  int mini_batch_size, double eta) {
  SGD(training_data, labelled_data(), epochs, mini_batch_size, eta);
}

labelled_eigen_data labelled_data_to_eigen_data(const labelled_data& data) {
  labelled_eigen_data eigen_data;
  for(pair<vector<double>, vector<double>> p : data) {
    eigen_data.push_back(make_pair(std2eigen(p.first), std2eigen(p.second)));
  }
  return eigen_data;
}

void Network::SGD(const labelled_data& training_data,
                  const labelled_data& test_data,
                  int epochs, int mini_batch_size, double eta) {
  // Local copy of training data
  labelled_eigen_data shuffled_training_data =
    labelled_data_to_eigen_data(training_data);

  for(int i = 0; i < epochs; i++) {
    // Shuffle local copy of training data
    Shuffle_(shuffled_training_data);

    // Iterate over all minibatches
    for(int j = 0; j < shuffled_training_data.size(); j += mini_batch_size) {
      // Report progress
      if(!test_data.empty()) {
        cout << "\r[Epoch " << (i + 1) << "] Minibatch "
             << (j / mini_batch_size + 1) << " / "
             << (shuffled_training_data.size() / mini_batch_size);
      }

      // Get current minibatch
      labelled_eigen_data mini_batch = labelled_eigen_data(
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

void Network::Shuffle_(labelled_eigen_data& data) {
  random_shuffle(data.begin(), data.end());
}

labelled_eigen_data Network::GetMiniBatch_(labelled_eigen_data& training_data,
                                           int mini_batch_size, int n) {
  if(n * mini_batch_size >= training_data.size()) return labelled_eigen_data();
  return labelled_eigen_data(
          training_data.begin() + n * mini_batch_size,
          training_data.begin() + min((n + 1) * mini_batch_size,
                                      (int) training_data.size()));
}

int Network::Evaluate_(const labelled_data& test_data) {
  int total = 0;

  for(auto& p : test_data) {
    vector<double> prediction = FeedForward(p.first);

    int predicted = 0;
    for(int i = 0; i < prediction.size(); i++) {
      if(prediction[i] > prediction[predicted]) predicted = i;
    }

    int expected = 0;
    for(int i = 0; i < p.second.size(); i++) {
      if(p.second[i] > p.second[expected]) expected = i;
    }

    if(predicted == expected) total++;
  }

  return total;
}
