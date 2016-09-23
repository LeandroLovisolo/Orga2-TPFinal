#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <vector>
#include <utility>
#include <cmath>

#include "Eigen/Dense"
#include "gtest/gtest_prod.h"

typedef std::vector<std::pair<std::vector<double>,
                              std::vector<double>>> labelled_data;

typedef std::vector<std::pair<Eigen::VectorXd,
                              Eigen::VectorXd>> labelled_eigen_data;

class Network {
 public:
  Network(const std::vector<int> &sizes);
  std::vector<double> FeedForward(const std::vector<double> &input);
  void SGD(const labelled_data& training_data, int epochs, int mini_batch_size,
           double eta);
  void SGD(const labelled_data& training_data, const labelled_data& test_data,
           int epochs, int mini_batch_size, double eta);

  int num_layers;
  std::vector<int> sizes;
  std::vector<Eigen::MatrixXd> weights;
  std::vector<Eigen::VectorXd> biases;

 protected:
  virtual void Shuffle_(labelled_eigen_data& data);

 private:
  Eigen::VectorXd Sigmoid_(const Eigen::VectorXd &input);
  Eigen::VectorXd SigmoidPrime_(const Eigen::VectorXd &input);
  std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
  Backpropagation_(const Eigen::VectorXd &input,
                   const Eigen::VectorXd &expected);
  void UpdateMiniBatch_(const labelled_eigen_data& minibatch, double eta);
  labelled_eigen_data GetMiniBatch_(labelled_eigen_data& training_data,
                                    int mini_batch_size, int n);
  int Evaluate_(const labelled_data& test_data);

  FRIEND_TEST(NetworkTest, NumLayers);
  FRIEND_TEST(NetworkTest, Sizes); 
  FRIEND_TEST(NetworkTest, Biases);
  FRIEND_TEST(NetworkTest, Weights);
  FRIEND_TEST(NetworkTest, Sigmoid);
  FRIEND_TEST(NetworkTest, SigmoidPrime);
  FRIEND_TEST(NetworkTest, FeedForward);
  FRIEND_TEST(NetworkTest, Backpropagation);
  FRIEND_TEST(NetworkTest, UpdateMiniBatch);
  FRIEND_TEST(NetworkTest, GetMiniBatch);
};

#endif // __NETWORK_H__
