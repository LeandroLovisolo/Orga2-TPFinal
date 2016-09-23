#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include "ini.h"
#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "network.h"

using namespace std;
using namespace Eigen;

class MockNetwork : public Network {
 public:
  MockNetwork(const std::vector<int> &sizes) : Network(sizes) {}
 protected:
  void Shuffle_(labelled_eigen_data& data) {
    if(data.size() == 0) return;
    auto first = data[0];
    data.erase(data.begin());
    data.push_back(first);
  }

  FRIEND_TEST(NetworkTest, MockShuffle);
};

class NetworkTest : public ::testing::Test {
 public:
  NetworkTest() {
    LoadTestData();
  }

  virtual void SetUp() {
    nn = make_unique<MockNetwork>(vector<int> { 2, 3, 1 });
    nn->weights[0] << t["layer1.w00"], t["layer1.w01"],
                      t["layer1.w10"], t["layer1.w11"],
                      t["layer1.w20"], t["layer1.w21"];
    nn->weights[1] << t["layer2.w00"], t["layer2.w01"], t["layer2.w02"];
    nn->biases[0]  << t["layer1.b0"], t["layer1.b1"], t["layer1.b2"];
    nn->biases[1]  << t["layer2.b0"];
  }

  void LoadTestData() {
    ifstream fs("../testdata");
    INI::Parser p(fs);
    for(auto const& kv : p.top()("default").values) {
      t[kv.first] = stod(kv.second);
    }
  }

  unique_ptr<MockNetwork> nn;
  unordered_map<string, double> t;
};

TEST_F(NetworkTest, NumLayers) {
  EXPECT_EQ(3, nn->num_layers);
}

TEST_F(NetworkTest, Sizes) {
  EXPECT_EQ((vector<int> { 2, 3, 1 }), nn->sizes);
}

TEST_F(NetworkTest, Biases) {
  ASSERT_EQ(2, nn->biases.size());
  EXPECT_EQ(3, nn->biases[0].rows());
  EXPECT_EQ(1, nn->biases[0].cols());
  EXPECT_EQ(1, nn->biases[1].rows());
  EXPECT_EQ(1, nn->biases[1].cols());
}

TEST_F(NetworkTest, Weights) {
  ASSERT_EQ(2, nn->weights.size());
  EXPECT_EQ(3, nn->weights[0].rows());
  EXPECT_EQ(2, nn->weights[0].cols());
  EXPECT_EQ(1, nn->weights[1].rows());
  EXPECT_EQ(3, nn->weights[1].cols());
}

TEST_F(NetworkTest, FeedForward) {
  vector<double> result = nn->FeedForward(
      vector<double> { t["feedforward.input1"], t["feedforward.input2"] });
  ASSERT_EQ(1, result.size());
  double precision = pow(10, -t["feedforward.output.precision"]);
  EXPECT_NEAR(t["feedforward.output"], result[0], precision);
}

TEST_F(NetworkTest, Sigmoid) {
  VectorXd v(3);
  v << t["sigmoid.input1"], t["sigmoid.input2"], t["sigmoid.input3"];
  VectorXd res = nn->Sigmoid_(v);
  double precision1 = pow(10, -t["sigmoid.output1.precision"]);
  double precision2 = pow(10, -t["sigmoid.output2.precision"]);
  double precision3 = pow(10, -t["sigmoid.output3.precision"]);
  EXPECT_NEAR(t["sigmoid.output1"], res(0), precision1);
  EXPECT_NEAR(t["sigmoid.output2"], res(1), precision2);
  EXPECT_NEAR(t["sigmoid.output3"], res(2), precision3);
}

TEST_F(NetworkTest, SigmoidPrime) {
  VectorXd v(3);
  v << t["sigmoid_prime.input1"],
       t["sigmoid_prime.input2"],
       t["sigmoid_prime.input3"];
  VectorXd res = nn->SigmoidPrime_(v);
  double precision1 = pow(10, -t["sigmoid_prime.output1.precision"]);
  double precision2 = pow(10, -t["sigmoid_prime.output2.precision"]);
  double precision3 = pow(10, -t["sigmoid_prime.output3.precision"]);
  EXPECT_NEAR(t["sigmoid_prime.output1"], res(0), precision1);
  EXPECT_NEAR(t["sigmoid_prime.output2"], res(1), precision2);
  EXPECT_NEAR(t["sigmoid_prime.output3"], res(2), precision3);
}

TEST_F(NetworkTest, Backpropagation) {
  VectorXd input(2);
  input << 1, 2;
  VectorXd expected(1);
  expected << 0;
  pair<vector<MatrixXd>, vector<VectorXd>> gradients =
      nn->Backpropagation_(input, expected);
  vector<MatrixXd> nabla_w = gradients.first;
  vector<VectorXd> nabla_b = gradients.second;

  // Make sure we've got the right number of gradient vectors
  ASSERT_EQ(2, nabla_w.size());
  ASSERT_EQ(2, nabla_b.size());

  // First layer weights
  ASSERT_EQ(3, nabla_w[0].rows());
  ASSERT_EQ(2, nabla_w[0].cols());
  double precision = pow(10, -t["layer1.nabla_w.precision"]);
  EXPECT_NEAR(t["layer1.nabla_w00"], nabla_w[0](0, 0), precision);
  EXPECT_NEAR(t["layer1.nabla_w01"], nabla_w[0](0, 1), precision);
  EXPECT_NEAR(t["layer1.nabla_w10"], nabla_w[0](1, 0), precision);
  EXPECT_NEAR(t["layer1.nabla_w11"], nabla_w[0](1, 1), precision);
  EXPECT_NEAR(t["layer1.nabla_w20"], nabla_w[0](2, 0), precision);
  EXPECT_NEAR(t["layer1.nabla_w21"], nabla_w[0](2, 1), precision);

  // Second layer weights
  ASSERT_EQ(1, nabla_w[1].rows());
  ASSERT_EQ(3, nabla_w[1].cols());
  precision = pow(10, -t["layer2.nabla_w.precision"]);
  EXPECT_NEAR(t["layer2.nabla_w00"], nabla_w[1](0, 0), precision);
  EXPECT_NEAR(t["layer2.nabla_w01"], nabla_w[1](0, 1), precision);
  EXPECT_NEAR(t["layer2.nabla_w02"], nabla_w[1](0, 2), precision);

  // First layer biases
  ASSERT_EQ(3, nabla_b[0].rows());
  ASSERT_EQ(1, nabla_b[0].cols());
  precision = pow(10, -t["layer1.nabla_b.precision"]);
  EXPECT_NEAR(t["layer1.nabla_b0"], nabla_b[0](0, 0), precision);
  EXPECT_NEAR(t["layer1.nabla_b1"], nabla_b[0](1, 0), precision);
  EXPECT_NEAR(t["layer1.nabla_b2"], nabla_b[0](2, 0), precision);

  // Second layer biases
  ASSERT_EQ(1, nabla_b[1].rows());
  ASSERT_EQ(1, nabla_b[1].cols());
  precision = pow(10, -t["layer2.nabla_b.precision"]);
  EXPECT_NEAR(t["layer2.nabla_b0"], nabla_b[1](0, 0), precision);
}

TEST_F(NetworkTest, UpdateMiniBatch) {
  labelled_eigen_data minibatch;
  VectorXd x0(2);
  x0 << t["minibatch.x00"], t["minibatch.x01"];
  VectorXd y0(1);
  y0 << t["minibatch.y0"];
  minibatch.push_back(make_pair(x0, y0));
  VectorXd x1(2);
  x1 << t["minibatch.x10"], t["minibatch.x11"];
  VectorXd y1(1);
  y1 << t["minibatch.y1"];
  minibatch.push_back(make_pair(x1, y1));
  VectorXd x2(2);
  x2 << t["minibatch.x20"], t["minibatch.x21"];
  VectorXd y2(1);
  y2 << t["minibatch.y2"];
  minibatch.push_back(make_pair(x2, y2));
  double eta = t["minibatch.eta"];
  double precision = pow(10, -t["minibatch.precision"]);
  nn->UpdateMiniBatch_(minibatch, eta);

  // First layer weights
  EXPECT_NEAR(t["minibatch.layer1.w00"], nn->weights[0](0, 0), precision);
  EXPECT_NEAR(t["minibatch.layer1.w01"], nn->weights[0](0, 1), precision);
  EXPECT_NEAR(t["minibatch.layer1.w10"], nn->weights[0](1, 0), precision);
  EXPECT_NEAR(t["minibatch.layer1.w11"], nn->weights[0](1, 1), precision);
  EXPECT_NEAR(t["minibatch.layer1.w20"], nn->weights[0](2, 0), precision);
  EXPECT_NEAR(t["minibatch.layer1.w21"], nn->weights[0](2, 1), precision);

  // Second layer weights
  EXPECT_NEAR(t["minibatch.layer2.w00"], nn->weights[1](0, 0), precision);
  EXPECT_NEAR(t["minibatch.layer2.w01"], nn->weights[1](0, 1), precision);
  EXPECT_NEAR(t["minibatch.layer2.w02"], nn->weights[1](0, 2), precision);

  // First layer biases
  EXPECT_NEAR(t["minibatch.layer1.b0"], nn->biases[0](0), precision);
  EXPECT_NEAR(t["minibatch.layer1.b1"], nn->biases[0](1), precision);
  EXPECT_NEAR(t["minibatch.layer1.b2"], nn->biases[0](2), precision);

  // Second layer biases
  EXPECT_NEAR(t["minibatch.layer2.b0"], nn->biases[1](0), precision);
}

TEST_F(NetworkTest, SGD) {
  // Generate training data
  labelled_data training_data;
  for(int i = 0; i < int(t["sgd.input_range_x"]); i++) {
    for(int j = 0; j < int(t["sgd.input_range_y"]); j++) {
      vector<double> x { (double) i, (double) j };
      vector<double> y { (double) (i + j) };
      training_data.push_back(make_pair(x, y));
    }
  }

  // Run training
  nn->SGD(training_data,
          int(t["sgd.epochs"]),
          int(t["sgd.minibatch_size"]),
          t["sgd.eta"]);

  double precision = pow(10, -t["sgd.precision"]);

  // First layer weights
  EXPECT_NEAR(t["sgd.layer1.w00"], nn->weights[0](0, 0), precision);
  EXPECT_NEAR(t["sgd.layer1.w01"], nn->weights[0](0, 1), precision);
  EXPECT_NEAR(t["sgd.layer1.w10"], nn->weights[0](1, 0), precision);
  EXPECT_NEAR(t["sgd.layer1.w11"], nn->weights[0](1, 1), precision);
  EXPECT_NEAR(t["sgd.layer1.w20"], nn->weights[0](2, 0), precision);
  EXPECT_NEAR(t["sgd.layer1.w21"], nn->weights[0](2, 1), precision);

  // Second layer weights
  EXPECT_NEAR(t["sgd.layer2.w00"], nn->weights[1](0, 0), precision);
  EXPECT_NEAR(t["sgd.layer2.w01"], nn->weights[1](0, 1), precision);
  EXPECT_NEAR(t["sgd.layer2.w02"], nn->weights[1](0, 2), precision);

  // First layer biases
  EXPECT_NEAR(t["sgd.layer1.b0"], nn->biases[0](0), precision);
  EXPECT_NEAR(t["sgd.layer1.b1"], nn->biases[0](1), precision);
  EXPECT_NEAR(t["sgd.layer1.b2"], nn->biases[0](2), precision);

  // Second layer biases
  EXPECT_NEAR(t["sgd.layer2.b0"], nn->biases[1](0), precision);
}

TEST_F(NetworkTest, GetMiniBatch) {
  labelled_eigen_data data;
  for(int i = 0; i < 20; i++) {
    VectorXd x(1), y(1);
    x << i;
    y << i;
    data.push_back(make_pair(x, y));
  }

  for(int n = 0; n < 5; n++) {
    labelled_eigen_data mini_batch = nn->GetMiniBatch_(data, 5, n);
    if(n < 4) {
      ASSERT_EQ(5, mini_batch.size());
      EXPECT_EQ(5 * n,     mini_batch[0].first(0));
      EXPECT_EQ(5 * n + 1, mini_batch[1].first(0));
      EXPECT_EQ(5 * n + 2, mini_batch[2].first(0));
      EXPECT_EQ(5 * n + 3, mini_batch[3].first(0));
      EXPECT_EQ(5 * n + 4, mini_batch[4].first(0));
    } else {
      ASSERT_EQ(0, mini_batch.size());
    }
  }
}

TEST_F(NetworkTest, MockShuffle) {
  labelled_eigen_data data;
  for(int i = 0; i < 3; i++) {
    VectorXd x(1), y(1);
    x << i;
    y << i;
    data.push_back(make_pair(x, y));
  }
  EXPECT_EQ(0, data[0].first(0));
  EXPECT_EQ(1, data[1].first(0));
  EXPECT_EQ(2, data[2].first(0));

  nn->Shuffle_(data);
  EXPECT_EQ(1, data[0].first(0));
  EXPECT_EQ(2, data[1].first(0));
  EXPECT_EQ(0, data[2].first(0));
}
