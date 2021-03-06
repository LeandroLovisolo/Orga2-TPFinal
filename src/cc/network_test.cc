#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "ini.h"
#include "gtest/gtest.h"

#include "matrix.h"
#include "network.h"

using namespace std;

template<class Matrix, class Vector=Matrix>
class MockNetwork : public Network<Matrix, Vector> {
 public:
  MockNetwork(const std::vector<int> &sizes) : Network<Matrix, Vector>(sizes) {}

  // Deterministic Shuffle_() implementation
  void Shuffle_(TrainingData<Vector>& data) {
    if(data.size() == 0) return;
    auto first = data[0];
    data.erase(data.begin());
    data.push_back(first);
  }
};

template<class Matrix>
class NetworkTest : public ::testing::Test {
  using Vector = Matrix;

 public:
  NetworkTest() {
    LoadTestData();
  }

  virtual void SetUp() {
    nn = make_unique<MockNetwork<Matrix, Vector>>(vector<int> { 2, 3, 1 });
    nn->weights[0].Set({ t["layer1.w00"], t["layer1.w01"],
                         t["layer1.w10"], t["layer1.w11"],
                         t["layer1.w20"], t["layer1.w21"] });
    nn->weights[1].Set({ t["layer2.w00"], t["layer2.w01"], t["layer2.w02"] });
    nn->biases[0].Set({ t["layer1.b0"], t["layer1.b1"], t["layer1.b2"] });
    nn->biases[1].Set({ t["layer2.b0"] });
  }

  void LoadTestData() {
    ifstream fs("../testdata");
    INI::Parser p(fs);
    for(auto const& kv : p.top()("default").values) {
      t[kv.first] = stod(kv.second);
      ts[kv.first] = kv.second;
    }
  }

  Matrix CreateMatrix(uint rows, uint cols) {
    return Matrix(rows, cols);
  }

  Matrix CreateVector(uint size) {
    return Matrix(size, 1);
  }

  TrainingData<Vector> CreateTrainingData() {
    return TrainingData<Vector>();
  }

  unique_ptr<MockNetwork<Matrix, Vector>> nn;
  unordered_map<string, float> t;
  unordered_map<string, string> ts;
};

typedef ::testing::Types<NaiveMatrix, SimdMatrix, EigenMatrix> MatrixTypes;

TYPED_TEST_CASE(NetworkTest, MatrixTypes);

#define nn (*this->nn)
#define t this->t
#define ts this->ts
#define Matrix this->CreateMatrix
#define Vector this->CreateVector
#define TrainingData this->CreateTrainingData

TYPED_TEST(NetworkTest, NumLayers) {
  EXPECT_EQ(3, nn.num_layers);
}

TYPED_TEST(NetworkTest, Sizes) {
  EXPECT_EQ((vector<int> { 2, 3, 1 }), nn.sizes);
}

TYPED_TEST(NetworkTest, Biases) {
  ASSERT_EQ(2, nn.biases.size());
  EXPECT_EQ(3, nn.biases[0].rows());
  EXPECT_EQ(1, nn.biases[0].cols());
  EXPECT_EQ(1, nn.biases[1].rows());
  EXPECT_EQ(1, nn.biases[1].cols());
}

TYPED_TEST(NetworkTest, Weights) {
  ASSERT_EQ(2, nn.weights.size());
  EXPECT_EQ(3, nn.weights[0].rows());
  EXPECT_EQ(2, nn.weights[0].cols());
  EXPECT_EQ(1, nn.weights[1].rows());
  EXPECT_EQ(3, nn.weights[1].cols());
}

TYPED_TEST(NetworkTest, FeedForward) {
  auto result = nn.FeedForward(Vector(2).Set({ t["feedforward.input1"],
                                               t["feedforward.input2"] }));
  ASSERT_EQ(1, result.size());
  float precision = pow(10, -t["feedforward.output.precision"]);
  EXPECT_NEAR(t["feedforward.output"], result(0), precision);
}

TYPED_TEST(NetworkTest, Sigmoid) {
  auto v = Vector(3).Set({ t["sigmoid.input1"],
                           t["sigmoid.input2"],
                           t["sigmoid.input3"] });
  auto res = nn.Sigmoid_(v);
  float precision1 = pow(10, -t["sigmoid.output1.precision"]);
  float precision2 = pow(10, -t["sigmoid.output2.precision"]);
  float precision3 = pow(10, -t["sigmoid.output3.precision"]);
  EXPECT_NEAR(t["sigmoid.output1"], res(0), precision1);
  EXPECT_NEAR(t["sigmoid.output2"], res(1), precision2);
  EXPECT_NEAR(t["sigmoid.output3"], res(2), precision3);
}

TYPED_TEST(NetworkTest, SigmoidPrime) {
  auto v = Vector(3).Set({ t["sigmoid_prime.input1"],
                           t["sigmoid_prime.input2"],
                           t["sigmoid_prime.input3"] });
  auto res = nn.SigmoidPrime_(v);
  float precision1 = pow(10, -t["sigmoid_prime.output1.precision"]);
  float precision2 = pow(10, -t["sigmoid_prime.output2.precision"]);
  float precision3 = pow(10, -t["sigmoid_prime.output3.precision"]);
  EXPECT_NEAR(t["sigmoid_prime.output1"], res(0), precision1);
  EXPECT_NEAR(t["sigmoid_prime.output2"], res(1), precision2);
  EXPECT_NEAR(t["sigmoid_prime.output3"], res(2), precision3);
}

TYPED_TEST(NetworkTest, Backpropagation) {
  auto input = Vector(2).Set({ 1, 2 });
  auto expected = Vector(1).Set({ 0 });
  auto gradients = nn.Backpropagation_(input, expected);
  auto nabla_w = gradients.first;
  auto nabla_b = gradients.second;

  // Make sure we've got the right number of gradient vectors
  ASSERT_EQ(2, nabla_w.size());
  ASSERT_EQ(2, nabla_b.size());

  // First layer weights
  ASSERT_EQ(3, nabla_w[0].rows());
  ASSERT_EQ(2, nabla_w[0].cols());
  float precision = pow(10, -t["layer1.nabla_w.precision"]);
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

TYPED_TEST(NetworkTest, UpdateMiniBatch) {
  auto minibatch = TrainingData();
  auto x0 = Vector(2).Set({ t["minibatch.x00"], t["minibatch.x01"] });
  auto y0 = Vector(1).Set({ t["minibatch.y0"] });
  minibatch.push_back(make_pair(x0, y0));
  auto x1 = Vector(2).Set({ t["minibatch.x10"], t["minibatch.x11"] });
  auto y1 = Vector(1).Set({ t["minibatch.y1"] });
  minibatch.push_back(make_pair(x1, y1));
  auto x2 = Vector(2).Set({ t["minibatch.x20"], t["minibatch.x21"] });
  auto y2 = Vector(1).Set({ t["minibatch.y2"] });
  minibatch.push_back(make_pair(x2, y2));

  float eta = t["minibatch.eta"];
  float precision = pow(10, -t["minibatch.precision"]);
  nn.UpdateMiniBatch_(minibatch, eta);

  // First layer weights
  EXPECT_NEAR(t["minibatch.layer1.w00"], nn.weights[0](0, 0), precision);
  EXPECT_NEAR(t["minibatch.layer1.w01"], nn.weights[0](0, 1), precision);
  EXPECT_NEAR(t["minibatch.layer1.w10"], nn.weights[0](1, 0), precision);
  EXPECT_NEAR(t["minibatch.layer1.w11"], nn.weights[0](1, 1), precision);
  EXPECT_NEAR(t["minibatch.layer1.w20"], nn.weights[0](2, 0), precision);
  EXPECT_NEAR(t["minibatch.layer1.w21"], nn.weights[0](2, 1), precision);

  // Second layer weights
  EXPECT_NEAR(t["minibatch.layer2.w00"], nn.weights[1](0, 0), precision);
  EXPECT_NEAR(t["minibatch.layer2.w01"], nn.weights[1](0, 1), precision);
  EXPECT_NEAR(t["minibatch.layer2.w02"], nn.weights[1](0, 2), precision);

  // First layer biases
  EXPECT_NEAR(t["minibatch.layer1.b0"], nn.biases[0](0), precision);
  EXPECT_NEAR(t["minibatch.layer1.b1"], nn.biases[0](1), precision);
  EXPECT_NEAR(t["minibatch.layer1.b2"], nn.biases[0](2), precision);

  // Second layer biases
  EXPECT_NEAR(t["minibatch.layer2.b0"], nn.biases[1](0), precision);
}

TYPED_TEST(NetworkTest, GetMiniBatch) {
  auto data = TrainingData();
  for(int i = 0; i < 20; i++) {
    auto x = Vector(1).Set({ (float) i });
    auto y = Vector(1).Set({ (float) i });
    data.push_back(make_pair(x, y));
  }

  for(int n = 0; n < 5; n++) {
    auto mini_batch = nn.GetMiniBatch_(data, 5, n);
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

TYPED_TEST(NetworkTest, MockShuffle) {
  auto data = TrainingData();
  for(int i = 0; i < 3; i++) {
    data.push_back(make_pair(Vector(1).Set({ (float) i }),
                             Vector(1).Set({ (float) i })));
  }
  EXPECT_EQ(0, data[0].first(0));
  EXPECT_EQ(1, data[1].first(0));
  EXPECT_EQ(2, data[2].first(0));

  nn.Shuffle_(data);
  EXPECT_EQ(1, data[0].first(0));
  EXPECT_EQ(2, data[1].first(0));
  EXPECT_EQ(0, data[2].first(0));
}

TYPED_TEST(NetworkTest, SGD) {
  // Generate training data
  auto training_data = TrainingData();
  for(int i = 0; i < int(t["sgd.input_range_x"]); i++) {
    for(int j = 0; j < int(t["sgd.input_range_y"]); j++) {
      training_data.push_back(
          make_pair(Vector(2).Set({ (float) i, (float) j }),
                    Vector(1).Set({ (float) i + j })));
    }
  }

  // Run training
  nn.SGD(training_data,
         int(t["sgd.epochs"]),
         int(t["sgd.minibatch_size"]),
         t["sgd.eta"]);

  float precision = pow(10, -t["sgd.precision"]);

  // First layer weights
  EXPECT_NEAR(t["sgd.layer1.w00"], nn.weights[0](0, 0), precision);
  EXPECT_NEAR(t["sgd.layer1.w01"], nn.weights[0](0, 1), precision);
  EXPECT_NEAR(t["sgd.layer1.w10"], nn.weights[0](1, 0), precision);
  EXPECT_NEAR(t["sgd.layer1.w11"], nn.weights[0](1, 1), precision);
  EXPECT_NEAR(t["sgd.layer1.w20"], nn.weights[0](2, 0), precision);
  EXPECT_NEAR(t["sgd.layer1.w21"], nn.weights[0](2, 1), precision);

  // Second layer weights
  EXPECT_NEAR(t["sgd.layer2.w00"], nn.weights[1](0, 0), precision);
  EXPECT_NEAR(t["sgd.layer2.w01"], nn.weights[1](0, 1), precision);
  EXPECT_NEAR(t["sgd.layer2.w02"], nn.weights[1](0, 2), precision);

  // First layer biases
  EXPECT_NEAR(t["sgd.layer1.b0"], nn.biases[0](0), precision);
  EXPECT_NEAR(t["sgd.layer1.b1"], nn.biases[0](1), precision);
  EXPECT_NEAR(t["sgd.layer1.b2"], nn.biases[0](2), precision);

  // Second layer biases
  EXPECT_NEAR(t["sgd.layer2.b0"], nn.biases[1](0), precision);
}

TYPED_TEST(NetworkTest, LoadCheckpoint) {
  string checkpoint =
    "3 3 2 2 "
    "1 2 3 4 5 6 "
    "-1 -2 "
    "7 8 9 10 "
    "-3 -4";
  nn.LoadCheckpoint_(checkpoint);

  EXPECT_EQ(nn.num_layers, 3);
  EXPECT_EQ(nn.sizes, (vector<int> { 3, 2, 2 }));
  EXPECT_EQ(nn.weights[0](0, 0), 1);
  EXPECT_EQ(nn.weights[0](0, 1), 2);
  EXPECT_EQ(nn.weights[0](0, 2), 3);
  EXPECT_EQ(nn.weights[0](1, 0), 4);
  EXPECT_EQ(nn.weights[0](1, 1), 5);
  EXPECT_EQ(nn.weights[0](1, 2), 6);
  EXPECT_EQ(nn.biases[0](0), -1);
  EXPECT_EQ(nn.biases[0](1), -2);
  EXPECT_EQ(nn.weights[1](0, 0), 7);
  EXPECT_EQ(nn.weights[1](0, 1), 8);
  EXPECT_EQ(nn.weights[1](1, 0), 9);
  EXPECT_EQ(nn.weights[1](1, 1), 10);
  EXPECT_EQ(nn.biases[1](0), -3);
  EXPECT_EQ(nn.biases[1](1), -4);
}

TYPED_TEST(NetworkTest, SaveCheckpoint) {
  vector<string> items {
    // num_layers, sizes[0], sizes[1], sizes[2]
    "3", "2", "3", "1",
    // Layer 1 weights
    ts["layer1.w00"], ts["layer1.w01"],
    ts["layer1.w10"], ts["layer1.w11"],
    ts["layer1.w20"], ts["layer1.w21"],
    // Layer 1 biases
    ts["layer1.b0"], ts["layer1.b1"], ts["layer1.b2"],
    // Layer 2 weights
    ts["layer2.w00"], ts["layer2.w01"], ts["layer2.w02"],
    // Layer 2 biases
    ts["layer2.b0"]
  };

  stringstream expected;
  bool first = true;
  for(const string& item : items) {
    if(first) first = false;
    else expected << " ";
    expected << item;
  }

  EXPECT_EQ(expected.str(), nn.SaveCheckpoint());
}
