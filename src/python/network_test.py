import network

import numpy as np
import unittest
from ConfigParser import ConfigParser

class TestNetwork(unittest.TestCase):

  def setUp(self):
    self.nn = network.Network([2, 3, 1])
    self.nn.weights = [
        np.array([[T["layer1.w00"], T["layer1.w01"]],
                  [T["layer1.w10"], T["layer1.w11"]],
                  [T["layer1.w20"], T["layer1.w21"]]]),
        np.array([[T["layer2.w00"],
                   T["layer2.w01"],
                   T["layer2.w02"]]])]
    self.nn.biases = [
        np.array([[T["layer1.b0"]],
                  [T["layer1.b1"]],
                  [T["layer1.b2"]]]), np.array([[T["layer2.b0"]]])]

  def test_num_layers(self):
    self.assertEquals(3, self.nn.num_layers)

  def test_sizes(self):
    self.assertEquals([2, 3, 1], self.nn.sizes)

  def test_biases(self):
    self.assertEquals(2, len(self.nn.biases))
    self.assertTupleEqual((3, 1), self.nn.biases[0].shape)
    self.assertTupleEqual((1, 1), self.nn.biases[1].shape)

  def test_weights(self):
    self.assertEquals(2, len(self.nn.weights))
    self.assertTupleEqual((3, 2), self.nn.weights[0].shape)
    self.assertTupleEqual((1, 3), self.nn.weights[1].shape)

  def test_feedforward(self):
    result = self.nn.feedforward(np.array([[T["feedforward.input1"]],
                                           [T["feedforward.input2"]]]))
    self.assertEqual(1, len(result))
    self.assertAlmostEqual(T["feedforward.output"], result[0],
                           int(T["feedforward.output.precision"]))

  def test_sigmoid(self):
    self.assertAlmostEqual(T["sigmoid.output1"],
                           network.sigmoid(T["sigmoid.input1"]),
                           int(T["sigmoid.output1.precision"]))
    self.assertAlmostEqual(T["sigmoid.output2"],
                           network.sigmoid(T["sigmoid.input2"]),
                           int(T["sigmoid.output2.precision"]))
    self.assertAlmostEqual(T["sigmoid.output3"],
                           network.sigmoid(T["sigmoid.input3"]),
                           int(T["sigmoid.output3.precision"]))

  def test_sigmoid_prime(self):
    self.assertAlmostEqual(T["sigmoid_prime.output1"],
                           network.sigmoid_prime(T["sigmoid_prime.input1"]),
                           int(T["sigmoid_prime.output1.precision"]))
    self.assertAlmostEqual(T["sigmoid_prime.output2"],
                           network.sigmoid_prime(T["sigmoid_prime.input2"]),
                           int(T["sigmoid_prime.output2.precision"]))
    self.assertAlmostEqual(T["sigmoid_prime.output3"],
                           network.sigmoid_prime(T["sigmoid_prime.input3"]),
                           int(T["sigmoid_prime.output3.precision"]))

  def test_backpropagation(self):
    nabla_b, nabla_w = self.nn.backprop(np.array([[1], [2]]), np.array([0]))

    # Make sure we've got the right number of gradient vectors
    self.assertEquals(2, len(nabla_w))
    self.assertEquals(2, len(nabla_b))

    # First layer weights
    self.assertTupleEqual((3, 2), nabla_w[0].shape)
    self.assertAlmostEquals(T["layer1.nabla_w00"],
                            nabla_w[0][0][0],
                            int(T["layer1.nabla_w.precision"]))
    self.assertAlmostEquals(T["layer1.nabla_w01"],
                            nabla_w[0][0][1],
                            int(T["layer1.nabla_w.precision"]))
    self.assertAlmostEquals(T["layer1.nabla_w10"],
                            nabla_w[0][1][0],
                            int(T["layer1.nabla_w.precision"]))
    self.assertAlmostEquals(T["layer1.nabla_w11"],
                            nabla_w[0][1][1],
                            int(T["layer1.nabla_w.precision"]))
    self.assertAlmostEquals(T["layer1.nabla_w20"],
                            nabla_w[0][2][0],
                            int(T["layer1.nabla_w.precision"]))
    self.assertAlmostEquals(T["layer1.nabla_w21"],
                            nabla_w[0][2][1],
                            int(T["layer1.nabla_w.precision"]))

    # Second layer weights
    self.assertTupleEqual((1, 3), nabla_w[1].shape)
    self.assertAlmostEquals(T["layer2.nabla_w00"],
                            nabla_w[1][0][0],
                            int(T["layer2.nabla_w.precision"]))
    self.assertAlmostEquals(T["layer2.nabla_w01"],
                            nabla_w[1][0][1],
                            int(T["layer2.nabla_w.precision"]))
    self.assertAlmostEquals(T["layer2.nabla_w02"],
                            nabla_w[1][0][2],
                            int(T["layer2.nabla_w.precision"]))

    # First layer biases
    self.assertTupleEqual((3, 1), nabla_b[0].shape)
    self.assertAlmostEquals(T["layer1.nabla_b0"],
                            nabla_b[0][0][0],
                            int(T["layer1.nabla_b.precision"]))
    self.assertAlmostEquals(T["layer1.nabla_b1"],
                            nabla_b[0][1][0],
                            int(T["layer1.nabla_b.precision"]))
    self.assertAlmostEquals(T["layer1.nabla_b2"],
                            nabla_b[0][2][0],
                            int(T["layer1.nabla_b.precision"]))

    # Second layer biases
    self.assertTupleEqual((1, 1), nabla_b[1].shape)
    self.assertAlmostEquals(T["layer2.nabla_b0"],
                            nabla_b[1][0][0],
                            int(T["layer2.nabla_b.precision"]))

  def test_update_mini_batch(self):
    minibatch = [
      (np.array([[T["minibatch.x00"]], [T["minibatch.x01"]]]),
       T["minibatch.y0"]),
      (np.array([[T["minibatch.x10"]], [T["minibatch.x11"]]]),
       T["minibatch.y1"]),
      (np.array([[T["minibatch.x20"]], [T["minibatch.x21"]]]),
       T["minibatch.y2"]),
    ]
    eta = T["minibatch.eta"]
    precision = int(T["minibatch.precision"])
    self.nn.update_mini_batch(minibatch, eta)

    # First layer weights
    self.assertAlmostEqual(T["minibatch.layer1.w00"],
                           self.nn.weights[0][0][0], precision)
    self.assertAlmostEqual(T["minibatch.layer1.w01"],
                           self.nn.weights[0][0][1], precision)
    self.assertAlmostEqual(T["minibatch.layer1.w10"],
                           self.nn.weights[0][1][0], precision)
    self.assertAlmostEqual(T["minibatch.layer1.w11"],
                           self.nn.weights[0][1][1], precision)
    self.assertAlmostEqual(T["minibatch.layer1.w20"],
                           self.nn.weights[0][2][0], precision)
    self.assertAlmostEqual(T["minibatch.layer1.w21"],
                           self.nn.weights[0][2][1], precision)

    # Second layer weights
    self.assertAlmostEqual(T["minibatch.layer2.w00"],
                           self.nn.weights[1][0][0], precision)
    self.assertAlmostEqual(T["minibatch.layer2.w01"],
                           self.nn.weights[1][0][1], precision)
    self.assertAlmostEqual(T["minibatch.layer2.w02"],
                           self.nn.weights[1][0][2], precision)

    # First layer biases
    self.assertAlmostEqual(T["minibatch.layer1.b0"],
                           self.nn.biases[0][0], precision)
    self.assertAlmostEqual(T["minibatch.layer1.b1"],
                           self.nn.biases[0][1], precision)
    self.assertAlmostEqual(T["minibatch.layer1.b2"],
                           self.nn.biases[0][2], precision)

    # Second layer biases
    self.assertAlmostEqual(T["minibatch.layer2.b0"],
                           self.nn.biases[1][0], precision)

  def test_sgd(self):
    # Make test deterministic
    def shuffle(data):
      if not data: return
      data.append(data.pop(0))
    self.nn.shuffle = shuffle

    # Generate training data
    training_data = [(np.array([[i], [j]]), i + j)
                     for i in xrange(int(T["sgd.input_range_x"]))
                     for j in xrange(int(T["sgd.input_range_y"]))]

    # Run training
    self.nn.SGD(training_data,
                int(T["sgd.epochs"]),
                int(T["sgd.minibatch_size"]),
                T["sgd.eta"])

    precision = int(T["sgd.precision"])

    # First layer weights
    self.assertAlmostEqual(T["sgd.layer1.w00"],
                           self.nn.weights[0][0][0], precision)
    self.assertAlmostEqual(T["sgd.layer1.w01"],
                           self.nn.weights[0][0][1], precision)
    self.assertAlmostEqual(T["sgd.layer1.w10"],
                           self.nn.weights[0][1][0], precision)
    self.assertAlmostEqual(T["sgd.layer1.w11"],
                           self.nn.weights[0][1][1], precision)
    self.assertAlmostEqual(T["sgd.layer1.w20"],
                           self.nn.weights[0][2][0], precision)
    self.assertAlmostEqual(T["sgd.layer1.w21"],
                           self.nn.weights[0][2][1], precision)

    # Second layer weights
    self.assertAlmostEqual(T["sgd.layer2.w00"],
                           self.nn.weights[1][0][0], precision)
    self.assertAlmostEqual(T["sgd.layer2.w01"],
                           self.nn.weights[1][0][1], precision)
    self.assertAlmostEqual(T["sgd.layer2.w02"],
                           self.nn.weights[1][0][2], precision)

    # First layer biases
    self.assertAlmostEqual(T["sgd.layer1.b0"],
                           self.nn.biases[0][0], precision)
    self.assertAlmostEqual(T["sgd.layer1.b1"],
                           self.nn.biases[0][1], precision)
    self.assertAlmostEqual(T["sgd.layer1.b2"],
                           self.nn.biases[0][2], precision)

    # Second layer biases
    self.assertAlmostEqual(T["sgd.layer2.b0"],
                           self.nn.biases[1][0], precision)

# Dictionary that holds numerical test data
T = {}

# Reads test data and store it in dictionary T
def load_test_data():
  config = ConfigParser()
  config.read("../testdata")
  for key, value in config.items("default"):
    T[key] = float(value)

if __name__ == '__main__':
  load_test_data()
  unittest.main()
