// inference.cpp
#include "src/layer.h"
#include "src/layer/gpu_conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/mnist.h"
#include "src/network.h"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

int main() {
  std::cout<<"Loading fashion-mnist data...";
  MNIST dataset("../data/fmnist/");
  dataset.read_test_data();
  std::cout<<"Done"<<std::endl;

  std::cout<<"Loading model...";
  Network dnn;

  std::cout<<"Creating layers...";
  Layer* conv1 = new gpu_Conv(1, 28, 28, 4, 5, 5, 1, 0, 0); // Output size: 24x24
  Layer* pool1 = new MaxPooling(4, 24, 24, 2, 2, 2); // Output size: 12x12
  Layer* conv2 = new gpu_Conv(4, 12, 12, 16, 5, 5, 1, 0, 0); // Output size: 8x8
  Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2); // Output size: 4x4
  Layer* fc3 = new FullyConnected(pool2->output_dim(), 32); // Input dimension: 16*16*16 = 256
  Layer* fc4 = new FullyConnected(32, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* softmax = new Softmax;
  std::cout<<"Done"<<std::endl;

  std::cout<<"Adding layers to network...";
  dnn.add_layer(conv1);
  dnn.add_layer(relu1);
  dnn.add_layer(pool1);
  dnn.add_layer(conv2);
  dnn.add_layer(relu2);
  dnn.add_layer(pool2);
  dnn.add_layer(fc3);
  dnn.add_layer(relu3);
  dnn.add_layer(fc4);
  dnn.add_layer(softmax);
  std::cout<<"Done"<<std::endl;

  std::cout<<"Loading weights...";
  // load pre-trained weights
  dnn.load("model.dat");
  std::cout<<"Done"<<std::endl;

// 打印出数据的行数和列数
  std::cout << "dataset.test_data rows: " << dataset.test_data.rows() << std::endl;
  std::cout << "dataset.test_data cols: " << dataset.test_data.cols() << std::endl;

  std::cout<<"Performing forward pass...";
  dnn.forward(dataset.test_data);
  std::cout<<"Done"<<std::endl;

  std::cout<<"Computing accuracy...";
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout<<std::endl;
  std::cout<<"Test Accuracy: "<<acc<< std::endl;
  std::cout<<std::endl;
  return 0;
}
