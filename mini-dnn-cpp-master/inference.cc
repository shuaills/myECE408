// inference.cpp
#include "src/mnist.h"
#include "src/network.h"

#include <iostream>
#include <Eigen/Dense>

int main() {

  // 初始化MNIST数据集
  std::cout<<"Loading fashion-mnist data...";
  MNIST dataset("../data/mnist/");
  dataset.read_test_data();
  std::cout<<"Done"<<std::endl;

  // 初始化网络
  Network dnn;


  // 在测试集上运行推理
  dnn.forward(dataset.test_data);
  
  // 获取输出
  Matrix output = dnn.output();

  // 输出结果
  std::cout << "Model output:" << std::endl << output << std::endl;

  return 0;
}