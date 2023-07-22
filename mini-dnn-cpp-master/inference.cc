#include <iostream>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf; 

// 假设已有Network和MNIST数据集
class Network {
  public:
    void forward(const Matrix& input);
    Matrix output() const; 
};

class MNIST {
  public:
    Matrix test_data; 
};

int main() {

  MNIST dataset;
  // 初始化测试数据集

  Network model;
  // 初始化模型

  model.forward(dataset.test_data);

  Matrix output = model.output();

  // 输出结果
  std::cout << "Model output: " << std::endl << output << std::endl;

  return 0;
}