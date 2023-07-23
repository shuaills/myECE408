#include "gpu_conv.h"
#include <math.h>
#include <iostream>
#include "gpu-new-forward.h"

void gpu_Conv::init() {
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
  //std::cout << weight.colwise().sum() << std::endl;
  //std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

void gpu_Conv::forward(const Matrix& bottom) {

  // 1. 计算输出大小
  height_out = (height_in - height_kernel + 2 * pad_h) / stride + 1; 
  width_out = (width_in - width_kernel + 2 * pad_w) / stride + 1;

  std::cout << "Output height: " << height_out << ", Output width: " << width_out << std::endl;

  // 2. 为设备端分配内存  
  int B = bottom.cols(); // batch size
  int H_out = height_out;
  int W_out = width_out;
  int M = channel_out; 
  int C = channel_in;
  int H = height_in;
  int W = width_in;
  int K = height_kernel;

  std::cout << "Batch size: " << B << ", Number of output channels: " << M << std::endl;

  float *device_y;
  float *device_x; 
  float *device_k;

  GPUInterface gpu_interface; 

  // 3. 拷贝数据到设备  
  std::cout << "Copying data to device..." << std::endl;
  gpu_interface.conv_forward_gpu_prolog(NULL, bottom.data(), weight.data(), &device_y, &device_x, &device_k, 
                                        B, M, C, H, W, K);
  std::cout << "Data copied to device." << std::endl;
  
  // 4. 在设备端计算
  std::cout << "Performing computation on device..." << std::endl;
  gpu_interface.conv_forward_gpu(device_y, device_x, device_k, B, M, C, H, W, K);
  std::cout << "Computation on device done." << std::endl;

  // 5. 将结果拷贝回host
  std::cout << "Copying result back to host..." << std::endl;
  top.resize(H_out * W_out * M, B);
  gpu_interface.conv_forward_gpu_epilog(top.data(), device_y, device_x, device_k, B, M, C, H, W, K);
  std::cout << "Result copied back to host." << std::endl;

}


void gpu_Conv::im2col(const Vector& image, Matrix& data_col) {
  return ;
}

// col2im, used for grad_bottom
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void gpu_Conv::col2im(const Matrix& data_col, Vector& image) {
  return ;
}

void gpu_Conv::backward(const Matrix& bottom, const Matrix& grad_top) {
  return ;
}

void gpu_Conv::update(Optimizer& opt) {
  return ;
}

std::vector<float> gpu_Conv::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void gpu_Conv::set_parameters(const std::vector<float>& param) {
  if(static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> gpu_Conv::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
