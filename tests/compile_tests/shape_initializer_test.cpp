#include "Delta/Utils.hpp"


int main(int argc, char const *argv[]){

  // Brace-initialization

  auto scalar_shape = Delta::Shape{100}; // Shape <int>
  std::cout << "Scalar Shape: " << scalar_shape;
  
  auto mat_shape = Delta::Shape{10,20};  // Shape <int, int>
  std::cout << "Matrix Shape: " << mat_shape; 

  auto tensor_shape = Delta::Shape{1, 28, 28}; // Shape <int, int, int>
  std::cout << "Tensor Shape: " << tensor_shape;

  // Vector constructor

  std::vector<int> dynamic_dims; 
  dynamic_dims.push_back(10);
  dynamic_dims.push_back(20);
  auto mat_shape_2 = Delta::Shape(dynamic_dims);  // Shape <>

  std::cout << "Matrix Shape (vector constructor): " << mat_shape_2;
  
  std::cout << (mat_shape == mat_shape_2) << std::endl;  // 1
  std::cout << (scalar_shape == mat_shape_2) << std::endl;  // 0

  scalar_shape = mat_shape;
  std::cout << "Copy Assigned Matrix Shape: " << scalar_shape;

  auto copied_mat_shape {mat_shape};
  std::cout << "Copy Constructed Matrix Shape:" << copied_mat_shape;

  return 0;
}
