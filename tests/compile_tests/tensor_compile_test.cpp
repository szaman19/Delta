#include "Delta/Tensor.hpp"
#include "CPU/Tensor_CPU_impl.hpp"

// The type of tensor included by the compiler decides the implementation 
// Maybe think about having the ability to switch impl would be nice.

using Tensor_Impl = Delta::Tensor_CPU_Impl<float>;
using Tensor = Delta::Tensor<float, Tensor_Impl>;

int main(int argc, char const *argv[]){
  {
    auto tensor_1 = Tensor(std::vector<int>({2, 2, 4, 4}), false); // Create a tensor of zeros, with gradient turned off

    std::cout << "The shape of the tensor is: "<< tensor_1.GetShape() << std::endl;
    // std::cout << "The size of the tensor is: "<< tensor_1.GetSize() << std::endl;
    // Testing out the to-string capabilities
    std::cout << "Printing out the elements: \n" << tensor_1 << std::endl;  

  }
  
  {
    auto tensor_1_with_grad = Tensor({2,2,2}, true);
    std::cout << tensor_1_with_grad<<std::endl;
  }
  
  
  {
    auto tensor_ones = Tensor(2.0, {4,4}, false);
    std::cout <<"The shape of the tensor is: "<< tensor_ones.GetShape() << std::endl;
    std::cout << tensor_ones<<std::endl;
  }
  {
    auto tensor_two = Tensor(2.0, {4}, false);
    std::cout <<"The shape of the tensor is: "<< tensor_two.GetShape() << std::endl;
  }

  return 0;
}
