#include "Delta/Tensor.hpp"
#include "CPU/Tensor_CPU_impl.hpp"
#include "CPU/Operator_CPU_impl.hpp"

// The type of tensor included by the compiler decides the implementation 
// Maybe think about having the ability to switch impl would be nice.
using Tensor_Impl = Delta::Tensor_CPU_Impl<float>;
using Tensor = Delta::Tensor<float, Tensor_Impl>;
using Sum_Op = Delta::Sum_CPU_impl<float>;

int main(int argc, char const *argv[]){
  auto tensor_1 = Tensor({2, 2, 2}, false); // Create a tensor of zeros, with gradient turned off

  std::cout << "The shape of the tensor is: "<<tensor_1.GetShape() << std::endl;
  std::cout << "The size of the tensor is: "<< tensor_1.GetSize() << std::endl;

  // Testing out the to-string capabilities
  std::cout << "Printing out the elements: \n" <<tensor_1 << std::endl;  

  auto summand_1 = Tensor({1}, true);
  auto summand_2 = Tensor({1}, true);
  auto sum_op = Sum_Op();

  auto& res = sum_op.forward(summand_1, summand_2);

  std::cout << "The shape of tensor is: " << res.GetShape() << std::endl;

  res.Backward();

  return 0;
}
