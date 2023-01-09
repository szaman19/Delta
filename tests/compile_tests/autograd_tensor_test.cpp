#include "Delta/Tensor.hpp"
#include "CPU/Tensor_CPU_impl.hpp"
#include "CPU/Operator_CPU_impl.hpp"

// The type of tensor included by the compiler decides the implementation 
// Maybe think about having the ability to switch impl would be nice.
using Tensor_Impl = Delta::Tensor_CPU_Impl<float>;
using Tensor = Delta::Tensor<float, Tensor_Impl>;
using Sum_Op = Delta::Sum_CPU_impl<float>;

int main(int argc, char const *argv[]){

  auto summand_1 = Tensor(1, {1}, true);
  auto summand_2 = Tensor(2, {1}, true);

  std::cout << summand_1;
  std::cout << summand_2;

  auto& res = Delta::Ops::Sum(summand_1, summand_2); // 2 + 1

  std::cout << "Check: " << res.GetData()[0] <<" == " << 3 << std::endl;

  auto multiplier = Tensor(5, {1}, true);
  
  auto& out = Delta::Ops::Mul(res, multiplier);

  std::cout << "Check: " << out.GetData()[0] << " == " << 15 << std::endl;
  out.Backward();

  std::cout << "Check: " << out.GetGrad()[0] << " == " << 1 << '\n';
  std::cout << "Check: " << res.GetGrad()[0] << " == " << 5 << '\n';
  std::cout << "Check: " << multiplier.GetGrad()[0] << " == " << 3 << '\n';
  std::cout << "Check: " << summand_1.GetGrad()[0] << " == " << 5 << '\n';
  std::cout << "Check: " << summand_2.GetGrad()[0] << " == " << 5 << std::endl;

  return 0;
}
