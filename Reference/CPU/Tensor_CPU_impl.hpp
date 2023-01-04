#pragma once
#include "Delta/Tensor.hpp"

namespace Delta{
  // Forward declaration (?)
  // Note: This seems unnecessary 
  template <typename DataType, typename TensorImpl>
  class Tensor; 

  template <typename DataType>
  class Tensor_CPU_Impl final: public Tensor<DataType, Tensor_CPU_Impl<DataType>>{
    public:
      int allocate_memory_impl(const size_t& size){
        std::cout << "Allocating memory of size " << size << " \n";
        this->m_data = std::make_unique<DataType>(size);
        return 1;
      }

      int allocate_grad_memory_impl(const size_t& size){
        std::cout << "Allocating gradient memory of size " << size << '\n';
        this->m_grad  = std::make_unique<DataType>(size);
        return 1;
      }
  };
}
  