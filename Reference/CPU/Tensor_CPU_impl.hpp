#pragma once
#include "Delta/Tensor.hpp"
#include "Delta/Utils.hpp"

namespace Delta{
  // Forward declaration (?)
  // Note: This seems unnecessary 
  template <typename DataType, typename TensorImpl>
  class Tensor; 

  template <typename DataType>
  class Tensor_CPU_Impl final: public Tensor<DataType, Tensor_CPU_Impl<DataType>>{
    public:
      int allocate_memory_impl(const size_t& size){
        this->m_data = new DataType[size];
        return 1;
      }

      int allocate_grad_memory_impl(const size_t& size){
        this->m_grad = new DataType[size];;
        return 1;
      }
      int fill_impl(const DataType& fill_val, const int& size){
        for(auto i = 0; i < size; i++){
          this->m_data[i] = fill_val;
        }
        return 1;
      }

      int fill_grad_impl(const DataType& fill_val, const int& size){
        for(auto i = 0; i < size; i++){
          this->m_grad[i] = fill_val;
        }
        return 1;
      }

      void clean_up_impl(){
        if (this->m_data){  
          delete this->m_data;  
        }
        if (this->m_grad){
          delete this->m_grad;
        }
      }

    std::ostream& print_impl(std::ostream& os);
  };

  template <typename DataType>
  std::ostream&
  Tensor_CPU_Impl<DataType>::
  print_impl(std::ostream& os){
    return tensor_to_str(os, this->GetData(), this->GetShape().m_dims);
  }
}
  