#pragma once
#include "Delta/Tensor.hpp"
#include "Delta/Operator.hpp"
#include "CPU/Tensor_CPU_impl.hpp"


namespace Delta{
  // Forward declaration (?)

  template <typename T>
  using Tensor_Impl_Type = Tensor_CPU_Impl<T>;

  template <typename T>
  using Tensor_Type = Tensor<T, Tensor_Impl_Type<T>>; 

  template <typename DataType>
  class Operator_CPU_impl final: public Operator<AbstractTensor<DataType>>{
  private:
    /* data */
  public:
    Operator_CPU_impl(/* args */);
    ~Operator_CPU_impl();
  };
  template <typename DataType>
  Operator_CPU_impl<DataType>::Operator_CPU_impl(/* args */){
  
  }
  
  template <typename DataType>
  Operator_CPU_impl<DataType>::~Operator_CPU_impl(){
  
  }

  template <typename DataType>
  class Sum_CPU_impl final: public Operator<AbstractTensor<DataType>>{
    public:
      using Container = AbstractTensor<DataType>;
      Container* m_parent_1 {nullptr};
      Container* m_parent_2 {nullptr};

      Tensor_Type<DataType>& forward(Container& summand_1,
                                     Container& summand_2){

        m_parent_1 = &summand_1;
        m_parent_2 = &summand_2;

        Tensor_Type<DataType>* resultant = new Tensor_Type<DataType>({1}, true);           
        return *resultant;
      }

      void backward_impl(const Container* upstream_gradient) override{

      }

      void backward_impl() override{
        auto parent_1_data = m_parent_1->GetData();
        auto parent_2_data = m_parent_2->GetData();

        auto* parent_1_grad = m_parent_1->GetGrad();
        auto* parent_2_grad = m_parent_2->GetGrad();

        auto _grad_size = m_parent_1->GetSize();

        for (auto i =0; i < _grad_size; ++i){
          parent_1_grad[i] += parent_2_data[i];
          parent_2_grad[i] += parent_1_data[i];
        }
      }
  };
} // Delta namespace