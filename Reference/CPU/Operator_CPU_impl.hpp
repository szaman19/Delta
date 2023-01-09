#pragma once
#include "Delta/Tensor.hpp"
#include "Delta/Operator.hpp"
#include "CPU/Tensor_CPU_impl.hpp"


namespace Delta{

  template <typename T>
  using TensorImplType = Tensor_CPU_Impl<T>;

  template <typename T>
  using TensorType = Tensor<T, TensorImplType<T>>; 

  // TODO: Figure out if it makes sense to set up Operator with AbstractTensor
  // vs TensorType
  template <typename DataType>
  class Sum_CPU_impl final: public Operator<AbstractTensor<DataType>>{
    public:
      using Container = AbstractTensor<DataType>;
      Container* m_parent_0 {nullptr};
      Container* m_parent_1 {nullptr};

      std::string GetName() override{
        return "Sum Operator";
      }
      TensorType<DataType>& forward(Container& input_0,
                                    Container& input_1){

        m_parent_0 = &input_0;
        m_parent_1 = &input_1;

        if (input_0.GetShape() != input_1.GetShape()){
          // TODO: Replace with error message and assert
          std::cout << "Possible error due to mismatched shape\n";
        }

        // TODO: Replace this when fixing the Shape struct to 
        // stop using the variadic templates

        auto shape = input_0.GetShape().m_dims;
        auto num_elem = input_0.GetSize();
        auto inp_data_0 = input_0.GetData();
        auto inp_data_1 = input_1.GetData();

        auto _result = new DataType[num_elem];
        for(auto i = 0; i < num_elem; ++i){
          _result[i] = inp_data_0[i] + inp_data_1[i];
        }

        
        TensorType<DataType>* resultant = new TensorType<DataType>(_result,                 // m_data
                                                                   shape,                    // m_shape
                                                                   false,                    // m_is_root
                                                                   true,                     // m_track_grad
                                                                   this);                    // m_parent_op      
        
        resultant->Add_Parent(static_cast<TensorType<DataType>*>(&input_0));
        resultant->Add_Parent(static_cast<TensorType<DataType>*>(&input_1));
        return *resultant;
      }

      void backward(Container* upstream_gradient) override{
        auto* parent_1_grad = m_parent_0->GetGrad();
        auto* parent_2_grad = m_parent_1->GetGrad();

        auto _grad_size = m_parent_0->GetSize();

        const auto* dl = upstream_gradient->GetGrad(); 

        for (auto i =0; i < _grad_size; ++i){
          parent_1_grad[i] += dl[i];
          parent_2_grad[i] += dl[i];
        }
      }

      void backward() override{
        std::cout << "Calling sum backward impl root \n";
        auto* parent_1_grad = m_parent_0->GetGrad();
        auto* parent_2_grad = m_parent_1->GetGrad();

        auto _grad_size = m_parent_1->GetSize();

        for (auto i =0; i < _grad_size; ++i){
          parent_1_grad[i] += DataType(1);
          parent_2_grad[i] += DataType(1);
        }
      }
  };

template <typename DataType>
  class Mul_CPU_impl final: public Operator<AbstractTensor<DataType>>{
    public:
      using Container = AbstractTensor<DataType>;
      Container* m_parent_1 {nullptr};
      Container* m_parent_2 {nullptr};
      
      std::string GetName() override{
        return "Mul Operator";
      }
      TensorType<DataType>& forward(Container& input_0,
                                    Container& input_1){

        m_parent_1 = &input_0;
        m_parent_2 = &input_1;

        if (input_0.GetShape() != input_1.GetShape()){
          // TODO: Replace with error message and assert
          std::cout << "Possible error due to mismatched shape\n";
        }

        auto shape = input_0.GetShape().m_dims;
        auto num_elem = input_0.GetSize();
        auto inp_data_0 = input_0.GetData();
        auto inp_data_1 = input_1.GetData();

        auto _result = new DataType[num_elem];
        for(auto i = 0; i < num_elem; ++i){
          _result[i] = inp_data_0[i] * inp_data_1[i];
        }

        TensorType<DataType>* resultant = new TensorType<DataType>(_result,                 // m_data
                                                                   shape,                    // m_shape
                                                                   false,                    // m_is_root
                                                                   true,                     // m_track_grad
                                                                   this);                    // m_parent_op      
        
        resultant->Add_Parent(static_cast<TensorType<DataType>*>(&input_0));
        resultant->Add_Parent(static_cast<TensorType<DataType>*>(&input_1));

        return *resultant;
      }

      void backward() override{
        backward(1.0);
      }

      void backward(Container* upstream_gradient) override{
        auto parent_1_data = m_parent_1->GetData();
        auto parent_2_data = m_parent_2->GetData();

        auto* parent_1_grad = m_parent_1->GetGrad();
        auto* parent_2_grad = m_parent_2->GetGrad();

        auto _grad_size = m_parent_1->GetSize();

        const auto* dl = upstream_gradient->GetGrad();

        for (auto i =0; i < _grad_size; ++i){
          parent_1_grad[i] += dl[i]*parent_2_data[i];
          parent_2_grad[i] += dl[i]*parent_1_data[i];
        }
      }

      void backward(const DataType& dl = 1) {
          auto parent_1_data = m_parent_1->GetData();
          auto parent_2_data = m_parent_2->GetData();

          auto* parent_1_grad = m_parent_1->GetGrad();
          auto* parent_2_grad = m_parent_2->GetGrad();

          auto _grad_size = m_parent_1->GetSize();

          for (auto i =0; i < _grad_size; ++i){
            parent_1_grad[i] += dl*parent_2_data[i];
            parent_2_grad[i] += dl*parent_1_data[i];
          }
      }

  };

  namespace Ops{
    template<class DataType>
    using Container = AbstractTensor<DataType>;

    template <typename DataType>
    TensorType<DataType>& Sum(Container<DataType>& input_0,
                              Container<DataType>& input_1){
      auto _op = new Sum_CPU_impl<DataType>();
      auto& res = _op->forward(input_0, input_1);
      return res;
    }

    template <typename DataType>
    TensorType<DataType>& Mul(Container<DataType>& input_0,
                              Container<DataType>& input_1){
      auto _op = new Mul_CPU_impl<DataType>();
      auto& res = _op->forward(input_0, input_1);
      return res;
    }
  }
} // Delta namespace