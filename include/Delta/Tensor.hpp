#pragma once
#include "Utils.hpp"

namespace Delta{
  // Forward declaration
  template <typename Container>
  class Operator;

  template <typename DataType>
  class AbstractTensor{
    public:
      AbstractTensor() = default;
      virtual ~AbstractTensor() = default;
      virtual int Allocate(size_t size) = 0;
      virtual DataType* GetData() = 0;
      virtual int GetSize() = 0;
      template <typename... U>
      virtual Shape<U...> GetShape() = 0; 
  }; // class definition AbstractTensor

  template <typename DataType, typename TensorImpl>
  class Tensor: public AbstractTensor{
    private:
      /// @brief 
      Device m_device;
      size_t m_size;
      Operator<Tensor> m_parent_op;
      std::unique_ptr<DataType> m_data;

    public:
      int Allocate(size_t size){
        return static_cast<TensorImpl*>(this)->Allocate();
      }
      DataType* GetData(){
        return static_cast<TensorImpl*>(this)->GetData();
      }
      int GetSize(){
        return static_cast<TensorImpl*>(this)->GetSize();
      }
      template <typename... U>
      Shape<U...> GetShape(){
        return static_cast<TensorImpl*>(this)->GetShape();
      }
  };  // class definition Tensor
} // namespace Delta