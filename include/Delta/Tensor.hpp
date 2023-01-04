#pragma once
#include "Utils.hpp"
#include <numeric>

namespace Delta{

  // forward declaration
  template <typename Container>
  class Operator;

  /// @brief The base data container 
  /// This class implements an abstract data container that are used by the Tensor
  /// and Operator class. 
  /// @tparam DataType 
  template <typename DataType>
  class AbstractTensor{

    protected:
      // TO DO: This is problematic for pointers with non-default destructors
      // like CUDA. Maybe worth it to write a wrapper around 
      // unique_ptr or add a template parameter for custom 
      // desctructors. 

      std::unique_ptr<DataType> m_data;  
      std::unique_ptr<DataType> m_grad;
      size_t m_size;
      Shape<> m_shape;
      Device m_device;

    public:
      AbstractTensor() = default;

      AbstractTensor(const std::initializer_list<int>& shape):m_shape(shape){
        m_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      }

      AbstractTensor(const std::initializer_list<int>& shape,const Device& device):
        AbstractTensor(shape), m_device(device){}

      // Delete copy constructor for now. Too much of a headache.

      AbstractTensor(const AbstractTensor& tensor) = delete; 

      AbstractTensor(AbstractTensor&& a):
        m_data(std::move(a.m_data)),
        m_grad(std::move(a.grad)),
        m_shape(std::move(a.m_shape)),
        m_device(std::move(a.m_device)){}
      virtual ~AbstractTensor() = default;
      // virtual int Allocate(size_t size) = 0;

      // const DataType* GetData(){ return m_data.get();}
      // const DataType* GetGrad(){return m_grad.get();}
      const int GetSize() {return m_size;};
      const Shape<> GetShape() const {return m_shape;}

      DataType* GetData(){return m_data.get();}
      DataType* GetGrad(){return m_grad.get();}

      friend std::ostream& operator<<(std::ostream& os,const AbstractTensor& tensor){        
        os << tensor.GetShape();
        return os;
      }
  }; // class definition AbstractTensor

  /// @brief Tensor container with gradient tracking information  
  /// @tparam DataType 
  /// @tparam TensorImpl 
  template <typename DataType, typename TensorImpl>
  class Tensor: public AbstractTensor<DataType>{
    private:
      /// @brief 
      std::vector<Tensor<DataType, TensorImpl>*> m_parent_tensors;
      Operator<AbstractTensor<DataType>>*  m_parent_op;
      bool m_is_root;
      bool m_track_grad;

    private:
      void calculate_grad(){
        static_cast<TensorImpl*>(this)->calculate_grad();
      }
    
    public:
      /// @brief 
      /// @param shape 
      /// @param track_grad 
      Tensor(const std::initializer_list<int>& shape, const bool& track_grad):
        AbstractTensor<DataType>(shape),m_track_grad(track_grad){
          const auto size = this->m_size;
          auto allocate_val = Allocate(size);
          if (!allocate_val){
            std::cout << "Memory allocation failed. \n";
          }

          if (m_track_grad){
            auto allocate_grad_val = Allocate_Grad(size);
            if (!allocate_grad_val){
              std::cout << "Memory allocation for gradients failed. \n";
            }
          }
        }

      /// @brief 
      /// @param size 
      /// @return 
      int Allocate(const size_t& size){
        std::cout << "Calling allocate" <<std::endl;
        return static_cast<TensorImpl*>(this)->allocate_memory_impl(size);
      }

      int Allocate_Grad(const size_t& size){
        return static_cast<TensorImpl*>(this)->allocate_grad_memory_impl(size);
      }

      /// @brief 
      void Backward(){
        std::cout << "Callling backward \n";
        if (m_track_grad){
          m_parent_op->backward();

          if (!m_is_root){
            for (auto i =0; i<m_parent_tensors.size(); ++i){
              m_parent_tensors[i]->Backward(this);
            }
          }
        }
      }

      void Backward(const Tensor* upstream_tensor){
        std::cout << "Callling backward of non-leaf node\n";
        m_parent_op->backward(static_cast<const AbstractTensor<DataType>*>(upstream_tensor));
      }
  };  // class definition Tensor
} // namespace Delta