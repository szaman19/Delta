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
      // desctructors. Or use a shared pointer. 

      DataType* m_data;  
      DataType* m_grad = nullptr;
      Shape<> m_shape;
      size_t m_size;
      Device m_device;

    public:
      AbstractTensor() = default;

      AbstractTensor(DataType* data,
                     const std::vector<int>& shape):
                      m_data(data),
                      m_shape(shape){
        m_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      }

      AbstractTensor(const std::vector<int>& shape,const Device& device):
        AbstractTensor(nullptr, shape), m_device(device){}

      // Delete copy constructor for now. Too much of a headache.

      AbstractTensor(const AbstractTensor& tensor) = delete; 

      AbstractTensor(AbstractTensor&& a):
        m_data(std::move(a.m_data)),
        m_grad(std::move(a.grad)),
        m_shape(std::move(a.m_shape)),
        m_device(std::move(a.m_device)){}
      // TODO: Maybe turn this pure virtual 
      virtual ~AbstractTensor() = default;
      // virtual int Allocate(size_t size) = 0;

      // const DataType* GetData(){ return m_data.get();}
      // const DataType* GetGrad(){return m_grad.get();}
      const int GetSize() {return m_size;};
      const Shape<> GetShape() const {return m_shape;}

      DataType* GetData(){return m_data;}
      DataType* GetGrad(){return m_grad;}

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
      
      bool m_is_root;
      bool m_track_grad;
      Operator<AbstractTensor<DataType>>*  m_parent_op;

    private:
      void calculate_grad(){
        static_cast<TensorImpl*>(this)->calculate_grad();
      }
    
    public:
      /// @brief 
      /// @param shape 
      /// @param track_grad 
      Tensor(DataType* data,
             const std::vector<int>& shape,
             const bool& is_root,
             const bool& track_grad,
             Operator<AbstractTensor<DataType>>*  parent_op):
              AbstractTensor<DataType>(data, shape),
              m_is_root(is_root),
              m_track_grad(track_grad),
              m_parent_op(parent_op){
                Setup_Memory(data, this->m_size);
              }

      Tensor(const std::vector<int>& shape,
             const bool& track_grad):
                Tensor(nullptr, shape, true, track_grad, nullptr){        
        Fill(DataType(0), this->m_size);
      }

      Tensor(const DataType& fill_val,
             const std::vector<int>& shape,
             const bool& track_grad):
              Tensor(nullptr, shape, true, track_grad, nullptr){
        Fill(fill_val, this->m_size);
      }

      ~Tensor(){
        static_cast<TensorImpl*>(this)->clean_up_impl();
      }
      
      void Setup_Memory(DataType* data, const size_t& size){
        if (data == nullptr){
          auto allocate_val = Allocate(size);
          if (!allocate_val){
            std::cout << "Data allocation failed" << std::endl;
          }
        }
        
        if (m_track_grad){
          auto allocate_grad_val = Allocate_Grad(size);
          if (!allocate_grad_val){
            std::cout << "Gradient allocation failed" << std::endl;
          }
        }
      }
      /// @brief 
      /// @param size 
      /// @return 
      int Allocate(const size_t& size){
        // std::cout << "Calling allocate" <<std::endl;
        return static_cast<TensorImpl*>(this)->allocate_memory_impl(size);
      }

      int Allocate_Grad(const size_t& size){
        return static_cast<TensorImpl*>(this)->allocate_grad_memory_impl(size);
      }
      
      // TODO: Template the fill_val here to be more than general than DataType
      int Fill(const DataType& fill_val, const int& size){
        return static_cast<TensorImpl*>(this)->fill_impl(fill_val, size);      
      }

      // TODO: Template the fill_val here to be more than general than DataType
      int Fill_Grad(const DataType& fill_val, const int& size){
        return static_cast<TensorImpl*>(this)->fill_grad_impl(fill_val, size);      
      }

      void 
      Add_Parent(Tensor<DataType, TensorImpl>* parent_ptr){
        m_parent_tensors.push_back(parent_ptr);
      }
      std::ostream& Print(std::ostream& os){
        return static_cast<TensorImpl*>(this)->print_impl(os);      
      }

      /// @brief 
      void Backward(){

        if (m_track_grad){
          Fill_Grad(1, this->m_size);
          // std::cout << "Calling backward 2\n";
          m_parent_op->backward();
          // std::cout << "Calling backward 3\n" << m_is_root << '\n';
          if (!m_is_root){

            for (auto i =0; i<m_parent_tensors.size(); ++i){
              // std::cout << 
              m_parent_tensors[i]->Backward(this);  
            }
          }
        }
      }

      void Backward(Tensor* upstream_tensor){
        
        if (m_track_grad){
          if (m_parent_op){
              m_parent_op->backward(static_cast<AbstractTensor<DataType>*>(this));  
          }
          if (!m_is_root){
            // std::cout << "Calling backward of non-leaf node\n";
            for (auto i =0; i<m_parent_tensors.size(); ++i){
              
              m_parent_tensors[i]->Backward(this);  
            }
            // std::cout << "Finished calling backward of non-leaf node\n";
          }
        }
      }

      friend std::ostream& operator<<(std::ostream& os, Tensor& tensor){
        tensor.Print(os);
        if (tensor.m_track_grad){
          os << " Gradient enabled \n";
        }
        return os;
      }
  };  // class definition Tensor

} // namespace Delta