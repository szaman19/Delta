#pragma once
#include "Utils.hpp"

namespace Delta{
  // Forward declaration
  template <typename DataType>
  class AbstractTensor;

  /// @brief 
  /// @tparam Container 
  template <typename Container>
  class Operator{
  private:
    /* data */
  public:
    Operator(/* args */);
    ~Operator();
    void backward(const Container* upstream_tensor){
      backward_impl(upstream_tensor);
    }

    void backward(){
      backward_impl();
    }
    virtual void backward_impl(const Container* upstream_tensor) = 0;
    virtual void backward_impl() = 0;
  };
  
  template <typename Container>
  Operator<Container>::Operator(/* args */){
  }

  template <typename Container>
  Operator<Container>::~Operator(){
  }
  
} // namespace Delta