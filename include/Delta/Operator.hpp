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
  };
  
  template <typename Container>
  Operator<Container>::Operator(/* args */){
  }

  template <typename Container>
  Operator<Container>::~Operator(){
  }
  
} // namespace Delta