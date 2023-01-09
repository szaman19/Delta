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

    Operator(/* args */) = default;
    ~Operator() = default;
    virtual std::string GetName(){
      return "Abstract Operator";
    }
    virtual void backward(Container* upstream_tensor){
      // TO DO: Make into an error
      std::cout << "Calling backward on an abstract operator makes no sense...\n";
    };
    virtual void backward(){
      // TO DO: Make into an error
      std::cout << "Calling backward on an abstract operator makes no sense...\n";
    };
  };
} // namespace Delta