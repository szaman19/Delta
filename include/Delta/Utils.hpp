#pragma once
#include <cstddef>
#include <iostream>
#include <vector>

namespace Delta{
  enum Device {
    CPU,
    GPU
  };

  template <typename... Args>
  struct Shape{
    /* data */
    std::vector<int> m_dims;
    /* constructors */
    Shape(Args&&... args){
      m_dims = {std::forward<Args>(args)...};
    }
    Shape(std::vector<int> dims):m_dims(dims){}
    
    /* Rule of Three */
    ~Shape() = default;

    Shape (const Shape& other):Shape(other.m_dims){}

    template <typename... T>
    Shape& operator=(const Shape<T...>& other){
      m_dims = other.m_dims;
      return *this;
    }

    /* operator overloads */
    template <typename... T>
    friend std::ostream& operator<<(std::ostream& os, const Shape<T...>& shape);

    template <typename... U, typename... V>
    friend bool operator==(const Shape<U...>& lhs, const Shape<V...>& rhs);
  };

  template <typename... T>
  std::ostream& operator<<(std::ostream& os, const Shape<T...>& shape){
    os << '(';
    const auto& dims = shape.m_dims; 
    const auto size = dims.size();

    if (size == 1){
      return os << dims[0] << ")" << std::endl;
    }

    for (auto iter = dims.begin(); iter != dims.end()-1; iter++){
      os << *iter << ", "; 
    }
      os << *(dims.end()-1) << ')'<<std::endl;
    return os;
  }


  template <typename... U, typename... V>
  bool operator==(const Shape<U...>& lhs, const Shape<V...>& rhs){
    const auto& lhs_dims = lhs.m_dims;
    const auto& lhs_size = lhs_dims.size();

    const auto& rhs_dims = rhs.m_dims;
    const auto& rhs_size = rhs_dims.size();

    if (lhs_size != rhs_size){
      return false;
    }

    bool return_val = true;

    for (auto i = 0; i <lhs_size; ++i ){
      return_val = (lhs_dims[i] == rhs_dims[i]) && return_val; 
    }
    return return_val;
  }
} // namespace Delta