#pragma once
#include <cstddef>
#include <iostream>
#include <vector>

#ifndef NDEBUG
#   define Delta_Assert(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define Delta_Assert(condition, message) do { } while (false)
#endif

namespace Delta{
  enum Device {
    CPU,
    GPU
  };

  template<typename T>
  std::ostream& vec_to_str(const T* vector, const int& num_elem){
    std::ostream& os = std::cout;
    os << "[";

    if (num_elem == 1){
      os << vector[0] << "]";
      return os;
    }

    for (auto i = 0; i < num_elem-1; ++i){
      os << vector[i] << ", ";
    }
    os << vector[num_elem-1]<< "]";
    return os;
  }

  template <typename... Args>
  struct Shape{
    /* data */
    std::vector<int> m_dims;
    /* constructors */
    Shape(Args&&... args){
      m_dims = {std::forward<Args>(args)...};
    }
    Shape(std::vector<int> dims):m_dims(dims){}
    Shape(std::initializer_list<int> dims):m_dims(dims){}
    
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
      const auto& dims = shape.m_dims; 
      const auto size = dims.size();
      return vec_to_str<int> (dims.data(), size);
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