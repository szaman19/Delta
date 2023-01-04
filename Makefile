CC = clang++
COMPILER_OPTIONS = -std=c++17 -Wall
COMPILE_TEST_DIR = tests/compile_tests
HEADER_DIR = include/
REFERENCE_DIR = Reference/

all: test_autograd test_shape_func test_tensor_func

test_autograd : $(COMPILE_TEST_DIR)/autograd_tensor_test.cpp
		$(CC) $(COMPILE_TEST_DIR)/autograd_tensor_test.cpp -I $(HEADER_DIR) -I $(REFERENCE_DIR) $(COMPILER_OPTIONS) -o autograd_tensor_test	

test_tensor_func: $(COMPILE_TEST_DIR)/tensor_compile_test.cpp
		$(CC) $(COMPILE_TEST_DIR)/tensor_compile_test.cpp -I $(HEADER_DIR) -I $(REFERENCE_DIR) $(COMPILER_OPTIONS) -o tensor_compile_test	

test_shape_func: $(COMPILE_TEST_DIR)/shape_initializer_test.cpp
		$(CC) $(COMPILE_TEST_DIR)/shape_initializer_test.cpp -I $(HEADER_DIR) $(COMPILER_OPTIONS) -o shape_init_test	
