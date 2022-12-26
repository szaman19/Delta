CC = clang++
COMPILER_OPTIONS = -std=c++17
COMPILE_TEST_DIR = tests/compile_tests
HEADER_DIR = include/

test_shape_func: $(COMPILE_TEST_DIR)/shape_initializer_test.cpp
		$(CC) $(COMPILE_TEST_DIR)/shape_initializer_test.cpp -I $(HEADER_DIR) $(COMPILER_OPTIONS) -o shape_init_test	

