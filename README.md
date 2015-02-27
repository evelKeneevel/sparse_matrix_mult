# sparse_matrix_mult
An C++11 implementation of sparse matrix formats, using both OpenCL and std::thread. Based on the paper: http://www.nvidia.com/object/nvidia_research_pub_001.html  
The code was written for a school project in January 2013.

Compile with:
g++ -g  -Wall -std=gnu++11  -I/usr/include/ -I./matrix_formats/ -I./random_matrix_test/  -L/usr/lib/libcln.so.6  -pthread random_matrix_test/main_prog.cpp -lOpenCL -o random_matrix_test/main_prog

Run with 
./random_matrix_test/main_prog
