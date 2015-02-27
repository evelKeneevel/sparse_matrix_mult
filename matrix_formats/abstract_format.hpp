
#ifndef ____abstract_format____
#define ____abstract_format____

#include <iostream> 
#include <string>
#include <istream>
#include <array>
#include <sstream>
#include <thread>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <utility>
#include <algorithm>
#include <mutex>
#include <exception>
#include <future>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <typeinfo>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x40000000)

using namespace std;

struct execute_info {
  cl_command_queue command_queue;
  cl_kernel kernel;
  execute_info( cl_command_queue & command_queue, cl_kernel & kernel ) {

     this-> command_queue = command_queue;
     this-> kernel = kernel;
  }
};

template<typename T, size_t ROWS, size_t COLUMNS>
class abstract_format {

protected: 
  cl_platform_id *platforms = NULL;
  cl_device_id *devices = NULL;   
  cl_uint num_devices = 0;
  cl_uint num_platforms = 0;
  cl_int return_status; // General return error value, after a function call
  cl_mem res_mem_obj;
  T* mult_vec = nullptr;
  std::array<T, COLUMNS>* mult_vec_ptr = nullptr;
  void create_device();
  virtual struct execute_info create_gpu_program() = 0;
  virtual std::array<T, ROWS> && execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ) = 0; 
  // OpenCL stuff ends here

  bool created_data = false; 
  bool created_gpu_data = false;
  bool created_gpu_device = false;
  int used_threads = 0;  
  array<T, ROWS*COLUMNS> * input_array;
  virtual void create_data_arrays( void ) = 0;
  size_t calculate_step();
  virtual std::array<T, ROWS> && threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec ) = 0;   
  virtual std::array<T, ROWS> && normal_multiplication( const array<T, COLUMNS> & multiplicant_array ) = 0; 
  array<T, ROWS> && multiply_with_gpu( const array<T, COLUMNS> & multiplicant_vec );
  const T (abstract_format::*multiply_and_add)( const T &, const T &, const T& ); // function pointer, points to class member functions
  inline const T int_mult( const T &a, const T &b, const T &c ) { return a*b +c; }
  inline const T fma_mult( const T &a, const T &b, const T &c ) { return fma( a, b, c ); }
 
  string type_identification();
  abstract_format();   
  abstract_format( const array<T, ROWS*COLUMNS> & input_array ); 
  abstract_format( const array<T, ROWS*COLUMNS> & input_array, int used_threads );
public:
  array<T, ROWS> && multiply_with_vector( const array<T, COLUMNS> & multiplicant_array );  
  array<T, ROWS> && operator*( const array<T, COLUMNS> & multiplicant_array );  
  virtual ~abstract_format(){ 
		      	     if( platforms != NULL ) free( platforms );
  		             if( devices != NULL ) free( devices );  
   		    	    };
};


template<typename T, size_t ROWS, size_t COLUMNS>
array<T, ROWS> && 
abstract_format<T, ROWS, COLUMNS>::operator*( const array<T, COLUMNS> & multiplicant_array ) {

   return std::move( this->multiply_with_vector( multiplicant_array ) ); 
}


template<typename T, size_t ROWS, size_t COLUMNS>
abstract_format<T, ROWS, COLUMNS>::abstract_format( const array<T, ROWS*COLUMNS> & input_array, int used_threads ) 
				 : abstract_format<T, ROWS, COLUMNS>() {

  // the default abstract_format() constructor gets called as well
  if( int(input_array.size()) < used_threads && used_threads != -1 ) used_threads = int(input_array.size());

  this-> used_threads = used_threads;
  this-> input_array = (array<T, ROWS*COLUMNS> *) &input_array;
}


template<typename T, size_t ROWS, size_t COLUMNS>
abstract_format<T, ROWS, COLUMNS>::abstract_format( const array<T, ROWS*COLUMNS> & input_array ) 
				 : abstract_format<T, ROWS, COLUMNS>() { 

  // the default abstract_format() constructor gets called as well
  this-> input_array = (array<T, ROWS*COLUMNS> *) &input_array;
}


template<typename T, size_t ROWS, size_t COLUMNS>
abstract_format<T, ROWS, COLUMNS>::abstract_format() { 

  // base constructor, if invalid data is supplied, an exception is thrown 
  string type_res; 

  try{ type_res = type_identification(); } 
  catch( std::exception & ){ cout<<"Unappropriate template type!"<<endl; }  
  // assigning the appripriate multiplication function, for the appropriate type T.
  if( type_res == "int" || type_res == "short" ) multiply_and_add = &abstract_format::int_mult;  
  else if( type_res == "float" || type_res == "double" 
	  || type_res == "long" || type_res == "edouble" ) multiply_and_add = &abstract_format::fma_mult; 
 
}


template<typename T, size_t ROWS, size_t COLUMNS>
string abstract_format<T, ROWS, COLUMNS>::type_identification() {
   
  string result = "error";	
  // All supported data types are here, an unsupported type throws an exception
  if( typeid(T).name()[0] == 'f' ) { result = "float"; }
  else if( typeid(T).name()[0] == 'i' ) result = "int";
  else if( typeid(T).name()[0] == 'd' ) result = "double";
  else if( typeid(T).name()[0] == 'l' ) result = "long";
  else if( typeid(T).name()[0] == 's' ) result = "short";
  else if( typeid(T).name()[0] == 'e' ) result = "edouble"; // long double type

  if( result == "error" ) throw std::bad_typeid(); // bad type exception 

  return result;
}


template<typename T, size_t ROWS, size_t COLUMNS>
size_t abstract_format<T, ROWS, COLUMNS>::calculate_step() {

  // Reassign the used_threads variable if array elements are less than the supplied used_threads value
  if( (*input_array).size() < size_t(used_threads) ) used_threads = int( (*input_array).size() );  

//  cout <<"Using "<< used_threads << " threads"<<endl;
 
  size_t step; // input_array.size()/used_threads defines the step;
  if( ceil((float) (*input_array).size()/used_threads) - (float) (*input_array).size()/used_threads > 0.5 ) 
	  step = floor( (float) (*input_array).size()/used_threads );  
  else step = ceil( (float) (*input_array).size()/used_threads );

  return step;
}


template<typename T, size_t ROWS, size_t COLUMNS>
void abstract_format<T, ROWS, COLUMNS>::create_device() {

    // STEP 1: Discover and initialize the platforms
    return_status = clGetPlatformIDs( num_platforms, NULL, &num_platforms);
    platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
    return_status = clGetPlatformIDs(num_platforms, platforms, &num_platforms);

    // STEP 2: Discover and initialize the devices
    return_status = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0, devices, &num_devices);
    devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
    return_status = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, &num_devices);
}


template<typename T, size_t ROWS, size_t COLUMNS>
array<T, ROWS> && 
abstract_format<T, ROWS, COLUMNS>::multiply_with_vector( const array<T, COLUMNS> & multiplicant_vec ) {

  this->mult_vec = (T*) multiplicant_vec.data();
  this->mult_vec_ptr = (array<T, COLUMNS>*) &multiplicant_vec;
 
  // No threads are used in this case, otherwise the threaded multiplication takes place 
  if(this->used_threads==0) { 
     // Creating the data arrays only once, if instance is reused then the data stays the same	
     if(!created_data) { create_data_arrays();  created_data = true; }

     return std::move( normal_multiplication( multiplicant_vec ) ); 

  } else if(this-> used_threads == -1) { 
	return std::move( multiply_with_gpu( multiplicant_vec ) );
  }
 
  return std::move( threaded_multiplication( multiplicant_vec ) ); 
}


template<typename T, size_t ROWS, size_t COLUMNS>
array<T, ROWS> && 
abstract_format<T, ROWS, COLUMNS>::multiply_with_gpu( const array<T, COLUMNS> & multiplicant_vec ) {

  // Creating GPU data only once
  if(!created_gpu_device) { create_device();  created_gpu_device = true; }  

  struct execute_info ret_info = create_gpu_program(); 
 
  return std::move( execute_gpu_program( ret_info.command_queue, ret_info.kernel ) ); 
}


#endif
