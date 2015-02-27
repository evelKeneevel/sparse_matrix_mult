
#ifndef ____ellpack____
#define ____ellpack____

#include "abstract_format.hpp"

using namespace std;

template<typename T, size_t ROWS, size_t COLUMNS>
class ell_treater : public abstract_format<T, ROWS, COLUMNS> {

  size_t* indices = nullptr; 
  T* data = nullptr; 
  uint* gpu_indices = nullptr;  
  uint K_dim_gpu = 0;
  size_t K_dim = 0; // smallest number of zeros appearing on a row of the provided matrix, for the whole matrix 
  // Function overloading doesn't work when running the functions as threads. If two versions of F exist, 
  // when run as threads, the compiler can't deduct from their arguments which one is being called
  void calculate_threaded_result( size_t start_index, size_t end_index, 
				  const array<T, COLUMNS> & multiplicant_vec, std::promise<array<T, ROWS>> && promise );

  std::array<T, ROWS> && threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec );  
  std::array<T, ROWS> && normal_multiplication( const array<T, COLUMNS> & multiplicant_array );
  void create_data_arrays(); 

  struct execute_info create_gpu_program();
  std::array<T, ROWS> && execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel );
  void prepare_gpu_data( T* & data, uint* & indices, uint & K_dim_gpu );
 
public:
  ell_treater( const array<T, ROWS*COLUMNS> & input_array, size_t K_dim );
  ell_treater( const array<T, ROWS*COLUMNS> & input_array, size_t used_threads, size_t K_dim );
  ~ell_treater(){ 
		  if( indices != nullptr ) delete [] indices;  
		  if( data != nullptr ) delete [] data;	
		  if( gpu_indices != nullptr ) delete [] gpu_indices;	
		};
};


template<typename T, size_t ROWS, size_t COLUMNS>
void 
ell_treater<T, ROWS, COLUMNS>::prepare_gpu_data( T* & data, uint* & gpu_indices, uint & K_dim_gpu ) {
 
  if( K_dim == 0 ) { // discover the K dimension, if not provided as an argument 
    size_t non_zeros = 0; 
    for( size_t i = 0; i < ROWS; i++ ) {

      for( size_t j = 0; j < COLUMNS; j++ ) 
	if( (*this->input_array)[ i*COLUMNS + j ] != 0 ) non_zeros++;

      if( non_zeros == COLUMNS ) { K_dim_gpu = COLUMNS; break; } // #elements == COLUMNS are non-zero, so we stop the loop 	
      if( non_zeros > K_dim_gpu ) K_dim_gpu = non_zeros;	
      non_zeros = 0; 
    }
  } else K_dim_gpu = uint( K_dim );

  cout << "K dim is: "<< K_dim_gpu << endl;

  data = new T[ ROWS*K_dim_gpu ]; // data must get all values assigned 
  gpu_indices = new uint[ ROWS*K_dim_gpu ]; // indices is left with some values unassigned (thus, uninitialized)

  size_t data_j = 0;
  for( size_t i = 0; i < ROWS; i++ ) {

     data_j = 0;
     for( size_t j = 0; j < COLUMNS; j++ ) { 
	
	if( (*this->input_array)[ i*COLUMNS + j ] != 0 ) {
	   data[ i*K_dim_gpu + data_j ] = (*this->input_array)[ i*COLUMNS + j ];
//	   cout<<"Placing elem ["<<i<<"]["<<j<<"] in data pos ["<<i<<"]["<<data_j<<"]"<<endl;
	   gpu_indices[ i*K_dim_gpu + data_j ] = j;
	   data_j++;	
	}
     }
     for( size_t l = data_j; l<K_dim_gpu; l++ ) data[ i*K_dim_gpu + l ] = 0; // Fill remainder of row with 0s
  }

}


template<typename T, size_t ROWS, size_t COLUMNS>
struct execute_info 
ell_treater<T, ROWS, COLUMNS>::create_gpu_program() {

    FILE *fp = fopen("./matrix_formats/ell/ell_kernel.cl", "r");
    if(!fp) { fprintf(stderr, "Failed to load kernel.\n");  exit(1); }

    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    if( !this->created_gpu_data ) { // creating GPU data once only

       prepare_gpu_data( data, gpu_indices, K_dim_gpu );
 
       this-> created_gpu_data = true;	 
    }

    // STEP 3: Create a context
    cl_context context = clCreateContext( NULL, this->num_devices, &(this->devices[0]),NULL,NULL, &this->return_status);
 
    // STEP 4: Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, this->devices[0], 0, &this->return_status);

    // STEP 5: Create device buffers 
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, ROWS*K_dim_gpu*sizeof(T), NULL, &this->return_status);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, ROWS*K_dim_gpu*sizeof(uint), NULL, &this->return_status);
    cl_mem d_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint), NULL, &this->return_status);
    cl_mem f_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, COLUMNS*sizeof(T), NULL, &this->return_status);
    this->res_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ROWS*sizeof(T), NULL, &this->return_status);

    // STEP 6: Write host data to device buffers
    // Last 3 arguments declare events that have to be completed before this current event starts
    this->return_status = clEnqueueWriteBuffer( command_queue, b_mem_obj, CL_TRUE, 0, 
						ROWS*K_dim_gpu*sizeof(T),(void*) data, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, c_mem_obj, CL_TRUE, 0, 
						ROWS*K_dim_gpu*sizeof(uint),(void*) gpu_indices, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, d_mem_obj, CL_TRUE, 0, 
						sizeof(uint),(void*) &K_dim_gpu, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, f_mem_obj, CL_TRUE, 0, 
						COLUMNS*sizeof(T),(void*) this->mult_vec, 0, NULL, NULL);

    // STEP 7: Create and compile the program  
    // second argument is 1 always ?
    cl_program program = clCreateProgramWithSource( context, 1, (const char **)&source_str, 
						    (const size_t *)&source_size, &this->return_status ); 

    this->return_status = clBuildProgram(program, this->num_devices, this->devices, NULL, NULL, NULL);

    // STEP 8: Create the kernel
    // Second argument is the kernel function name declared with the __kernel qualifier, in the .cl file  
    cl_kernel kernel = clCreateKernel(program,string(this->type_identification()+"_compute").data(),&this->return_status);

    // STEP 9: Set the kernel arguments
    // Second argument is the argument index, in the __kernel function. Arguments to the kernel are referred by 
    // indices that go from 0 for the leftmost argument to n - 1, where n is the total number of arguments 
    // declared by a kernel

// The function call clSetKernelArg(kernel, 1, K_dim*sizeof(T), 0 ); allocates an object of size K_dim*sizeof(T) 
// for LOCAL use by the kernel. Locality is defined with the last 0 argument. The kernel argument passed 
// is a pointer (which implies an array)

    this->return_status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&b_mem_obj);
    this->return_status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&c_mem_obj);
    this->return_status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_mem_obj);
    this->return_status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&f_mem_obj);
    this->return_status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&this->res_mem_obj);

    struct execute_info ex_info( command_queue, kernel );

    this->return_status = clReleaseProgram( program );
    this->return_status = clReleaseContext( context );

    return ex_info;
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
ell_treater<T, ROWS, COLUMNS>::execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ) {
 
    // STEP 10: Configure the work-item structure
    size_t globalWorkSize[1];
//    size_t localWorkSize[1];
    globalWorkSize[0] = ROWS;  // There are ROWS global work-items 
//    localWorkSize[0] = 1; // COLUMNS/4; 

    // STEP 11: Enqueue the kernel for execution

    // Arg #3 is work_dim: the number of dimensions used to specify 
    // the global work-items and work-items in the work-group. 
    // work_dim must be greater than zero and less than or equal to CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS.

    // Arg #5 is *global_work_size. Points to an array of work_dim unsigned values that describe 
    // the number of global work-items in work_dim dimensions that will execute the kernel function.
    // The total number of global work-items is computed as global_work_size[0] *...* global_work_size[work_dim - 1]. 
    this->return_status = clEnqueueNDRangeKernel(command_queue, kernel, 
					         1, // number of dimensions used 
					   	 NULL, 
     					   	 globalWorkSize, 
					   	 NULL, //localWorkSize,  //NULL,
					   	 0, NULL, NULL);

    // STEP 12: Read the output buffer back to the host
    // Read the memory buffer C on the device to the local variable Result
    std::array<T, ROWS> *Result = new std::array<T, ROWS>; //LIST_SIZE); 

    // Arg #2 is buffer, refers to a valid buffer object.
    // The data is read and copied to C. CL_TRUE makes clEnqueueReadBuffer to not return until the buffer data 
    // has been read and copied into Result
    // Copies c_mem_obj to Result
    this->return_status = clEnqueueReadBuffer( command_queue, this->res_mem_obj, CL_TRUE, 0, 
				               ROWS*sizeof(T), (void *)(*Result).data(), 0, NULL, NULL );

    this->return_status = clReleaseKernel(kernel);
    this->return_status = clReleaseCommandQueue(command_queue);

    // Display the result to the screen
    cout<<"ELL GPU result :"; for( T &i : *Result ) cout << i << " ";  cout << endl;

    return std::move( *Result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
ell_treater<T, ROWS, COLUMNS>::threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec ) {

  vector<std::thread> thread_vector;     
  vector<std::future<array<T, ROWS>>> futures;
  array<T, ROWS> *result = new array<T, ROWS>; for( size_t i = 0; i < (*result).size(); i++ ) (*result)[i] = T(0);
 
  size_t step = this->calculate_step();

  for( size_t i=0; i<size_t(this->used_threads); i++ ) {  

     std::promise<array<T, ROWS>> promise;
     futures.push_back( std::future<array<T, ROWS>>() );
     futures.back() = promise.get_future();	

     size_t finish = ( (i+1)*step < (*this->input_array).size() ) ? (i+1)*step -1 : (*this->input_array).size() -1;
     // If some elements are remaining, they are all placed in the last thread's work
     if( i == size_t(this->used_threads -1) && finish < (*this->input_array).size() -1 ) finish = (*this->input_array).size() -1;
 
//     cout << i*step << " " << finish << " step is:" << step << endl;	
   
     thread_vector.push_back( std::thread{ &ell_treater::calculate_threaded_result, this, 
					   i*step, finish, std::ref( multiplicant_vec ), std::move(promise) } );

     if( finish == (*this->input_array).size() -1 ) i = this->used_threads; // Break loop, if all elements are reached
  }
 
  for( std::thread &i : thread_vector ) i.join(); // range_based loop 

  for( std::future<array<T, ROWS>> &fut : futures ) { 
	array<T, ROWS> partial_res( fut.get() ); 
	for( size_t j=0; j<(*result).size(); j++ ) (*result)[j] += partial_res[j];	
  }

  cout<<"Threaded ell result : "; for( size_t i =0; i< (*result).size(); i++ ) cout << (*result)[i]<<" "; cout<<endl; 
  thread_vector.clear();  futures.clear();
  return std::move( *result );
} 


template<typename T, size_t ROWS, size_t COLUMNS>
void ell_treater<T, ROWS, COLUMNS>::calculate_threaded_result( size_t start_index, size_t end_index, 
						      	       const array<T, COLUMNS> & multiplicant_vec, 
							       std::promise<array<T, ROWS>> && promise ) { 
  size_t my_K_dim = 0;  size_t rows = 1;
  std::array<T, ROWS> partial_result;  for( size_t i = 0; i< partial_result.size(); i++ ) partial_result[i] = T(0);

  // Computes the dimensions MxN ( data_rows x K_dim ) for data & indices arrays, for each particular thread 
  // i, j will point to the position in a two-dimensional array
  size_t non_zeros = 0, j = start_index % COLUMNS, i = ( start_index - j )/COLUMNS; 

//  cout << "start index "<< start_index << " Starting at ["<<i<<"]["<<j<<"]"<<endl;

  for( size_t times = start_index; times <= end_index; times++ ) { 

       if( (*this->input_array)[ i*COLUMNS+j ] != 0 ) non_zeros++;
 
       if( non_zeros == COLUMNS ) { my_K_dim = COLUMNS; } // #elements == COLUMNS are non-zero, so we stop the loop
       else if( non_zeros > my_K_dim ) my_K_dim = non_zeros;	

       // advance further in the input array	 
       if( j == COLUMNS-1 ) { j = 0; i++; non_zeros = 0; if( times != end_index ) rows++; } 
       else j++;
  }

//  cout << "Thread K_dim is "<< my_K_dim << " data rows are "<< rows <<endl;

  T* my_data = new T[ rows*my_K_dim ]; 
  size_t* my_indices = new size_t[ rows*my_K_dim ];

  j = start_index % COLUMNS;
  i = ( start_index - j )/COLUMNS; 
  size_t data_i = 0, data_j = 0;
  size_t row_offset = i; // Our data correspond to a particular start of the result array 
 
  for( size_t times = start_index; times <= end_index; times++ ) { 

       if( (*this->input_array)[ i*COLUMNS+j ] != 0 ) { 

	   my_data[ data_i*my_K_dim + data_j ] = (*this->input_array)[ i*COLUMNS+j ];
	   my_indices[ data_i*my_K_dim + data_j ] = j; 
	   data_j++; 
       }	

       if( j == COLUMNS-1 ) { // reached end of input row

	   j = 0; i++; 
	   for( size_t l = data_j; l<my_K_dim; l++ ) { my_data[ data_i*my_K_dim + l ] = T(0); }

	   data_j = 0;  data_i++;
	
       } else j++; // advance further in the input array	 
  }

  // Maybe my_data wasn't filled totally, so uninitialized elements exist.
  for( size_t i=data_i*my_K_dim + data_j; i<rows*my_K_dim; i++ )  my_data[i] = T(0);  
//  for( size_t i=0; i<rows*my_K_dim; i++ ) cout<< my_data[i] <<" "; cout<<endl;

  j = start_index % COLUMNS;
  i = ( start_index - j )/COLUMNS; 

//  cout << "start index "<< start_index << " Starting at ["<<i<<"]["<<j<<"]"<<endl;
 
  for( size_t i = 0; i < rows; i++ )  
     for( size_t j = 0; j < my_K_dim; j++ ) { 

      if( my_data[ i*my_K_dim+j ] != 0 )  
	partial_result[i+row_offset] = (this->*(this->multiply_and_add))( my_data[ i*my_K_dim + j ], 
									  multiplicant_vec[ my_indices[ i*my_K_dim + j ] ], 
									  partial_result[ i+row_offset ] ); 	
     }

  delete [] my_data; delete [] my_indices; // delete thread allocated stuff

  promise.set_value( partial_result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
ell_treater<T, ROWS, COLUMNS>::normal_multiplication( const array<T, COLUMNS> & multiplicant_vec ) {

  array<T, ROWS> *result = new array<T, ROWS>; for( size_t i = 0; i < (*result).size(); i++ ) (*result)[i] = T(0);
 
  for( size_t i=0; i<ROWS; i++ )
    for( size_t j=0; j<K_dim; j++ ) {

      if( data[ i*K_dim+j ] != 0 )  
	(*result)[i] = (this->*(this->multiply_and_add))( data[ i*K_dim+j ], 
							  multiplicant_vec[ (indices)[i*K_dim+j] ],   
							  (*result)[i] ); 
    }	  

  cout<<"Normal ell result : "; for( size_t i =0; i< (*result).size(); i++ ) cout << (*result)[i]<<" "; cout<<endl;
  // data & indices are allocated in memory

  return std::move( *result );
}



template<typename T, size_t ROWS, size_t COLUMNS>
void 
ell_treater<T, ROWS, COLUMNS>::create_data_arrays() { 
 
  if( K_dim == 0 ) { // discover the K dimension, if not provided as an argument 
    size_t non_zeros = 0; 
    for( size_t i = 0; i < ROWS; i++ ) {

      for( size_t j = 0; j < COLUMNS; j++ ) 
	if( (*this->input_array)[ i*COLUMNS + j ] != 0 ) non_zeros++;

      if( non_zeros == COLUMNS ) { K_dim = COLUMNS; break; } // #elements == COLUMNS are non-zero, so we stop the loop 	
      if( non_zeros > K_dim ) K_dim = non_zeros;	
      non_zeros = 0; 
    }
  }

  cout << "K dim is: "<< K_dim << endl;

  data = new T[ ROWS*K_dim ]; 
  indices = new size_t[ ROWS*K_dim ];

  for( size_t i = 0; i < ROWS; i++ ) {
	
     size_t data_idx = 0;
     for( size_t j = 0; j < COLUMNS; j++ ) { 
	
	if( (*this->input_array)[ i*COLUMNS + j ] != 0 ) {
	   data[ i*K_dim + data_idx ] = (*this->input_array)[ i*COLUMNS + j ];
//	   cout<<"Placing elem ["<<i<<"]["<<j<<"] in data pos ["<<i<<"]["<<data_idx<<"]"<<endl;
	   indices[ i*K_dim + data_idx ] = j;
	   data_idx++;	
	}
     }
     for( size_t l = data_idx; l<K_dim; l++ ) data[ i*K_dim + l ] = 0; // Fill remainder of row with 0s
  }

  



}


template<typename T, size_t ROWS, size_t COLUMNS>
ell_treater<T, ROWS, COLUMNS>::ell_treater( const array<T, ROWS*COLUMNS> & input_array, size_t K_dim ) 
			     : abstract_format<T, ROWS, COLUMNS>( input_array ) {
    
   if( K_dim > 0 ) this-> K_dim = K_dim;
}


template<typename T, size_t ROWS, size_t COLUMNS>
ell_treater<T, ROWS, COLUMNS>::ell_treater( const array<T, ROWS*COLUMNS> & input_array, 
					    size_t used_threads, size_t K_dim ) 
			     : abstract_format<T, ROWS, COLUMNS>( input_array, used_threads ) {

   if( K_dim > 0 ) this-> K_dim = K_dim;
}


#endif

