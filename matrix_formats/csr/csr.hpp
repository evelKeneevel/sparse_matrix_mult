
#ifndef ____csr____
#define ____csr____

#include "abstract_format.hpp"
//#include <chrono>
#include <sys/syscall.h>
#include <sys/types.h>

const size_t LOCAL_WORK_SIZE = 32;


using namespace std;

template<typename T, size_t ROWS, size_t COLUMNS>
class csr_treater : public abstract_format<T, ROWS, COLUMNS> {

  vector<T> data;
  vector<size_t> indices;
  array<size_t, ROWS+1> pointers;
  size_t nonzero_elements; // holds the total number of elements to process
  uint* my_indices = nullptr;
  uint* my_pointers = nullptr;
  T* my_data = nullptr;
  std::array<T, ROWS> && normal_multiplication( const array<T, COLUMNS> & multiplicant_vec );
  std::array<T, ROWS> && threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec );
  void create_data_arrays(); 
  void create_threaded_result( const array<T, COLUMNS> & multiplicant_vec, 
			       size_t start, size_t finish, std::promise<array<T, ROWS>> && results_promise );
  void create_gpu_data( uint * my_indices, T * my_data, uint* my_pointers, size_t & nonzero_elements );
  struct execute_info create_gpu_program();
  std::array<T, ROWS> && execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ); 
public: 
  csr_treater( const array<T, ROWS*COLUMNS> & input_array );
  csr_treater( const array<T, ROWS*COLUMNS> & input_array, size_t used_threads );
  ~csr_treater(){ data.clear(); indices.clear(); pointers.empty(); 
		  if( my_data != nullptr ) delete [] my_data;		
		  if( my_indices != nullptr ) delete [] my_indices;	
		  if( my_pointers != nullptr ) delete [] my_pointers;	
		};
};


template<typename T, size_t ROWS, size_t COLUMNS>
void csr_treater<T, ROWS, COLUMNS>::create_gpu_data( uint* my_indices, T * my_data, 
						     uint* my_pointers, size_t & nonzero_elements ) { 

// It's way faster to work with plain matrixes, compared to other data structures like vectors e.c.t. 

   int non_zero_elem_of_row = 0;
   my_pointers[0] = 0;
   nonzero_elements = 0;

   for( size_t i=0; i < ROWS; i++ ) 
     for( size_t j=0; j < COLUMNS; j++ ) {	

	if( (*this->input_array)[i*COLUMNS + j] != 0 ) {

	   my_indices[ nonzero_elements ] = j;
	   my_data[ nonzero_elements ] = (*this->input_array)[i*COLUMNS+j];
	   non_zero_elem_of_row++;  
	   nonzero_elements++;
        }
 
	if( j == COLUMNS -1 ) {
	   my_pointers[ i+1 ] = non_zero_elem_of_row;
	   non_zero_elem_of_row = 0;
	}
    }

  for( size_t i=0; i<ROWS; i++ ) my_pointers[i+1] += my_pointers[i];

//  cout << "Pointers :"; for( size_t i=0; i<ROWS+1; i++ ) cout << my_pointers[i] << " "; cout<<endl;
//  cout << "Indices :"; for( size_t i=0; i<ROWS*COLUMNS; i++ ) cout << my_indices[i] << " "; cout<<endl;
//  cout << "Data :"; for( size_t i=0; i<ROWS*COLUMNS; i++ ) cout << my_data[i] << " "; cout<<endl;
}

/*
template<typename T, size_t ROWS, size_t COLUMNS>
void calculate_stuff( T* data, int* indices, int* pointers, int* multiplicant_vec ) {

  T result[ROWS];

  for( int i=0; i<ROWS; i++ ) {

     int elem_start = pointers[i];
     int elem_finish = pointers[i+1];
     T part_res = 0;
  
     int vals[LOCAL_WORK_SIZE]; //for( int i=0; i<LOCAL_WORK_SIZE; i++ ) vals[i] = 0;

     for( int warp_idx =0; warp_idx<LOCAL_WORK_SIZE; warp_idx++ ) { 

       vals[warp_idx] = 0;
       for( int j = elem_start + warp_idx; j < elem_finish; j += LOCAL_WORK_SIZE ) {
	  vals[warp_idx] += data[j]*multiplicant_vec[ indices[j] ];
       }
     }

     for( int i=0; i<LOCAL_WORK_SIZE; i++ ) part_res += vals[i];
     result[i] = part_res;
  }

  for( size_t i=0; i<ROWS; i++ ) cout << result[i] << " "; cout << endl;
}
*/

template<typename T, size_t ROWS, size_t COLUMNS>
struct execute_info 
csr_treater<T, ROWS, COLUMNS>::create_gpu_program() {

    FILE *fp = fopen("./matrix_formats/csr/csr_kernel.cl", "r");
    if(!fp) { fprintf(stderr, "Failed to load kernel.\n");  exit(1); }

    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    if( !this->created_gpu_data ) { // creating the GPU data only once

       // These arrays are filled up to a certain point. Number of valid values will be equal to nonzero_elements
       my_indices = new uint[ROWS*COLUMNS];  my_data = new T[ROWS*COLUMNS];
       my_pointers = new uint[ROWS+1];  

       create_gpu_data( my_indices, my_data, my_pointers, nonzero_elements );
       this->created_gpu_data = true;
    }
  
    // STEP 3: Create a context
    cl_context context = clCreateContext( NULL, this->num_devices, &(this->devices[0]),NULL,NULL, &this->return_status);
 
    // STEP 4: Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, this->devices[0], 0, &this->return_status);

    // STEP 5: Create device buffers 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, nonzero_elements*sizeof(T), NULL, &this->return_status);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, (ROWS+1)*sizeof(uint), NULL, &this->return_status); 
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, nonzero_elements*sizeof(uint),NULL,&this->return_status);
    cl_mem d_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, COLUMNS*sizeof(T), NULL, &this->return_status);
    this->res_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ROWS*sizeof(T), NULL, &this->return_status);

    // STEP 6: Write host data to device buffers
    // Last 3 arguments declare events that have to be completed before this current event starts
    // !!! Taking into consideration only the first nonzero_elements of my_data, my_indices !!! 
    this->return_status = clEnqueueWriteBuffer( command_queue, a_mem_obj, CL_TRUE, 0, 
						nonzero_elements*sizeof(T), my_data, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, b_mem_obj, CL_TRUE, 0, 
						(ROWS+1)*sizeof(uint), my_pointers, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, c_mem_obj, CL_TRUE, 0, 
						nonzero_elements*sizeof(uint), my_indices, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, d_mem_obj, CL_TRUE, 0, 
						COLUMNS*sizeof(T), (void*) this->mult_vec, 0, NULL, NULL);

    // clEnqueueWriteBuffer performs an implicit flush of the command-queue.

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

    this->return_status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    this->return_status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    this->return_status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    this->return_status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_mem_obj);
    this->return_status = clSetKernelArg(kernel, 4, LOCAL_WORK_SIZE*sizeof(T), 0 ); // __local array passed to kernel 
    this->return_status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&this->res_mem_obj);

    struct execute_info ex_info( command_queue, kernel );

    this->return_status = clReleaseProgram( program );
    this->return_status = clReleaseContext( context );
 
    return ex_info;
}



template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
csr_treater<T, ROWS, COLUMNS>::execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ) { 

 
    // STEP 10: Configure the work-item structure
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    globalWorkSize[0] = LOCAL_WORK_SIZE*ROWS;  // There are #ROWS work-items
    localWorkSize[0] = LOCAL_WORK_SIZE; //  COLUMNS/4; //ROWS*COLUMNS/20;  // FIX THIS!!

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
					   	 localWorkSize,  //NULL,
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

    cout << setprecision(16); // show 16 digits

    // Display the result to the screen
    cout<<"CSR GPU result :"; for( T &i : *Result ) cout << i << " ";  cout << endl;

    return std::move( *Result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
void 
csr_treater<T, ROWS, COLUMNS>::create_threaded_result( const array<T, COLUMNS> & multiplicant_vec, size_t start, 
						       size_t finish, std::promise<array<T, ROWS>> && results_promise ) {
 
  array<T, ROWS> partial_result; for( T &i : partial_result ) i = T(0);

//  array<T, ROWS> *partial_result = new array<T, ROWS>;  for( T &i : *partial_result ) i = T(0);

  array<size_t, ROWS+1> partial_pointers;  for( size_t &i : partial_pointers ) i = size_t(0);
  vector<T> partial_data;
  vector<size_t> partial_indices;
  size_t non_zero_elem_of_row = 0;  

  size_t j = start%COLUMNS;
  size_t i = (start - j)/COLUMNS;

  for( size_t times=start; times <= finish; times++ ) { 

	if( (*this->input_array)[i*COLUMNS+j] != 0 ) {
	   partial_indices.push_back( j );		
	   partial_data.push_back( (*this->input_array)[i*COLUMNS+j] ); 		
	   non_zero_elem_of_row++;  
        }
 
	if( j == COLUMNS -1 ) { // reached end of row
	   partial_pointers[ i+1 ] = non_zero_elem_of_row; 
	   non_zero_elem_of_row = 0;
	   j = 0; i++;

	} else j++; // continue row traversal

	if( times == finish )  
	   partial_pointers[ i + 1 ] = non_zero_elem_of_row;
  }

  for( size_t i=0; i<partial_pointers.size()-1; i++ ) 
     if( partial_pointers[i+1] != 0 ) partial_pointers[i+1] += partial_pointers[i];

  size_t start_i=1;
  while( partial_pointers[start_i] == 0 ) start_i++;

  // proceed until the finish value through the elements stored in data matrices.
  for( size_t counter =0; counter < partial_data.size(); counter++ ) { 

     partial_result[start_i -1] = (this->*(this->multiply_and_add))( partial_data[ counter ], 
					 			        multiplicant_vec[ partial_indices[counter] ], 
								        partial_result[start_i -1] );	

     if( counter == partial_pointers[start_i]-1 ) { 
//        cout << "Limit! store result ("<< partial_data[counter] <<"*"<< multiplicant_vec[ partial_indices[counter] ] 
//	       << ") at index " << start_i-1<< ", counter is "<< counter << endl;
	start_i++;
     }
  }

//  cout << setprecision(16);
  cout << "Partial result is: ";  for( T &i : partial_result ) cout << " "<< i; cout << endl;
  results_promise.set_value( partial_result ); 
  cout <<"Lala\n"; 
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
csr_treater<T, ROWS, COLUMNS>::threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec ) {

  for( size_t &i : pointers ) i = 0;
  vector<std::thread> thread_vector; 
  vector<std::future<array<T, ROWS>>> results_futures;
  array<T, ROWS> *result = new array<T, ROWS>; for( T &i : *result ) i = T(0); 

  size_t step = this->calculate_step();
 
  for( size_t i=0; i<size_t(this->used_threads); i++ ) {  
     // Creating new promise objects. Get the future of each promise object, and store it in the futures vectors.
     // Each thread returns a partial multiplication result. 
     //	In the end, all these arrays are joined, to give the final result 
/*     std::promise<array<T, ROWS>> results_promise;   
     results_futures.push_back( results_promise.get_future() ); 
*/
     std::promise<array<T, ROWS>> results_promise;
     results_futures.push_back( std::future<array<T, ROWS>>() );
     results_futures.back() = results_promise.get_future();	

     size_t finish = ( (i+1)*step < (*this->input_array).size() ) ? (i+1)*step -1 : (*this->input_array).size() -1;
     // If some elements are remaining, they are all placed in the last thread's work
     if( i == size_t(this->used_threads -1) && finish < (*this->input_array).size() -1 ) finish = (*this->input_array).size() -1;
 
     cout << i*step << " " << finish << " step is:" << step << endl;	

     thread_vector.push_back( std::thread{ &csr_treater::create_threaded_result, this, std::ref( multiplicant_vec ), 
					   i*step, finish, std::move( results_promise ) } ); 
   
     if( finish == (*this->input_array).size() -1 ) i = this->used_threads;// Break loop, if all elements have been read
  }

  for( std::thread &i : thread_vector ) i.join(); // All threads finish their work 

  for( std::future<array<T, ROWS>> &fut : results_futures ) { // adding all partial results 
	array<T, ROWS> partial_res( fut.get() ); 
	for( size_t j=0; j<(*result).size(); j++ ) (*result)[j] += partial_res[j];
  }
 
  cout << "CSR threads result is: ";  for( T &i : *result ) cout << " "<< i; cout << endl;
  return std::move( *result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
csr_treater<T, ROWS, COLUMNS>::csr_treater( const array<T, ROWS*COLUMNS> & input_array, size_t used_threads )
			     : abstract_format<T, ROWS, COLUMNS>( input_array, used_threads ) {

}


template<typename T, size_t ROWS, size_t COLUMNS>
csr_treater<T, ROWS, COLUMNS>::csr_treater( const array<T, ROWS*COLUMNS> & input_array ) 
			     : abstract_format<T, ROWS, COLUMNS>( input_array ) { 

}


template<typename T, size_t ROWS, size_t COLUMNS>
void csr_treater<T, ROWS, COLUMNS>::create_data_arrays() { 

   size_t non_zero_elem_of_row = 0;
   pointers[0] = 0;

   for( size_t i=0; i < ROWS; i++ ) 
     for( size_t j=0; j < COLUMNS; j++ ) {	

	if( (*this->input_array)[i*COLUMNS + j] != 0 ) {
	   indices.push_back( j );		
	   data.push_back( (*this->input_array)[i*COLUMNS+j] ); 		
	   non_zero_elem_of_row++;  
        }
 
	if( j == COLUMNS -1 ) {
	   pointers[ i+1 ] = non_zero_elem_of_row;
	   non_zero_elem_of_row = 0;
	}
    }

  for( size_t i=0; i<pointers.size()-1; i++ ) pointers[i+1] += pointers[i];
//  cout << "Pointers :"; for( size_t &i : pointers ) cout << i << " "; cout<<endl;
//  cout << "Indices :"; for( size_t &i : indices ) cout << i << " "; cout<<endl;
//  cout << "Data :"; for( T &i : data ) cout << i << " "; cout<<endl;
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
csr_treater<T, ROWS, COLUMNS>::normal_multiplication( const array<T, COLUMNS> & multiplicant_vec) {
 
  array<T, ROWS> *result = new array<T, ROWS>; // for( T &i : result ) i = 0; // Fill result array with 0

  for( size_t i=0; i<ROWS; i++ ) {

      size_t elem_start = pointers[i];     
      size_t elem_finish = pointers[i+1]; 
      T elem_result = 0;

      // This ugly function call calls the multiply_and_add function pointer that is inherited by
      // the abstract_format class. This function pointer points to the appropriate multiply function
      // depending on the T type.
      for( size_t j=elem_start; j<elem_finish; j++ ) 
        elem_result = (this->*(this->multiply_and_add))( data[ j ], multiplicant_vec[ indices[j] ], 
							 elem_result );	
 
      (*result)[i] = elem_result;
  }


  cout << setprecision(16);
  cout << "Normal CSR result: "; for( T &i : *result ) cout << i << " "; cout << endl;
  return std::move( *result );
}


#endif
