
#ifndef ____dia____
#define ____dia____

/// TO DO! Add destructors
#include "abstract_format.hpp"

using namespace std;

template<typename T> 
struct diagonal { 

    vector<T> diagonal_elements;
    int diagonal_number = 0;
    bool only_zeros = true;
};

template<typename T> 
struct diagonal_pair { 

    vector<pair<T, size_t>> diagonal_elements;
    int diagonal_number = 0;
};


template<typename T, size_t ROWS, size_t COLUMNS>
class diagonal_treater : public abstract_format<T, ROWS, COLUMNS> {

  vector<int> gpu_offset; 
  T* gpu_diagonals_data = nullptr; 
  uint gpu_data_columns = 0; 
  vector<diagonal<T>> diagonals; 
  void create_diagonals( array<T, ROWS*COLUMNS> & arg );
  int calculate_diagonal( size_t & elem_index );
  void discard_empty_diagonals( void );
  std::array<T, ROWS> && threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec );  
  std::array<T, ROWS> && normal_multiplication( const array<T, COLUMNS> & multiplicant_array );
  void create_data_arrays( void );
  void create_partial_result( const array<T, COLUMNS> &multiplicant_array, size_t start, size_t finish, 
			      std::promise<array<T, ROWS>> && promise );
  struct execute_info create_gpu_program();
  std::array<T, ROWS> && execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ); 
  void prepare_gpu_data( vector<int> & offset, T* & data, uint & data_columns );
public:
  diagonal_treater( const array<T, ROWS*COLUMNS> & input_array );
  diagonal_treater( const array<T, ROWS*COLUMNS> & input_array, size_t used_threads );
  ~diagonal_treater() {	
			delete [] gpu_diagonals_data;
			gpu_offset.clear();	
		      };	
};



template<typename T, size_t ROWS, size_t COLUMNS>
void diagonal_treater<T, ROWS, COLUMNS>::prepare_gpu_data( vector<int> & offset, T* & gpu_diagonals_data, uint & data_columns ) {
 
  // Scans the input matrix from all its diagonals. At the first appearance of a non-zero element, 
  // the diagonal is labeled non-empty
  for( int times=-int(ROWS)+1; times<int(COLUMNS); times++ ) { // covers all negative diagonals, and 0 diagonal 

     size_t i, j;
     if( times < 0 ) { i = -times; j = 0; }
     else { i = 0; j = times; }
	
     for( size_t k = j;  k < COLUMNS; k++ ) {
		
	if( i*COLUMNS + k >= ROWS*COLUMNS ) break;

 	if( this->input_array->data()[ i*COLUMNS + k ] != 0 ) { 
//		cout << "diagonal "<< int(times)  << " has non-zero elements"<<endl; 
		offset.push_back( times );
		break;
	} 
	i++;
     }
  }

  data_columns = uint( offset.size() );
  gpu_diagonals_data = new T[ ROWS*data_columns ]; // matrix is not initialized! Will contain garbage, but it's not a problem

  for( size_t k=0; k<size_t(data_columns); k++ ) { 

     size_t placement_start;
     size_t i, j;
     if( offset[k] < 0 ) { i = -offset[k]; j = 0; placement_start = -offset[k]*data_columns + k; }
     else { i = 0; j = offset[k]; placement_start = k; }
	
     for( size_t k = j;  k < COLUMNS; k++ ) {
		
	if( i*COLUMNS + k >= ROWS*COLUMNS ) break;

//	cout << "Index is ["<<i<<"]["<< k << "] " << i*COLUMNS + k << endl;	
	gpu_diagonals_data[ placement_start ] = this->input_array->data()[ i*COLUMNS + k ]; // assign values ONLY to cells of interest 
	placement_start += data_columns;
	i++;
     }
  }

}


template<typename T, size_t ROWS, size_t COLUMNS>
struct execute_info diagonal_treater<T, ROWS, COLUMNS>::create_gpu_program() {

    FILE *fp = fopen("./matrix_formats/dia/dia_kernel.cl", "r");
    if(!fp) { fprintf(stderr, "Failed to load kernel.\n");  exit(1); }

    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    if( !this->created_gpu_data ) { // creating the GPU data only once

       prepare_gpu_data( gpu_offset, gpu_diagonals_data, gpu_data_columns );

       this->created_gpu_data = true;
    }
 
    // STEP 3: Create a context
    cl_context context = clCreateContext( NULL, this->num_devices, &(this->devices[0]),NULL,NULL, &this->return_status);
 
    // STEP 4: Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, this->devices[0], 0, &this->return_status);

    // STEP 5: Create device buffers 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ROWS*gpu_data_columns*sizeof(T), NULL, &this->return_status);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, gpu_data_columns*sizeof(int), NULL, &this->return_status);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, COLUMNS*sizeof(T), NULL, &this->return_status);
    cl_mem d_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint), NULL, &this->return_status);
    cl_mem e_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint), NULL, &this->return_status);
    this->res_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ROWS*sizeof(T), NULL, &this->return_status);

    // STEP 6: Write host data to device buffers
    // Last 3 arguments declare events that have to be completed before this current event starts
    this->return_status = clEnqueueWriteBuffer( command_queue, a_mem_obj, CL_TRUE, 0, 
						ROWS*gpu_data_columns*sizeof(T), gpu_diagonals_data, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, b_mem_obj, CL_TRUE, 0, 
						gpu_data_columns*sizeof(int), gpu_offset.data(), 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, c_mem_obj, CL_TRUE, 0, 
						COLUMNS*sizeof(T), this->mult_vec, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, d_mem_obj, CL_TRUE, 0, 
						sizeof(uint), &gpu_data_columns, 0, NULL, NULL);
    uint mult_vec_cols( COLUMNS );
    this->return_status = clEnqueueWriteBuffer( command_queue, e_mem_obj, CL_TRUE, 0, 
						sizeof(uint), &mult_vec_cols, 0, NULL, NULL);

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
    this->return_status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&e_mem_obj);
    this->return_status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&this->res_mem_obj);

    struct execute_info ex_info( command_queue, kernel );

    this->return_status = clReleaseProgram( program );
    this->return_status = clReleaseContext( context );

    return ex_info;
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> &&
diagonal_treater<T, ROWS, COLUMNS>::execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ) { 
 
    // STEP 10: Configure the work-item structure
    size_t globalWorkSize[1];
//    size_t localWorkSize[1];
    globalWorkSize[0] = ROWS;  // There are #ROWS work-items
//    localWorkSize[0] = COLUMNS/4; //ROWS*COLUMNS/20;  // FIX THIS!!

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
    cout<<"DIA GPU result :"; for(size_t i = 0; i < ROWS; i++) cout << (*Result)[i] << " "; cout << endl;

    return std::move( *Result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
void diagonal_treater<T, ROWS, COLUMNS>::create_partial_result( const array<T, COLUMNS> &multiplicant_array, 
								size_t start, size_t finish, 
						       		std::promise<array<T, ROWS>> && promise ) {
 
  int diagonal_number;
  size_t position_placement;	
  array<T, ROWS> *partial_result = new array<T, ROWS>; for( size_t i=0; i<ROWS; i++ ) (*partial_result)[i] = T(0); 

  vector<diagonal_pair<T>> my_diagonals; 
  my_diagonals.reserve( ROWS + COLUMNS -1 );
  my_diagonals.resize( ROWS + COLUMNS -1, diagonal_pair<T>() ); // Fill the vector with structure diagonal

  for( size_t i= start; i<=finish; i++ ) 
    if( (*(this->input_array))[i] != 0 ) { // Storing only non-zero elements

	diagonal_number = calculate_diagonal(i);
	size_t diag_vect_index = diagonal_number + ROWS -1; 

	if( my_diagonals.at( diag_vect_index ).diagonal_number == 0 )  
			my_diagonals.at( diag_vect_index ).diagonal_number = diagonal_number;  

	if( diagonal_number >= 0 ) { 
//	   cout << " Index " << i << " goes to " << (i - diagonal_number)/divider << " of diag: " <<diagonal_number<<". ";
	   position_placement =  (i - diagonal_number)/COLUMNS;

	} else {
//	   cout << " Index " << i << " goes to " << (i - ((int) COLUMNS)*-diagonal_number)/divider 
//		<< " of diag: " << diagonal_number<< ". ";    
	   position_placement =  ( i - ((int) COLUMNS)*-diagonal_number )/COLUMNS;
	}
	
        my_diagonals.at( diag_vect_index ).diagonal_elements 
			  .push_back( std::pair<T, size_t>((*this->input_array)[i], position_placement) );  
    }
  
  for( auto diagonal_iter = my_diagonals.begin(); diagonal_iter != my_diagonals.end(); diagonal_iter++ ) 
    for( auto finger = (*diagonal_iter).diagonal_elements.begin(); 
		finger != (*diagonal_iter).diagonal_elements.end(); finger++ ) {

	size_t elem_pos = std::get<1>(*finger);

	if( (*diagonal_iter).diagonal_number < 0 ) { 

 	  (*partial_result)[ elem_pos - (*diagonal_iter).diagonal_number ] = 
				(this->*(this->multiply_and_add))( (std::get<0>(*finger)), multiplicant_array[ elem_pos ],
								   (*partial_result)[ elem_pos - (*diagonal_iter).diagonal_number ]); 
	} else { 	
 
 	  (*partial_result)[ elem_pos ] = 
		(this->*(this->multiply_and_add))( (std::get<0>(*finger)), 
						   multiplicant_array[ elem_pos + (*diagonal_iter).diagonal_number ],
						   (*partial_result)[ elem_pos ] ); 
	}
    }

  promise.set_value( *partial_result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
diagonal_treater<T, ROWS, COLUMNS>::threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec ) { 

  vector<std::thread> thread_vector;     
  vector<std::future<array<T, ROWS>>> futures;
  array<T, ROWS> *result = new array<T, ROWS>; for( size_t i=0; i<ROWS; i++ ) (*result)[i] = T(0);

  size_t step = this->calculate_step();

  for( size_t i=0; i<size_t(this->used_threads); i++ ) {  
  
    size_t finish = ( (i+1)*step < (*this->input_array).size() ) ? (i+1)*step -1 : (*this->input_array).size() -1;
    // If some elements are remaining, they are all placed in the last thread's work
    if( i == size_t(this->used_threads -1) && finish < (*this->input_array).size() -1 ) finish = (*this->input_array).size() -1;
 
//    cout << i*step << " " << finish << " step is:" << step << endl;	
   
    size_t start = i*step;	
    std::promise<array<T, ROWS>> promise;
    futures.push_back( promise.get_future() );

    thread_vector.push_back( std::thread{ &diagonal_treater::create_partial_result, this, 
					   std::ref(multiplicant_vec), start, finish
					 , std::move( promise ) } ); // move the promise into the thread function

    if( finish == (*this->input_array).size() -1 ) i = this->used_threads;// Break loop, if all elements have been reached
  }

  for( std::thread &i : thread_vector ) i.join(); // range_based loop 

  for( std::future<array<T, ROWS>> &fut : futures ) { 
	array<T, ROWS> partial_res( fut.get() ); 
	for( size_t j=0; j<(*result).size(); j++ ) (*result)[j] += partial_res[j];
	partial_res.~array<T, ROWS>(); // cleanup results from threads
  }

  cout<<"Threaded dia result : ";  for( size_t i=0; i< (*result).size(); i++ ) cout << (*result)[i]<<" "; cout << endl; 

  return std::move( *result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
array<T, ROWS> && 
diagonal_treater<T, ROWS, COLUMNS>::normal_multiplication( const array<T, COLUMNS> & multiplicant_array ) {

  array<T, ROWS> *result = new array<T, ROWS>(); for( size_t i=0; i<ROWS; i++ ) (*result)[i] = 0;
  
//  array<T, ROWS> result; for( size_t i=0; i<ROWS; i++ ) result[i] = 0;  // = new array<T, ROWS>(); 

  for( auto i = diagonals.begin(); i != diagonals.end(); i++ ) { 

     auto multiplicant_start = multiplicant_array.begin();	
     auto result_store_start = (*result).begin();	
 
     if( (*i).diagonal_number < 0 ) { result_store_start -= (*i).diagonal_number; }
     else { multiplicant_start += (*i).diagonal_number; }
	
     for( auto it = (*i).diagonal_elements.begin(); it != (*i).diagonal_elements.end(); it++ ) {

	*result_store_start = (this->*(this->multiply_and_add))( *multiplicant_start, *it, 
							 	 *result_store_start );	
	result_store_start++;
	multiplicant_start++;
     }	
  }

  cout << "Normal dia result : "; 
  for( size_t i=0; i< (*result).size(); i++ ) cout << (*result)[i] << " "; cout << endl; 

  return std::move(*result);
}


template<typename T, size_t ROWS, size_t COLUMNS>
void diagonal_treater<T, ROWS, COLUMNS>::discard_empty_diagonals( void ) {

  auto it = diagonals.begin();
  while( it != diagonals.end() )  
     if( (*it).only_zeros ) it = diagonals.erase( it ); // After erase, it is the next element in the container
     else it++;   
}


template<typename T, size_t ROWS, size_t COLUMNS>
void diagonal_treater<T, ROWS, COLUMNS>::create_data_arrays( void ) {

  diagonals.reserve( ROWS + COLUMNS -1 );
  diagonals.resize( ROWS + COLUMNS -1, diagonal<T>() ); // Fill the vector with structure diagonal

  create_diagonals( *this->input_array );
  discard_empty_diagonals();
}


template<typename T, size_t ROWS, size_t COLUMNS>
diagonal_treater<T, ROWS, COLUMNS>::diagonal_treater( const array<T, ROWS*COLUMNS> & input_array ) 
				  : abstract_format<T, ROWS, COLUMNS>( input_array ) {

}


template<typename T, size_t ROWS, size_t COLUMNS>
diagonal_treater<T, ROWS, COLUMNS>::diagonal_treater( const array<T, ROWS*COLUMNS> & input_array, size_t used_threads )
				  : abstract_format<T, ROWS, COLUMNS>( input_array, used_threads ) {

}


template<typename T, size_t ROWS, size_t COLUMNS>
void diagonal_treater<T, ROWS, COLUMNS>::create_diagonals( array<T, ROWS*COLUMNS> & input_array ) {

    int diagonal_number;

    for( size_t i = 0; i<input_array.size(); i++ ) {

	// moves the value arg.at(i)
	diagonal_number = calculate_diagonal(i);
	size_t diag_vect_index;

	diag_vect_index = diagonal_number + ROWS -1; 
	
	if( diagonals.at( diag_vect_index ).only_zeros == true && input_array.at(i) != 0 ) 
		diagonals.at( diag_vect_index ).only_zeros = false;
	if( diagonals.at( diag_vect_index ).diagonal_number == 0 )  
		diagonals.at( diag_vect_index ).diagonal_number = diagonal_number;  

        diagonals.at( diag_vect_index ).diagonal_elements.push_back( input_array.at(i) );
    }
}


template<typename T, size_t ROWS, size_t COLUMNS>
int diagonal_treater<T, ROWS, COLUMNS>::calculate_diagonal( size_t & elem_index ) {

    // element lies at A[elem_row][elem_column], and elem_index = array_dimension * elem_row + elem_column
    size_t elem_column = elem_index % COLUMNS;   
    size_t elem_row = (elem_index - elem_column)/COLUMNS;

    return int(elem_column) - int(elem_row); 
}

#endif

