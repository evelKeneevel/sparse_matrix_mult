#ifndef ____coo____
#define ____coo____

using namespace std;

const size_t COO_LOCAL_WORK_SIZE = 1;

template<typename T, size_t ROWS, size_t COLUMNS>
class coo_treater : public abstract_format<T, ROWS, COLUMNS> {

  int nonzero_elements;
  uint* elem_per_row = nullptr;
  uint* gpu_row_indexes = nullptr;
  uint* gpu_column_indexes = nullptr;
  T* gpu_data = nullptr; 
  cl_mem vals_mem_obj;
  vector<T> data;
  vector<size_t> row_indexes, column_indexes;
  // Function overloading doesn't work when running the functions as threads. If two versions of F exist, 
  // when run as threads, the compiler can't deduct from their arguments which one is being called
  void create_data_arrays( void );
  std::array<T, ROWS> && normal_multiplication( const array<T, COLUMNS> & multiplicant_vec );
  std::array<T, ROWS> && threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec );  
  void calculate_threaded_result( size_t start_index, size_t end_index, 
				  const array<T, COLUMNS> & multiplicant_vec, std::promise<array<T, ROWS>> && promise ); 

  struct execute_info create_gpu_program();
  std::array<T, ROWS> && execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel );  
  void create_gpu_data( T* gpu_data, uint* gpu_row_indexes, uint* gpu_column_indexes, uint* elem_per_row ); 
public:
  coo_treater( const array<T, ROWS*COLUMNS> & input_array );
  coo_treater( const array<T, ROWS*COLUMNS> & input_array, size_t used_threads );
  ~coo_treater(){  // delete all allocated matrixes 
		   if( elem_per_row != nullptr ) { delete [] elem_per_row; elem_per_row = nullptr; } 
		   if( gpu_row_indexes != nullptr ) { delete [] gpu_row_indexes; gpu_row_indexes = nullptr; }
		   if( gpu_column_indexes != nullptr ) { delete [] gpu_column_indexes; gpu_column_indexes = nullptr; }
		   if( gpu_data != nullptr ) { delete [] gpu_data; gpu_data = nullptr; }
		};
};

/*
template<typename T, size_t ROWS, size_t COLUMNS>
void calculate_stuff( T* data, int * row_indexes, int* column_indexes, T* multiplicant_vec, int total_elements ) {
 
  //array<T, ROWS> *result = new array<T, ROWS>; // for( T &i : result ) i = 0; // Fill result array with 0
//  result = T[ROWS]; // No initialization needed this time 
  T result[ROWS];
  for( size_t i=0; i<total_elements; i++ ) { cout<< row_indexes[i] <<" "; } cout << endl; 
  for( size_t i=0; i<total_elements; i++ ) { cout<< column_indexes[i] <<" "; } cout<< endl;
 
  for( int i=0; i<ROWS; i++ ) {
  
    int j=0;
    T part_res = 0;
    T vals[32];
    


       cout << "Elem at ["<<row_indexes[j]<<"]["<<column_indexes[j]<<"]"<< " j is "<< j <<endl;  
       part_res += data[j]*multiplicant_vec[ column_indexes[j] ];   
       j++;	

    cout << "I is "<< i << endl;
    result[i] = part_res;

  }
  
  for( size_t i=0; i<ROWS; i++  ) cout << result[i]<< " "; cout<<endl;

}
*/


template<typename T, size_t ROWS, size_t COLUMNS>
void coo_treater<T, ROWS, COLUMNS>::create_gpu_data( T* gpu_data, uint* gpu_row_indexes, 
						     uint* gpu_column_indexes,
						     uint* elem_per_row ) {
  this->nonzero_elements = 0;

  for( int i=0; i < int(ROWS); i++ ) 
    for( int j=0; j < int(COLUMNS); j++ ) { 
      if( (*this->input_array)[i*COLUMNS + j] != 0 ) { 

	 gpu_data[ nonzero_elements ] = (*this->input_array)[i*COLUMNS + j];
	 gpu_row_indexes[ nonzero_elements ] = i;
	 gpu_column_indexes[ nonzero_elements ] = j;
	 nonzero_elements++;
      }
    }

//  for( size_t i=0; i<nonzero_elements; i++ ) cout << gpu_data[i] << " "; cout <<endl;
//  for( size_t i=0; i<ROWS; i++ ) cout << elem_per_row[i] << " "; cout <<endl;
//  for( size_t i=0; i< nonzero_elements; i++ ) cout << row_indexes[i]<<" "; cout << endl;
}


template<typename T, size_t ROWS, size_t COLUMNS>
struct execute_info coo_treater<T, ROWS, COLUMNS>::create_gpu_program() {

    FILE *fp = fopen("./matrix_formats/coo/coo_kernel.cl", "r");
    if(!fp) { fprintf(stderr, "Failed to load kernel.\n");  exit(1); }

    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
  
    if( !this->created_gpu_data ) {

       gpu_data = new T[ROWS*COLUMNS];  gpu_row_indexes = new uint[ROWS*COLUMNS]; 
       gpu_column_indexes = new uint[ROWS*COLUMNS];  elem_per_row = new uint[ROWS];

       create_gpu_data( gpu_data, gpu_row_indexes, gpu_column_indexes, elem_per_row );
       this-> created_gpu_data = true;	 
    }
 
    // STEP 3: Create a context
    cl_context context = clCreateContext( NULL, this->num_devices, &(this->devices[0]),NULL,NULL, &this->return_status);
 
    // STEP 4: Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, this->devices[0], 0, &this->return_status);

    // STEP 5: Create device buffers 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, nonzero_elements*sizeof(T), NULL, &this->return_status);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, nonzero_elements*sizeof(uint), NULL, &this->return_status);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, nonzero_elements*sizeof(uint), NULL, &this->return_status);
    cl_mem d_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, COLUMNS*sizeof(T), NULL, &this->return_status);
    cl_mem e_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint), NULL, &this->return_status);
    this->vals_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nonzero_elements*sizeof(T), NULL, &this->return_status);
//    this->res_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ROWS*sizeof(T), NULL, &this->return_status);

    // STEP 6: Write host data to device buffers
    // Last 3 arguments declare events that have to be completed before this current event starts
    this->return_status = clEnqueueWriteBuffer( command_queue, a_mem_obj, CL_TRUE, 0, 
						nonzero_elements*sizeof(T), gpu_data, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, b_mem_obj, CL_TRUE, 0, 
						nonzero_elements*sizeof(uint), gpu_row_indexes, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, c_mem_obj, CL_TRUE, 0, 
						nonzero_elements*sizeof(uint), gpu_column_indexes, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, d_mem_obj, CL_TRUE, 0, 
						COLUMNS*sizeof(T) , this->mult_vec, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, e_mem_obj, CL_TRUE, 0, 
						sizeof(uint) , &this->nonzero_elements, 0, NULL, NULL);
//    this->return_status = clEnqueueWriteBuffer( command_queue, this->vals_mem_obj, CL_TRUE, 0, 
//						nonzero_elements*sizeof(T), vals, 0, NULL, NULL);
 
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
    this->return_status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&this->vals_mem_obj);
//    this->return_status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&this->res_mem_obj);

    struct execute_info ex_info( command_queue, kernel );

    this->return_status = clReleaseProgram( program );
    this->return_status = clReleaseContext( context );

    return ex_info;
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
coo_treater<T, ROWS, COLUMNS>::execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ) {


    // STEP 10: Configure the work-item structure
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    globalWorkSize[0] = ROWS*COO_LOCAL_WORK_SIZE;  // There are #ROWS work-items
    localWorkSize[0] = COO_LOCAL_WORK_SIZE; //ROWS*COLUMNS/20;  // FIX THIS!!

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
    std::array<T, ROWS> *Result = new std::array<T, ROWS>;  for( size_t i = 0; i < ROWS; i++ ) (*Result)[i] = 0;
    T* vals = new T[nonzero_elements];

    // Arg #2 is buffer, refers to a valid buffer object.
    // The data is read and copied to C. CL_TRUE makes clEnqueueReadBuffer to not return until the buffer data 
    // has been read and copied into Result
    // Copies c_mem_obj to Result
//    this->return_status = clEnqueueReadBuffer( command_queue, this->res_mem_obj, CL_TRUE, 0, 
//				               ROWS*sizeof(T), (void *)(*Result).data(), 0, NULL, NULL );

    this->return_status = clEnqueueReadBuffer( command_queue, this->vals_mem_obj, CL_TRUE, 0, 
				               nonzero_elements*sizeof(T), (void *) vals, 0, NULL, NULL );
   
    this->return_status = clReleaseKernel(kernel);
    this->return_status = clReleaseCommandQueue(command_queue);


//    for( size_t i=1; i<ROWS; i++ ) elem_per_row[i] += elem_per_row[i-1];
 
    for( size_t i = 0; i<size_t(nonzero_elements); i++ )
	(*Result)[ gpu_row_indexes[i] ] += vals[i]; 

//    cout<<"vals result :"; for( size_t i=0; i<this->nonzero_elements; i++ ) cout << vals[i] << " ";  cout << endl;

    // Display the result to the screen
    cout<<"COO GPU result :"; for( T &i : *Result ) cout << i << " ";  cout << endl;
    delete [] vals;

    return std::move( *Result );

}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
coo_treater<T, ROWS, COLUMNS>::threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec ) {

  vector<std::thread> thread_vector;     
  vector<std::future<array<T, ROWS>>> futures;
  array<T, ROWS>* result = new array<T, ROWS>; for( size_t i=0; i<(*result).size(); i++ ) (*result)[i] = 0; 

  size_t step = this->calculate_step();
 
  for( size_t i=0; i<size_t(this->used_threads); i++ ) {  

     std::promise<array<T, ROWS>> promise;
     futures.push_back( promise.get_future() );

     size_t finish = ( (i+1)*step < (*this->input_array).size() ) ? (i+1)*step -1 : (*this->input_array).size() -1;
     // If some elements are remaining, they are all placed in the last thread's work
     if( i ==size_t(this->used_threads -1) && finish < (*this->input_array).size() -1 ) finish = (*this->input_array).size() -1;
 
//     cout << i*step << " " << finish << " step is:" << step << endl;	
   
     thread_vector.push_back( std::thread{ &coo_treater::calculate_threaded_result, this, i*step, finish, 
					   std::ref( multiplicant_vec ), std::move(promise) } );

     if( finish == (*this->input_array).size() -1 ) i = this->used_threads; // Break loop, if all elements are read
  }
 
  for( std::thread &i : thread_vector ) i.join(); // range_based loop 

  for( std::future<array<T, ROWS>> &fut : futures ) { 
	array<T, ROWS> partial_res( fut.get() ); 
	for( size_t j=0; j<(*result).size(); j++ ) (*result)[j] += partial_res[j];	
  }

  cout<<"Threaded coo result : ";  for( size_t i=0; i< (*result).size(); i++ ) cout << (*result)[i] << " "; cout << endl; 
  thread_vector.clear();  futures.clear();
  return std::move( *result );
} 


template<typename T, size_t ROWS, size_t COLUMNS>
void coo_treater<T, ROWS, COLUMNS>::calculate_threaded_result( size_t start_index, size_t end_index, 
							       const array<T, COLUMNS> & multiplicant_vec, 
							       std::promise<array<T, ROWS>> && promise ) { 

  std::array<T, ROWS> partial_result; for( size_t i = 0; i< partial_result.size(); i++ ) partial_result[i] = T(0);
  vector<T> partial_data;
  vector<size_t> row_indexes, column_indexes;
 
  size_t j = start_index%COLUMNS;
  size_t i = (start_index - j)/COLUMNS;

  for( size_t times=start_index; times <= end_index; times++ ) { 

	if( (*this->input_array)[i*COLUMNS+j] != 0 ) {
	   column_indexes.push_back( j );	
	   row_indexes.push_back( i );	
	   partial_data.push_back( (*this->input_array)[i*COLUMNS+j] ); 		
        }
 
	if( j == COLUMNS -1 ) { // reached end of row
	   j = 0; i++;

	} else j++; // continue row traversal
  }

  for( size_t i = 0; i < partial_data.size(); i++ )  
    partial_result[ row_indexes[i] ] = (this->*(this->multiply_and_add))( multiplicant_vec[ column_indexes[i] ],
					 			          partial_data[i], 
								          partial_result[ row_indexes[i] ] );
	 
  promise.set_value( partial_result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
void coo_treater<T, ROWS, COLUMNS>::create_data_arrays( void ) {

  for( size_t i = 0; i < ROWS; i++ )  //  (*this->input_array).size(); i++ ) 
    for( size_t j = 0; j < COLUMNS; j++ ) 	
      if( (*this->input_array)[i*COLUMNS+j] != 0 ) {

	  data.push_back( (*this->input_array)[i*COLUMNS+j] );
	  column_indexes.push_back( j );
  	  row_indexes.push_back( i );
      }
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
coo_treater<T, ROWS, COLUMNS>::normal_multiplication( const array<T, COLUMNS> & multiplicant_vec ) { 

  array<T, ROWS> *result = new array<T, ROWS>; for( size_t i = 0; i < (*result).size(); i++ ) (*result)[i] = T(0);

  for( size_t i = 0; i < data.size(); i++ ) 
        (*result)[ row_indexes[i] ] = (this->*(this->multiply_and_add))( multiplicant_vec[ column_indexes[i] ],
					 			         data[i], 
								         (*result)[ row_indexes[i] ] );	
  
  cout<<"Normal coo result : ";  for( size_t i =0; i< (*result).size(); i++ ) cout << (*result)[i] << " ";  cout<<endl;

  return std::move( *result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
coo_treater<T, ROWS, COLUMNS>::coo_treater( const array<T, ROWS*COLUMNS> & input_array )
			     : abstract_format<T, ROWS, COLUMNS>( input_array ) { 
  // calls the parent constructor
}


template<typename T, size_t ROWS, size_t COLUMNS>
coo_treater<T, ROWS, COLUMNS>::coo_treater( const array<T, ROWS*COLUMNS> & input_array, size_t used_threads ) 
			     : abstract_format<T, ROWS, COLUMNS>( input_array, used_threads ) { 
  // calls the parent constructor
}


#endif
