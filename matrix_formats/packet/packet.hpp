#ifndef ____packet____
#define ____packet____

#include "../coo/coo.hpp"
//#include "dia.hpp"

const size_t LOCAL_MEM_SIZE_THRESHOLD = 4000;

using namespace std;

template<typename T, size_t ROWS, size_t COLUMNS>
class packet_treater : public abstract_format<T, ROWS, COLUMNS>{

  T* P_data_gpu = nullptr; 
  uint* P_indexes_gpu = nullptr;
  T* P_data = nullptr;
  size_t* P_indexes = nullptr;
  size_t P1_ROWS, P1_COLUMNS; 
  array<T, ROWS*COLUMNS> *excluded_elements_array = nullptr;
  const size_t ADDED_PACKET_SIZES = ROWS*COLUMNS/2;
  void create_data_arrays(); 
  void create_gpu_data( T* & P_data, uint* & P_indexes );

  struct execute_info create_gpu_program();
  std::array<T, ROWS> && execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ); 

  std::array<T, ROWS> && normal_multiplication( const array<T, COLUMNS> & multiplicant_vec );	 
  std::array<T, ROWS> && threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec );
  void calculate_threaded_result( size_t start_index, size_t end_index, size_t thread_index,
				  const array<T, COLUMNS> & multiplicant_vec, std::promise<array<T, ROWS>> && promise ); 
public:
  packet_treater( const array<T, ROWS*COLUMNS> & input_array, int used_threads );
  packet_treater( const array<T, ROWS*COLUMNS> & input_array );
  ~packet_treater(){ 
		     if( excluded_elements_array != nullptr ) delete [] excluded_elements_array;  
		     if( P_data != nullptr ) delete [] P_data;
		     if( P_indexes != nullptr ) delete [] P_indexes;    		
		     if( P_data_gpu != nullptr ) delete [] P_data_gpu; 
		     if( P_indexes_gpu != nullptr ) delete [] P_indexes_gpu;
 		     this->input_array = nullptr;  
		   };
};


/*
template<typename T, size_t ROWS, size_t COLUMNS>
void calculate_stuff( const T* data, const int* indexes, size_t P1_ROWS, size_t P1_COLUMNS, int* mult_vec ) {

  T result[ROWS];

  for( int i=0; i<ROWS; i++ ) {

     T part_res = 0;
     int start, finish; 
     int local_mult_vec[P1_COLUMNS];
     int columns = COLUMNS;
     int offset = 0;	

     if( i < P1_ROWS ) { 

	offset = 0;
	start = 0; finish = P1_COLUMNS; 	
	for( int j = start; j<finish; j++ ) { local_mult_vec[j] = mult_vec[j]; cout << local_mult_vec[j] <<" "; }
	cout << endl;
     } else { 

	offset = P1_COLUMNS;	
	start = P1_COLUMNS; finish = columns;
	int k = 0; 
	for( int j = start; j<finish; j++ ) { local_mult_vec[k] = mult_vec[j]; cout << local_mult_vec[k] <<" "; k++;  }
	cout << endl;
     }
 
     for( int j = start; j < finish; j++ ) { 
		
	if( data[i*columns+j] != 0 ) {	
	   part_res += data[i*columns+j]*local_mult_vec[ indexes[i*columns+j] ];   // mult_vec[ indexes[i*columns+j] ]; 
	   cout <<  data[i*columns+j] << " * "<< local_mult_vec[ indexes[i*columns+j] - offset ] << " using index value " << indexes[i*columns+j] - offset <<endl;
	}

     }
	
     //for( int i=0; i<LOCAL_WORK_SIZE; i++ ) part_res += vals[i];
     result[i] = part_res;
  }


  for( size_t i=0; i<ROWS; i++ ) cout << result[i] << " "; cout << endl;
}
*/


template<typename T, size_t ROWS, size_t COLUMNS>
void packet_treater<T, ROWS, COLUMNS>::create_gpu_data( T* & P_data_gpu, uint* & P_indexes_gpu ) { 

  P_data_gpu = new T[ ROWS*COLUMNS ];
  P_indexes_gpu = new uint[ ROWS*COLUMNS ]; 

  size_t p_col = 0;  bool filled_row = false;
//  cout << setprecision(16);
 
  for( size_t i = 0; i < P1_ROWS; i++ ) {

    p_col = 0; filled_row = false;
    for( size_t j = 0; j < COLUMNS; j++ ) {

        (*excluded_elements_array)[i*COLUMNS+j] = (*this->input_array)[i*COLUMNS+j];

	if( (*this->input_array)[i*COLUMNS+j] != 0 && filled_row == false ) {

	   P_data_gpu[ i*COLUMNS + p_col ] = (*this->input_array)[i*COLUMNS+j];
	   P_indexes_gpu[ i*COLUMNS + p_col ] = j;
           (*excluded_elements_array)[i*COLUMNS+j] = 0;

           if( p_col == P1_COLUMNS -1 ) { filled_row = true; p_col++; } 
	   else p_col++;	
	}

	if( j == COLUMNS-1 ) 
	   for( size_t l = p_col; l < COLUMNS; l++ ) P_data_gpu[ i*COLUMNS + l ] = T(0);        
    }
  }

 
  for( size_t i = P1_ROWS; i < ROWS; i++ ) { // Treating the other half of the input array 

    p_col = P1_COLUMNS; filled_row = false;
    for( size_t j = P1_COLUMNS; j < COLUMNS; j++ ) {

        (*excluded_elements_array)[i*COLUMNS+j] = (*this->input_array)[i*COLUMNS+j];

	if( (*this->input_array)[i*COLUMNS+j] != 0 && filled_row == false ) {

	   P_data_gpu[ i*COLUMNS + p_col ] = (*this->input_array)[i*COLUMNS+j];	
	   P_indexes_gpu[ i*COLUMNS + p_col ] = j;
           (*excluded_elements_array)[i*COLUMNS+j] = 0;

           if( p_col == COLUMNS -1 ) { filled_row = true; } 
	   else p_col++;	
	}

	if( j == COLUMNS-1 ) { 

	   if( filled_row == true ) { // Packet cannot sustaiin more elements

	      for( size_t l = 0; l < P1_COLUMNS; l++ ) { 

		 P_data_gpu[ i*COLUMNS + l ] = 0;  
		 (*excluded_elements_array)[ i*COLUMNS + l ] = (*this->input_array)[i*COLUMNS+l];
	      }	
	   } else {
	      // Scanning the row, and the elements of the row that are residing left of bottom packet
	      // The bottom packet is not full yet, so we try to fill it  	
    	      for( size_t k = 0; k < P1_COLUMNS; k++ ) {

        	(*excluded_elements_array)[i*COLUMNS+k] = (*this->input_array)[i*COLUMNS+k];
		P_data_gpu[i*COLUMNS+k] = 0;

		if( (*this->input_array)[i*COLUMNS+k] != 0 && filled_row == false ) {

	   		P_data_gpu[ i*COLUMNS + p_col ] = (*this->input_array)[i*COLUMNS+k];	
	   		P_indexes_gpu[ i*COLUMNS + p_col ] = k;
           		(*excluded_elements_array)[i*COLUMNS+k] = 0;

           		if( p_col == COLUMNS -1 ) { filled_row = true; p_col++; } 
	   		else p_col++;	
		}
	      }
	      // If there's any elements not initialized at the packet	
	      for( size_t l = p_col; l < COLUMNS; l++ ) P_data_gpu[ i*COLUMNS + l ] = T(0);
	      	
	   }
	} // end of j == COLUMNS -1
    }
  }

/*
  cout << endl;   
   for( size_t i=0; i<ROWS; i++ ) {
      for( size_t j=0; j<COLUMNS; j++ ) { 
	 cout << P_data_gpu[ i*COLUMNS + j ] << " ";
      }	
      cout << "    ";	
      for( size_t j=0; j<COLUMNS; j++ ) { 
	 cout << P_indexes_gpu[ i*COLUMNS + j ] << " ";
      }     	
      cout << endl;	
   }

   cout << "Excluded elements"<< endl;	
   for( size_t i=0; i<ROWS; i++ ) {
      for( size_t j=0; j<COLUMNS; j++ ) { 
	 cout << (*excluded_elements_array)[ i*COLUMNS + j ] << " ";
      }	
      cout << endl;	
   }
*/

}


template<typename T, size_t ROWS, size_t COLUMNS>
struct execute_info 
packet_treater<T, ROWS, COLUMNS>::create_gpu_program() {

    FILE *fp = fopen("./matrix_formats/packet/packet_kernel.cl", "r");
    if(!fp) { fprintf(stderr, "Failed to load kernel.\n");  exit(1); }

    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    if( !this->created_gpu_data ) { // creating the GPU data only once

       create_gpu_data( P_data_gpu, P_indexes_gpu );

       this->created_gpu_data = true;
    }
 
//    calculate_stuff<T, ROWS, COLUMNS>( P_data_gpu, P_indexes_gpu, P1_ROWS, P1_COLUMNS, this->mult_vec );
 
    // STEP 3: Create a context
    cl_context context = clCreateContext( NULL, this->num_devices, &(this->devices[0]),NULL,NULL, &this->return_status);
 
    // STEP 4: Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, this->devices[0], 0, &this->return_status);

    // STEP 5: Create device buffers 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ROWS*COLUMNS*sizeof(T), NULL, &this->return_status);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, ROWS*COLUMNS*sizeof(uint), NULL, &this->return_status); 
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint), NULL, &this->return_status);
    cl_mem d_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint), NULL, &this->return_status);
    cl_mem e_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint), NULL, &this->return_status);
    cl_mem f_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, COLUMNS*sizeof(T), NULL, &this->return_status);
    this->res_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ROWS*sizeof(T), NULL, &this->return_status);

    // STEP 6: Write host data to device buffers
    // Last 3 arguments declare events that have to be completed before this current event starts
    // !!! Taking into consideration only the first nonzero_elements of my_data, my_indices !!! 
    this->return_status = clEnqueueWriteBuffer( command_queue, a_mem_obj, CL_TRUE, 0, 
						ROWS*COLUMNS*sizeof(T), (void*) P_data_gpu, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, b_mem_obj, CL_TRUE, 0, 
						ROWS*COLUMNS*sizeof(uint), (void*) P_indexes_gpu, 0, NULL, NULL);

    uint columns(COLUMNS); uint p1_rows(P1_ROWS); uint p1_columns(P1_COLUMNS); 	

    this->return_status = clEnqueueWriteBuffer( command_queue, c_mem_obj, CL_TRUE, 0, 
						sizeof(uint), (void*) &p1_rows, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, d_mem_obj, CL_TRUE, 0, 
						sizeof(uint), (void*) &p1_columns, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, e_mem_obj, CL_TRUE, 0, 
						sizeof(uint), (void*) &columns, 0, NULL, NULL);
    this->return_status = clEnqueueWriteBuffer( command_queue, f_mem_obj, CL_TRUE, 0, 
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
    this->return_status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&e_mem_obj);
    this->return_status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&f_mem_obj);

    // Multiplicant vec can be stored in local GPU memory, if its size is smaller than LOCAL_MEM_SIZE_THRESHOLD
    // Otherwise, local_size will be 1. The kernel code detects this, and uses the local or global memory dep. on size 		
    size_t local_size = ( COLUMNS <= LOCAL_MEM_SIZE_THRESHOLD ) ? COLUMNS : 1;  
    this->return_status = clSetKernelArg(kernel, 6, local_size*sizeof(T), 0 ); // __local array passed to kernel 
    this->return_status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&this->res_mem_obj);

    struct execute_info ex_info( command_queue, kernel );

    this->return_status = clReleaseProgram( program );
    this->return_status = clReleaseContext( context );

    return ex_info;
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && 
packet_treater<T, ROWS, COLUMNS>::execute_gpu_program( cl_command_queue & command_queue, cl_kernel & kernel ) { 


    // STEP 10: Configure the work-item structure
    size_t globalWorkSize[1];
//    size_t localWorkSize[1];
    globalWorkSize[0] = ROWS; // LOCAL_WORK_SIZE*ROWS;  // There are #ROWS work-items
//    localWorkSize[0] = LOCAL_WORK_SIZE; //  COLUMNS/4; //ROWS*COLUMNS/20;  // FIX THIS!!

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

    // Calling the threaded COO version to do the coo calculation
    array<T, ROWS> coo_result( (coo_treater<T, ROWS, COLUMNS>( *excluded_elements_array, 4 ))
				.multiply_with_vector( *this->mult_vec_ptr ) );

    for( size_t i=0; i<ROWS; i++ ) (*Result)[i] += coo_result[i];

    // Display the result to the screen
    cout<<"Packet GPU result :"; for( T &i : *Result ) cout << i << " ";  cout << endl;

    return std::move( *Result );

}


template<typename T, size_t ROWS, size_t COLUMNS>
void packet_treater<T, ROWS, COLUMNS>::create_data_arrays() { 
 
  P_data = new T[ ROWS*COLUMNS ];
  P_indexes = new size_t[ ROWS*COLUMNS ]; 

  size_t p_col = 0;  bool filled_row = false;
 
  for( size_t i = 0; i < P1_ROWS; i++ ) {

    p_col = 0; filled_row = false;
    for( size_t j = 0; j < COLUMNS; j++ ) {

        (*excluded_elements_array)[i*COLUMNS+j] = (*this->input_array)[i*COLUMNS+j];

	if( (*this->input_array)[i*COLUMNS+j] != 0 && filled_row == false ) {

	   P_data[ i*COLUMNS + p_col ] = (*this->input_array)[i*COLUMNS+j];	
	   P_indexes[ i*COLUMNS + p_col ] = j;
           (*excluded_elements_array)[i*COLUMNS+j] = 0;

           if( p_col == P1_COLUMNS -1 ) { filled_row = true; p_col++; } 
	   else p_col++;	

//	   cout << "adding  ["<<i<<"]["<<j<<"] with p_col "<<p_col<<", filled row? "<< filled_row<<endl;
	}

	if( j == COLUMNS-1 ) { 
//	   cout << "Finished on ["<<i<<"]["<<j<<"] with p_col "<<p_col<<endl;
	   for( size_t l = p_col; l < COLUMNS; l++ ) P_data[ i*COLUMNS + l ] = 0;
	}

    }
  }

  
  for( size_t i = P1_ROWS; i < ROWS; i++ ) { // Treating the other half of the input array 

    p_col = P1_COLUMNS; filled_row = false;
    for( size_t j = P1_COLUMNS; j < COLUMNS; j++ ) {

        (*excluded_elements_array)[i*COLUMNS+j] = (*this->input_array)[i*COLUMNS+j];

	if( (*this->input_array)[i*COLUMNS+j] != 0 && filled_row == false ) {

	   P_data[ i*COLUMNS + p_col ] = (*this->input_array)[i*COLUMNS+j];	
	   P_indexes[ i*COLUMNS + p_col ] = j;
           (*excluded_elements_array)[i*COLUMNS+j] = 0;

           if( p_col == COLUMNS -1 ) { filled_row = true; } 
	   else p_col++;	

//	   cout << "adding  ["<<i<<"]["<<j<<"] with p_col "<<p_col<<", filled row? "<< filled_row<<endl;
	}

	if( j == COLUMNS-1 ) { 

	   if( filled_row == true ) {

	      for( size_t l = 0; l < P1_COLUMNS; l++ ) { 

		 P_data[ i*COLUMNS + l ] = 0;  
		 (*excluded_elements_array)[ i*COLUMNS + l ] = (*this->input_array)[i*COLUMNS+l];
	      }	
	   } else {
	      // Scanning the row, and the elements of the row that are residing left of bottom packet
	      // The bottom packet is not full yet, so we try to fill it  		
    	      for( size_t k = 0; k < P1_COLUMNS; k++ ) {

        	(*excluded_elements_array)[i*COLUMNS+k] = (*this->input_array)[i*COLUMNS+k];
		P_data[i*COLUMNS+k] = 0;

		if( (*this->input_array)[i*COLUMNS+k] != 0 && filled_row == false ) {

	   		P_data[ i*COLUMNS + p_col ] = (*this->input_array)[i*COLUMNS+k];	
	   		P_indexes[ i*COLUMNS + p_col ] = k;
           		(*excluded_elements_array)[i*COLUMNS+k] = 0;

           		if( p_col == COLUMNS -1 ) { filled_row = true; p_col++; } 
	   		else p_col++;	
		}
	      }
	      // If there's any elements not initialized at the packet	
	      for( size_t l = p_col; l < COLUMNS; l++ ) P_data[ i*COLUMNS + l ] = T(0);
	   }
	} // end of j == COLUMNS -1
    }
  }

/*
  cout << endl;   
   for( size_t i=0; i<ROWS; i++ ) {
      for( size_t j=0; j<COLUMNS; j++ ) { 
	 cout << P_data[ i*COLUMNS + j ] << " ";
      }	
      cout << "    ";	
      for( size_t j=0; j<COLUMNS; j++ ) { 
	 cout << P_indexes[ i*COLUMNS + j ] << " ";
      }     	
      cout << endl;	
   }

   cout << "Excluded elements"<< endl;	
   for( size_t i=0; i<ROWS; i++ ) {
      for( size_t j=0; j<COLUMNS; j++ ) { 
	 cout << (*excluded_elements_array)[ i*COLUMNS + j ] << " ";
      }	
      cout << endl;	
   }
*/



}


template<typename T, size_t ROWS, size_t COLUMNS>
void packet_treater<T, ROWS, COLUMNS>::calculate_threaded_result( size_t start_index, size_t end_index, size_t thread_index, 
				  				  const array<T, COLUMNS> & multiplicant_vec, std::promise<array<T, ROWS>> && promise ) { 
 
  array<T, ROWS> partial_result;  for( T &i : partial_result ) i = 0;
  vector<T> partial_data;   
  vector<size_t> partial_row, partial_column;

  size_t j = start_index%COLUMNS;
  size_t i = (start_index - j)/COLUMNS;
  bool allow_store = false; 

  // No need for mutexes in the excluded_elements_array, since the threads write to different places of the array each time
  for( size_t times = start_index; times <= end_index; times++ ) { 

    if( (i < P1_ROWS && j < P1_COLUMNS) || (i >= P1_ROWS && j >= P1_COLUMNS) ) allow_store = true;
    else allow_store = false;	

//    cout << "Examining element ["<<i<<"]["<<j<<"], allow store: "<< allow_store <<endl;

    (*excluded_elements_array)[i*COLUMNS+j] = 0;

     if(allow_store) {
       if( partial_data.size() <= ceil( (float) ADDED_PACKET_SIZES/this->used_threads ) ) {
	
	if( (*this->input_array)[i*COLUMNS+j] != 0 ) {
	   partial_column.push_back( j );
	   partial_row.push_back( i );		
	   partial_data.push_back( (*this->input_array)[i*COLUMNS+j] ); 		
        }

       } else (*excluded_elements_array)[i*COLUMNS+j] =(*this->input_array)[i*COLUMNS+j];

     } else (*excluded_elements_array)[i*COLUMNS+j] =(*this->input_array)[i*COLUMNS+j];

     if( j == COLUMNS -1 ) { j = 0; i++; }  // reached end of row
     else j++; // continue row traversal     
  }
 
  for( size_t i = 0; i< partial_data.size(); i++ )  
     partial_result[ partial_row[i] ] = (this->*(this->multiply_and_add))( partial_data[i], 
						      			   multiplicant_vec[ partial_column[i] ],
						       			   partial_result[ partial_row[i] ] ); 

  promise.set_value( partial_result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && packet_treater<T, ROWS, COLUMNS>::threaded_multiplication( const array<T, COLUMNS> & multiplicant_vec ) {

  vector<std::thread> thread_vector;     
  vector<std::future<array<T, ROWS>>> futures;

  size_t step = this->calculate_step();

  for( size_t i=0; i<size_t(this->used_threads); i++ ) {  

     std::promise<array<T, ROWS>> promise;
     futures.push_back( promise.get_future() );

     size_t finish = ( (i+1)*step < (*this->input_array).size() ) ? (i+1)*step -1 : (*this->input_array).size() -1;
     // If some elements are remaining, they are all placed in the last thread's work
     if( i == size_t(this->used_threads -1) && finish < (*this->input_array).size() -1 ) finish = (*this->input_array).size() -1;
 
//     cout << i*step << " " << finish << " step is:" << step << endl;	
   
     thread_vector.push_back( std::thread{ &packet_treater::calculate_threaded_result, this, i*step, finish, i, 
					   std::ref( multiplicant_vec ), std::move(promise) } );

     if( finish == (*this->input_array).size() -1 ) i = this->used_threads; // Break loop, if all elements have been reached
  }
 
  for( std::thread &i : thread_vector ) i.join(); // range_based loop 

  // Using the coo_treater class to compute the elements that don't fit in the packets 
  array<T, ROWS>* result = new array<T, ROWS>( (coo_treater<T, ROWS, COLUMNS> ( *excluded_elements_array, this->used_threads )).multiply_with_vector( multiplicant_vec ) );
 
  for( std::future<array<T, ROWS>> &fut : futures ) { 
	array<T, ROWS> partial_res( fut.get() ); 
	for( size_t j=0; j<(*result).size(); j++ ) (*result)[j] += partial_res[j];	
  }

  cout<<"Packet Threaded result : ";  for( size_t i=0; i< (*result).size(); i++ ) cout << (*result)[i] << " "; cout << endl;
  thread_vector.clear();  futures.clear();
  return std::move( *result );  
}


template<typename T, size_t ROWS, size_t COLUMNS>
std::array<T, ROWS> && packet_treater<T, ROWS, COLUMNS>::normal_multiplication( const array<T, COLUMNS> & multiplicant_vec ) { 

  array<T, ROWS> *result = new array<T, ROWS>;  for( size_t i =0; i<ROWS; i++ ) (*result)[i] = T(0);

  for( size_t i = 0; i<ROWS; i++ )
    for( size_t j = 0; j <COLUMNS; j++ ) 
      if( P_data[i*COLUMNS+j] != 0 ) {	
	(*result)[i] = (this->*(this->multiply_and_add))( P_data[i*COLUMNS+j], 
						          multiplicant_vec[ P_indexes[i*COLUMNS+j] ],
						          (*result)[i] ); 
      }	

//  cout <<"Normal packet result: "; for( size_t i=0; i<ROWS; i++ ) cout << (*result)[i] << " ";  cout << endl;
	 
  array<T, ROWS> rem_result( (coo_treater<T, ROWS, COLUMNS>( *excluded_elements_array )).multiply_with_vector( multiplicant_vec ) );
  for( size_t i=0; i<ROWS; i++ ) (*result)[i] += rem_result[i];

  cout <<"Normal packet result: "; for( size_t i=0; i<ROWS; i++ ) cout << (*result)[i] << " ";  cout << endl;
  return std::move( *result );
}


template<typename T, size_t ROWS, size_t COLUMNS>
packet_treater<T, ROWS, COLUMNS>::packet_treater( const array<T, ROWS*COLUMNS> & input_array )
				: abstract_format<T, ROWS, COLUMNS>( input_array ) { 

  P1_ROWS = ( ceil( ROWS/2 ) > ROWS - ceil( ROWS/2 ) ) ? ceil( ROWS/2 ) : ROWS - ceil( ROWS/2 );
  P1_COLUMNS = ( ceil( COLUMNS/2 ) > COLUMNS - ceil( COLUMNS/2 ) ) ?  ceil( COLUMNS/2 ) : COLUMNS - ceil( COLUMNS/2 );
//  cout << P1_ROWS <<" x "<< P1_COLUMNS << " , "<< ROWS - P1_ROWS <<" x " << COLUMNS - P1_COLUMNS << endl;

  excluded_elements_array = new array<T, ROWS*COLUMNS>;
}


template<typename T, size_t ROWS, size_t COLUMNS>
packet_treater<T, ROWS, COLUMNS>::packet_treater( const array<T, ROWS*COLUMNS> & input_array, int used_threads )
				: abstract_format<T, ROWS, COLUMNS>( input_array, used_threads ) { 

  P1_ROWS = ( ceil( ROWS/2 ) > ROWS - ceil( ROWS/2 ) ) ? ceil( ROWS/2 ) : ROWS - ceil( ROWS/2 );
  P1_COLUMNS = ( ceil( COLUMNS/2 ) > COLUMNS - ceil( COLUMNS/2 ) ) ?  ceil( COLUMNS/2 ) : COLUMNS - ceil( COLUMNS/2 );
//  cout << P1_ROWS <<" x "<< P1_COLUMNS << " , "<< ROWS - P1_ROWS <<" x " << COLUMNS - P1_COLUMNS << endl;

  excluded_elements_array = new array<T, ROWS*COLUMNS>;
}

#endif

