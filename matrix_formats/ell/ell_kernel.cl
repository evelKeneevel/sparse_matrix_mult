
// Array is divided into ROWS. get_local_id() gives the current working ROW, and based on that are all the
// computations done. Result is stored in C array. Result is always assigned, and never accumulated!!
// This avoids using/adding old values lying in the C array. With assignment, always a new value is given to C
// size_t type should be avoided as a kernel argument!! It probably isn't supported

// __local objects data & indices points to an array, it's size is K_dim*sizeof(T), seen on the following funcion: 
// clSetKernelArg(kernel, 1, K_dim*sizeof(T), 0 ); Also, a local work group size has to be defined as well 

// This kernel builds his own data & indice arrays, but does it slower than the CPU
__kernel void int_compute_kernel_does_the_work(
			  __global const int* input_array,  // __constant 
			  __local int* data, 
			  __local int* indices, 
			  __global const int* K_dim,  		
			  __global const int* COLUMNS,      // __constant 
			  __global const int* mult_vec,     // __constant 
			  __global int* result ) {

   __private int i = get_global_id(0);
   __private const int k_dim = *K_dim;
   __private int part_res = 0, data_j = 0;
   __private const columns = *COLUMNS;

   for( __private int j=0; j<columns; j++ ) {

	if( input_array[ i*columns + j ] != 0 ) {	

	   data[ data_j ] = input_array[ i*columns + j ];
	   indices[ data_j ] = j;
	   data_j++;	
	}
   } 

   // EXTREMELY important to zero padd the indices array, otherwise undefined memory locations will be accessed, 
   // which can cause the X server to hang
   // The remaining values of data[] are zeroed, so no issue will arise with wrong values
   for( data_j; data_j<k_dim; data_j++ ) { data[ data_j ] = 0;  indices[data_j] = 0; } // Fill remainder of row with 0s

   // Redundant call, is useful if local workobjects are more than one
   // barrier( CLK_LOCAL_MEM_FENCE );

   for( __private int j=0; j<k_dim; j++ ) 
	 part_res += data[ j ]*mult_vec[ indices[ j ] ];
   
   result[ i ] =  part_res;

}

// This kernel is way faster, because it only computes stuff
__kernel void int_compute(
			  __global const int* data, 
			  __global const uint* indices, 
			  __global const uint* K_dim,  		
			  __global const int* mult_vec, 
			  __global int* result ) {

   __private uint i = get_global_id(0);
   __private const uint k_dim = *K_dim;
   __private int part_res = 0;

   for( __private uint j=0; j<k_dim; j++ ) {
      
      if( data[ i*k_dim + j ] != 0 ) {
	 part_res += data[ i*k_dim + j ]*mult_vec[ indices[ i*k_dim + j ] ];
      }
   }

   result[ i ] =  part_res;

}

// This kernel is way faster, because it only computes stuff
__kernel void float_compute(
			  __global const float* data, 
			  __global const uint* indices, 
			  __global const uint* K_dim,  		
			  __global const float* mult_vec, 
			  __global float* result ) {

   __private uint i = get_global_id(0);
   __private const uint k_dim = *K_dim;
   __private float part_res = 0;

   for( __private uint j=0; j<k_dim; j++ ) { 
      if( data[ i*k_dim + j ] != 0 ) {
         part_res = fma( data[ i*k_dim + j ], mult_vec[ indices[ i*k_dim + j ] ], part_res );
      }
   }  

   result[ i ] =  part_res;
}



