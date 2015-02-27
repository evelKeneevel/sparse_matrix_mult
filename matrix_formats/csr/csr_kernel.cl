
// One thread for each matrix row
__kernel void int_row_compute(   
			   __global const int* data, 
			   __global const int* pointers, 
			   __global const int* indices, 
			   __global const int* multiplicant_vec, 
			   __global int* result 
				) {

      __private int i = get_global_id(0);
      __private int elem_start = pointers[i];     
      __private int elem_finish = pointers[i+1]; 
      __private int elem_result = 0;

      for( __private int j = elem_start; j < elem_finish; j++ ) {
 
	 elem_result += data[j]*multiplicant_vec[ indices[j] ];
      }

      result[i] = elem_result;
}


// A workgroup consisting of 32 workitems is used to process each matrix row 
__kernel void float_compute(   
			   __global const float* data, 
			   __global const uint* pointers, 
			   __global const uint* indices, 
			   __global const float* multiplicant_vec, 
		  	   __local float* vals, 
			   __global float* result 
				) {

      __private uint local_size = get_local_size(0);
      __private uint i = get_global_id(0)/local_size;
      __private uint warp_idx = get_local_id(0);
      __private uint elem_start = pointers[i];     
      __private uint elem_finish = pointers[i+1]; 
      __private float inter;

      // Access times between private and local memory are trivial

      vals[warp_idx] = 0;

      for( __private uint j = elem_start + warp_idx; j < elem_finish; j += local_size ) {

	 // fma(a,b,c) == a*b +c 
	 vals[ warp_idx ] = fma( data[j] , multiplicant_vec[ indices[j] ], vals[ warp_idx ] );
      }
	
     write_mem_fence( CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE );

     if( warp_idx == 0 ) { // master thread adds all results from the rest of the workgroup 
	
        for( __private uint j = 1; j < local_size; j++ ) 
		vals[0] += vals[j];
	  //vals[0] = fma( vals[j] , (float) 1, vals[0] );

        result[i] = vals[0];
     }
}


// A workgroup consisting of local_size workitems is used to process each matrix row 
__kernel void int_compute(   
			   __global const int* data, 
			   __global const uint* pointers, 
			   __global const uint* indices, 
			   __global const int* multiplicant_vec, 
 			   __local int* vals, 
			   __global int* result 
				) {

      __private uint local_size = get_local_size(0);
      __private uint i = get_global_id(0)/local_size;
      __private uint warp_idx = get_local_id(0);
      __private uint elem_start = pointers[i];     
      __private uint elem_finish = pointers[i+1]; 
     
      vals[warp_idx] = 0;

      for( __private uint j = elem_start + warp_idx; j < elem_finish; j += local_size ) {
 
	 vals[ warp_idx ] += data[j]*multiplicant_vec[ indices[j] ];
      }

      // Code like if( warp_idx > 0 ) { vals[0] += vals[warp_idx]; } doesn't work!!!
      // This code is a hack to make the above functionality work
      for( __private uint j = 1; j < local_size; j++ ) 
	if( warp_idx == j ) { vals[0] += vals[warp_idx]; break; } 

      write_mem_fence( CLK_LOCAL_MEM_FENCE );

      if(warp_idx == 0) // First thread writes row result to its corresponding result row cell
        result[i] = vals[0];
}



