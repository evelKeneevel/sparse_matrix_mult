
// 1 thread per matrix (packet matrix) row
__kernel void float_compute(
			  __global const float* data, 
			  __global const uint* indexes, 
			  __global const uint* P1_ROWS,	
			  __global const uint* P1_COLUMNS,
			  __global const uint* COLUMNS,
			  __global const float* mult_vec, 
			  __local float* local_mult_vec,
			  __global float* result 
					) {

     __private uint i = get_global_id(0);
     __private float part_res = 0;
     __private uint start, finish; 
     __private uint columns = *COLUMNS;

     if( i < *P1_ROWS ) { start = 0; finish = *P1_COLUMNS; } 
     else { start = *P1_COLUMNS; finish = columns; }

     if( columns <= 4000 ) {

       // Copying to local memory the mult_vec array
       for( __private uint l = 0; l < columns; l++ ) local_mult_vec[l] = mult_vec[l];	
 
       for( __private uint j = start; j < finish; j++ ) 
	  if( data[i*columns+j] != 0 )	
	    part_res = fma( data[i*columns+j], local_mult_vec[ indexes[i*columns+j] ], part_res );

     } else {

	for( __private uint j = start; j < finish; j++ ) 
	  if( data[i*columns+j] != 0 )	
	    part_res = fma( data[i*columns+j], mult_vec[ indexes[i*columns+j] ], part_res );
     }

     result[i] = part_res;
}


__kernel void int_compute(
			  __global const int* data, 
			  __global const uint* indexes, 
			  __global const uint* P1_ROWS,	
			  __global const uint* P1_COLUMNS,
			  __global const uint* COLUMNS,
			  __global const int* mult_vec, 
			  __local int* local_mult_vec,
			  __global int* result 
					) {

     __private uint i = get_global_id(0);
     __private int part_res = 0;
     __private uint start, finish; 
     __private uint columns = *COLUMNS;

     if( i < *P1_ROWS ) { start = 0; finish = *P1_COLUMNS; } 
     else { start = *P1_COLUMNS; finish = columns; }

     if( columns <= 4000 ) {

       // Copying to local memory the mult_vec array
       for( __private uint l = 0; l < columns; l++ ) local_mult_vec[l] = mult_vec[l];	
 
       for( __private uint j = start; j < finish; j++ ) 
	  if( data[i*columns+j] != 0 )	
	     part_res += data[i*columns+j]*local_mult_vec[ indexes[i*columns+j] ];  

     } else {

	for( __private uint j = start; j < finish; j++ ) 
	  if( data[i*columns+j] != 0 )	
	     part_res += data[i*columns+j]*mult_vec[ indexes[i*columns+j] ];  
     }

     result[i] = part_res;
}


