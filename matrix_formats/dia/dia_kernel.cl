
// Array is divided into ROWS. get_local_id() gives the current working ROW, and based on that are all the
// computations done. Result is stored in C array. Result is always assigned, and never accumulated!!
// This avoids using/adding old values lying in the C array. With assignment, always a new value is given to C

__kernel void float_compute(__global const float* diagonals_data, 
			    __global const int* offset, 
			    __global const float* multiplicant_vec,  
			    __global const uint* columns, 
			    __global const uint* mult_vec_cols, 
			    __global float* result ) {
 
     __private float part_res = 0;
     __private uint row_index = get_global_id(0), data_columns = *columns;	
     __private uint mult_vec_columns = *mult_vec_cols; 
     __private int mult_vec_index; 

     for( uint column_index = 0; column_index < data_columns; column_index++ ) {

	mult_vec_index = ((int) row_index) + offset[ column_index ];
	if( mult_vec_index >= 0 && mult_vec_index <  ((int) mult_vec_columns) )	
	  part_res = fma( diagonals_data[ data_columns*row_index + column_index ], multiplicant_vec[(uint) mult_vec_index], part_res );
     }

     result[ row_index ] = part_res; // The += operator creates problems
}


__kernel void int_compute(__global const int* diagonals_data, 
			  __global const int* offset, 
			  __global const int* multiplicant_vec,  
			  __global const uint* columns, 
			  __global const uint* mult_vec_cols, 
			  __global int* result ) {
 
     __private int part_res = 0;
     __private uint row_index = get_global_id(0), data_columns = *columns;	
     __private uint mult_vec_columns = *mult_vec_cols; 
     __private int mult_vec_index; 

     for( uint column_index = 0; column_index < data_columns; column_index++ ) {

	mult_vec_index = ((int) row_index) + offset[ column_index ];
	if( mult_vec_index >= 0 && mult_vec_index <  ((int) mult_vec_columns) )
	  part_res += diagonals_data[ data_columns*row_index + column_index ]*multiplicant_vec[(uint) mult_vec_index];
     }

     result[ row_index ] = part_res; // diagonals_data[11];  // part_res; 
}


