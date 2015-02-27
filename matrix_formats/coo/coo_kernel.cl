

// This kernel is focused on processing elements belonging to a row, and
// place the result to the corresponding result position. It is slow 
__kernel void int_compute_slow( 
			  __global const int* data, 
			  __global const int* row_indexes,
			  __global const int* column_indexes, 
			  __global const int* multiplicant_vec,  
			  __global int* result ) {

    __private int row_index = get_global_id(0);
    __private int part_res = 0;
    __private int j=0;

    while( row_index > row_indexes[j] ) j++; 

    while( row_index == row_indexes[j] ) { 

       part_res += data[j]*multiplicant_vec[ column_indexes[j] ];   
       j++;	
    }

    result[ row_index ] = part_res; 
}

__kernel void float_compute( 
			    __global const float* data, 
			    __global const uint* row_indexes,
			    __global const uint* column_indexes, 
			    __global const float* multiplicant_vec, 
			    __global const uint* nonzero_elements,  
			    __global float* vals
				) {

    __private uint global_size = get_global_size(0); // ROWS*COO_LOCAL_WORK_SIZE
    __private uint nonzero = *nonzero_elements;

    // There exist ROWS workgroups, of COO_LOCAL_WORK_SIZE size each. 
    for( __private uint i = get_local_id(0) + get_group_id(0)*get_local_size(0); i < nonzero; i += global_size ) {
	vals[i] = fma( data[i], multiplicant_vec[ column_indexes[i] ], 0 );
    }

}


__kernel void int_compute( 
			  __global const int* data, 
			  __global const uint* row_indexes,
			  __global const uint* column_indexes, 
			  __global const int* multiplicant_vec, 
			  __global const uint* nonzero_elements,  
			  __global int* vals
				) {

    __private uint global_size = get_global_size(0); // ROWS*COO_LOCAL_WORK_SIZE
    __private uint nonzero = *nonzero_elements;

    // There exist ROWS workgroups, of COO_LOCAL_WORK_SIZE size each. 
    for( __private uint i = get_local_id(0) + get_group_id(0)*get_local_size(0); i < nonzero; i += global_size ) {

	vals[ i ] = data[i]*multiplicant_vec[ column_indexes[i] ];
    }

}





