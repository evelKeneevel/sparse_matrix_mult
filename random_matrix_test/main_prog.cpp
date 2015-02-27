
#include <iostream> 
#include <string>
#include <istream>
#include <array>
#include <sstream>
#include <thread>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <tuple>
#include <algorithm>
#include <mutex>
#include <exception>
#include <future>
#include <chrono>

#include "matrix_creator.hpp"
#include "ell/ellpack.hpp" 
#include "dia/dia.hpp" 
#include "csr/csr.hpp" 
#include "coo/coo.hpp" 
#include "packet/packet.hpp"

using namespace std;

//int total_ms;
int pak_b, pak_th_b, pak_gpu_b, coo_b, coo_th_b, coo_gpu_b, csr_b, csr_th_b, csr_gpu_b,
     dia_b, dia_th_b, dia_gpu_b, ell_b, ell_th_b, ell_gpu_b;

int pak_t = 0, pak_th_t=0, pak_gpu_t=0, coo_t=0, coo_th_t=0, coo_gpu_t=0, csr_t=0, csr_th_t=0, csr_gpu_t=0,
     dia_t=0, dia_th_t=0, dia_gpu_t=0, ell_t=0, ell_th_t=0, ell_gpu_t = 0;

const size_t REPEATS = 2;



template<typename T, size_t ARR_ROWS, size_t ARR_COLUMNS>
void call_stuff( ) {


  array<T, ARR_ROWS*ARR_COLUMNS> * arra = array_creator<T, ARR_ROWS, ARR_COLUMNS>(); 
  array<T, ARR_COLUMNS> * mult_vect = vector_creator<T, ARR_COLUMNS>();
  array<T, ARR_ROWS> * correct_res = simple_mult<T, ARR_ROWS, ARR_COLUMNS>( *arra, *mult_vect );

  std::chrono::high_resolution_clock::time_point t0, t1;
  std::chrono::milliseconds total_ms;


  for( size_t i=0; i<REPEATS; i++ ) {
      // Packet
      t0 = std::chrono::high_resolution_clock::now();
      packet_treater<T, ARR_ROWS, ARR_COLUMNS> pak( *arra );
//      pak * ( *mult_vect );  
      pak_b = compare_arrays<T, ARR_ROWS>( (*correct_res), pak * (*mult_vect) ); 
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required packet: " << total_ms.count() << endl;  
      pak_t +=  total_ms.count();

      // Packet, threaded	
      t0 = std::chrono::high_resolution_clock::now();
      packet_treater<T, ARR_ROWS, ARR_COLUMNS> pak_th( *arra, 4 ); 
//      pak_th * ( *mult_vect );  
      pak_th_b = compare_arrays<T, ARR_ROWS>( (*correct_res), pak_th * (*mult_vect) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required threaded packet: " << total_ms.count() << endl;  
      pak_th_t +=  total_ms.count();
 
      // Packet, GPU
      t0 = std::chrono::high_resolution_clock::now();
      packet_treater<T, ARR_ROWS, ARR_COLUMNS> pak_gpu( *arra, -1 ); 
//      pak_gpu * ( *mult_vect );  
      pak_gpu_b = compare_arrays<T, ARR_ROWS>( (*correct_res), pak_gpu * (*mult_vect) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required GPU packet: " << total_ms.count() << endl;   
      pak_gpu_t +=  total_ms.count();

  }


  for( size_t i=0; i<REPEATS; i++ ) {
      // COO
      t0 = std::chrono::high_resolution_clock::now();
      coo_treater<T, ARR_ROWS, ARR_COLUMNS> coo( *arra );
//      coo * ( *mult_vect );  
      coo_b = compare_arrays<T, ARR_ROWS>( (*correct_res), coo * (*mult_vect) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required coo: " << total_ms.count() << endl;  
      coo_t +=  total_ms.count();
 
      // COO threads
      t0 = std::chrono::high_resolution_clock::now();
      coo_treater<T, ARR_ROWS, ARR_COLUMNS> coo_th( *arra, 4 );
//      coo_th * ( *mult_vect );  
      coo_th_b = compare_arrays<T, ARR_ROWS>( (*correct_res), coo_th * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required threaded coo: " << total_ms.count() << endl;  
      coo_th_t +=  total_ms.count();
 
      // COO gpu
      t0 = std::chrono::high_resolution_clock::now();
      coo_treater<T, ARR_ROWS, ARR_COLUMNS> coo_gpu( *arra, -1 );
//      coo_gpu * ( *mult_vect );  
      coo_gpu_b = compare_arrays<T, ARR_ROWS>( (*correct_res), coo_gpu * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required coo gpu: " << total_ms.count() << endl;  
      coo_gpu_t +=  total_ms.count();

  }


  for( size_t i=0; i<REPEATS; i++ ) {
      // CSR stuff
      t0 = std::chrono::high_resolution_clock::now();
      csr_treater<T, ARR_ROWS, ARR_COLUMNS> csr( *arra );  
//      csr * ( *mult_vect );  
      csr_b = compare_arrays<T, ARR_ROWS>( (*correct_res), csr * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required csr: " << total_ms.count() << endl;     
      csr_t +=  total_ms.count();
/*
      // CSR threads
      t0 = std::chrono::high_resolution_clock::now();
      csr_treater<T, ARR_ROWS, ARR_COLUMNS> csr_th( *arra, 1 ); 
//      csr_th * ( *mult_vect );  
      csr_th_b = compare_arrays<T, ARR_ROWS>( (*correct_res), csr_th * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required csr_uber_threaded: " << total_ms.count() << endl;     
      csr_th_t +=  total_ms.count();
*/
      // CSR gpu
      t0 = std::chrono::high_resolution_clock::now();
      csr_treater<T, ARR_ROWS, ARR_COLUMNS> csr_gpu( *arra, -1 );  
//      csr_gpu * ( *mult_vect );  
      csr_gpu_b = compare_arrays<T, ARR_ROWS>( (*correct_res), csr_gpu * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required csr_gpu: " << total_ms.count() << endl;     
      csr_gpu_t +=  total_ms.count();
      
  } 


  for( size_t i=0; i<REPEATS; i++ ) {
      // DIA stuff 
      t0 = std::chrono::high_resolution_clock::now();
      diagonal_treater<T, ARR_ROWS, ARR_COLUMNS> dia( *arra );
//      dia * ( *mult_vect );  
      dia_b = compare_arrays<T, ARR_ROWS>( (*correct_res), dia * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time dia required: " << total_ms.count() << endl;    
      dia_t +=  total_ms.count();
 
      // DIA threads 
      t0 = std::chrono::high_resolution_clock::now();
      diagonal_treater<T, ARR_ROWS, ARR_COLUMNS> dia_th( *arra, 4 );
//      dia_th * ( *mult_vect );  
      dia_th_b = compare_arrays<T, ARR_ROWS>( (*correct_res), dia_th * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0); 
      cout << "Time threaded dia required: " << total_ms.count() << endl;    
      dia_th_t +=  total_ms.count();

      // DIA gpu
      t0 = std::chrono::high_resolution_clock::now();
      diagonal_treater<T, ARR_ROWS, ARR_COLUMNS> dia_gpu( *arra, -1 );
//      dia_gpu * ( *mult_vect );  
      dia_gpu_b = compare_arrays<T, ARR_ROWS>( (*correct_res), dia_gpu * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time gpu dia required: " << total_ms.count() << endl;     
      dia_gpu_t +=  total_ms.count();

  }


  for( size_t i=0; i<REPEATS; i++ ) {
      // ELL stuff
      t0 = std::chrono::high_resolution_clock::now();
      ell_treater<T, ARR_ROWS, ARR_COLUMNS> ell( *arra, 0 );
//      ell * ( *mult_vect );  
      ell_b = compare_arrays<T, ARR_ROWS>( (*correct_res), ell * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required ell: " << total_ms.count() << endl;  
      ell_t +=  total_ms.count();
 
      // ELL threads
      t0 = std::chrono::high_resolution_clock::now();
      ell_treater<T, ARR_ROWS, ARR_COLUMNS> ell_th( *arra, 4, 0 );
//      ell_th * ( *mult_vect );  
      ell_th_b = compare_arrays<T, ARR_ROWS>( (*correct_res), ell_th * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required ell threaded: " << total_ms.count() << endl;  
      ell_th_t +=  total_ms.count();

      // ELL gpu
      t0 = std::chrono::high_resolution_clock::now();
      ell_treater<T, ARR_ROWS, ARR_COLUMNS> ell_gpu( *arra, -1, 0 );
//      ell_gpu * ( *mult_vect );  
      ell_gpu_b = compare_arrays<T, ARR_ROWS>( (*correct_res), ell_gpu * ( *mult_vect ) );
      t1 = std::chrono::high_resolution_clock::now();
      total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      cout << "Time required ell gpu: " << total_ms.count() << endl;  
      ell_gpu_t +=  total_ms.count();
      
  }


/*
     cout<< "Pak :"<< pak_b << " Pak thread "<< pak_th_b << " Pak GPU "<< pak_gpu_b <<endl;
     cout<< "Pak time :"<< pak_t/REPEATS<<" Pak thread time :"<< pak_th_t/REPEATS << " Pak GPU time :"<< pak_gpu_t/REPEATS <<endl;  
     cout<< "COO :"<< coo_b << " COO thread "<< coo_th_b << " COO GPU " << coo_gpu_b <<endl;
     cout<< "COO time :"<< coo_t/REPEATS <<" COO thread time :"<< coo_th_t/REPEATS << " COO GPU time :"<< coo_gpu_t/REPEATS <<endl; 
     cout <<"CSR :"<< csr_b << " CSR thread "<< csr_th_b << " CSR GPU "<< csr_gpu_b <<endl;
     cout<< "CSR time :"<< csr_t/REPEATS<<" CSR thread time :"<< csr_th_t/REPEATS << " CSR GPU time :"<< csr_gpu_t/REPEATS <<endl; 
     cout <<"DIA :"<< dia_b << " DIA thread "<< dia_th_b << " DIA GPU "<< dia_gpu_b <<endl; 
     cout<< "DIA time :"<< dia_t/REPEATS<<" DIA thread time :"<< dia_th_t/REPEATS << " DIA GPU time :"<< dia_gpu_t/REPEATS <<endl; 
     cout <<"ELL :"<< ell_b << " ELL thread "<< ell_th_b << " ELL GPU "<< ell_gpu_b <<endl;
     cout<< "ELL time :"<< ell_t/REPEATS<<" ELL thread time :"<< ell_th_t/REPEATS << " ELL GPU time :"<< ell_gpu_t/REPEATS <<endl; 
*/
}


int main( int argc, char* argv[] ) {


  const size_t arr_rows = 50; 
  const size_t arr_cols = 70;

  call_stuff<int, arr_rows, arr_cols>( ); 


}

