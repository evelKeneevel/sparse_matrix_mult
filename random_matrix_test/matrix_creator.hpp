
// Prevents multiple inclusions
#ifndef ____matrix_creator____
#define ____matrix_creator____

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
#include <iomanip>
#include <fstream>
#include <istream>
#include <string>
#include <iostream>

//const size_t ARRAY_ROWS = 6;
//const size_t ARRAY_COLUMNS = 3;
const size_t EMPTY_DIAGONALS_1 = 0; //2;
const size_t EMPTY_DIAGONALS_2 = 0; //3;
const int NUMB_DIV = 1;

using namespace std;


template<typename T, size_t ARRAY_ROWS> 
int compare_arrays( const array<T, ARRAY_ROWS> & correct_res, const array<T, ARRAY_ROWS> & B ) {

   int wrong_result = 0;
   cout << setprecision(16); // show 16 digits
   cout << endl;
   for( size_t i=0; i<ARRAY_ROWS; i++ ) {

//     cout << correct_res[i] << "  " << B[i] << endl;

     if( correct_res[i] - B[i] != 0 ) { 
	cout << " ****** Wrong result! ****** on index "<<i<< ": "<< correct_res[i] << " & "<< B[i] <<" ! error: " << correct_res[i] - B[i] <</*" \n"<<*/ endl; 
	wrong_result += 1; 
//	return; 
     }
   } 
   if( wrong_result == 0 ) cout <<" !!!! Correct result !!!!"<<endl;
   cout << endl;
   return wrong_result;  
}


template<typename T> 
T mult_function( const T & arr_val, const T & vec_val, const T & part_res ) {

  if( typeid(T).name()[0] == 'f' ) { return fma( arr_val, vec_val, part_res ); } 
  else if( typeid(T).name()[0] == 'i' ) { return arr_val*vec_val + part_res; }
  else if( typeid(T).name()[0] == 'd' ) { return fma( arr_val, vec_val, part_res );  }
  else if( typeid(T).name()[0] == 'l' ) { return fma( arr_val, vec_val, part_res );  }
  else if( typeid(T).name()[0] == 'e' ) { return fma( arr_val, vec_val, part_res );  }
  else if( typeid(T).name()[0] == 's' ) { return arr_val*vec_val + part_res; }
 
//  cout << typeid(T).name() << endl; 
  return T(-1);

}


template<typename T, size_t ARRAY_ROWS, size_t ARRAY_COLUMNS> 
std::array<T, ARRAY_ROWS> * 
simple_mult( const std::array<T, ARRAY_ROWS*ARRAY_COLUMNS> & arr, const std::array<T, ARRAY_COLUMNS> & vec ) {

  std::array<T, ARRAY_ROWS> *res = new std::array<T, ARRAY_ROWS>();
  T partial_res = 0;
  cout<<setprecision(16);
  
  for( size_t i=0; i<ARRAY_ROWS; i++ ) { 
    for( size_t j=0; j<ARRAY_COLUMNS; j++ ) {
	
       partial_res = mult_function<T>( arr[ i*ARRAY_COLUMNS+ j] , vec[ j ] ,partial_res );	

    }
    (*res)[i] = (T) partial_res;
    partial_res = 0;	
  }

  cout << "CORRECT RESULT IS: ";
  for( size_t i = 0; i< (*res).size(); i++ ) cout << (*res)[i] << " "; cout<<endl; // range_based loop 
  
  return res;
}


template<typename T, size_t ARRAY_COLUMNS> 
std::array<T, ARRAY_COLUMNS> * /* && */ vector_creator() {

   std::array<T, ARRAY_COLUMNS> *vector = new std::array<T, ARRAY_COLUMNS>();
   
   for( size_t i=0; i<ARRAY_COLUMNS; i++ ) { (*vector)[i] = (T) (rand()%10)/NUMB_DIV; /* T( ((rand()%10)*NUMB) );*/ cout << " "<< (*vector)[i] ;  }
   cout << "\n" << endl;
	
   return vector;  //std::move( *vector );
}


template<typename T, size_t ARRAY_ROWS, size_t ARRAY_COLUMNS> 
std::array<T, ARRAY_ROWS*ARRAY_COLUMNS> * /* && */ array_creator() {

   std::array<T, ARRAY_ROWS*ARRAY_COLUMNS> *arr = new std::array<T, ARRAY_ROWS*ARRAY_COLUMNS>();
   
   for( size_t i=0; i<(*arr).size(); i++ ) (*arr)[i] =  (T) (rand()%10)/NUMB_DIV;   //T( ((rand()%10)*NUMB) ); 

   for( size_t k=0; k<EMPTY_DIAGONALS_1; k++ ) {

	size_t di = rand()%ARRAY_ROWS;
	cout << " Emptying from cell ["<< di << "][0]"<< endl; 
	
	size_t j = di*ARRAY_COLUMNS;
	while( j < ARRAY_COLUMNS*ARRAY_ROWS ) { 
		
		(*arr)[j] = 0;	 
		j += ARRAY_COLUMNS+1;
	}
   }
    
   for( size_t k=0; k<EMPTY_DIAGONALS_2; k++ ) {

	size_t di = rand()%ARRAY_COLUMNS;
	cout << " Emptying from cell [0]["<< di << "]"<< endl; 
	
	size_t j = di;
	while( j < ARRAY_ROWS*ARRAY_COLUMNS ) {
		(*arr)[j] = 0;	 
		j += ARRAY_COLUMNS+1;
	}
   } 	

   for( size_t i=0; i<ARRAY_ROWS; i++ ) {
      for( size_t j=0; j<ARRAY_COLUMNS; j++ ) { 
	 cout << (*arr)[ i*ARRAY_COLUMNS + j ] << " ";
      }	
      cout << endl;	
   }

   cout << endl;
   for( size_t i=0; i< (*arr).size(); i++ ) 
	if( (*arr)[i] < 0 ) cout << "Index "<< i << " has " << (*arr)[i] << endl; 

   return arr;  //std::move( *arr );
}

#endif
