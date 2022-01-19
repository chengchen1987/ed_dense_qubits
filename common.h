#ifndef COMMON_H
#define COMMON_H

#define l_int long long
// I will set all integer related to "Dim" as l_int, in case of large systems
// do not use unsigned type because some intel MKL functions only support sigened type

#include <random>
#include <mkl.h>

// random number
double ran_num();
// bit operation, default l_int, 'int' for length of int 
int numOfBit1(const l_int& a);
void findBit1(const l_int& a, const int& n, int* b);
void print_binary(const l_int& a, const int& n);
//
l_int Power(const int& m, const int& n);
l_int Factorial(const int& m);
l_int Combination(const int& m, const int& n);
//
void NormalizedCopy(const double* f, double* g, const l_int& Dim);
void SetToZero(l_int* tmp, l_int length);
void SetToZero(double* f, const l_int dim);
//
void MatrixEvd(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w);
void MatrixEvd_Fortran(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w);
void MatrixSvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt);
//

double VN_entropy(const l_int& dim, double* wf);

// parse input 
void GetParaFromInput_int(const char* fname, const char* string_match, int& para);
void GetParaFromInput_real(const char* fname, const char* string_match, double& para);
void GetParaFromInput_char(const char* fname, const char* string_match, char& para);

void Vec_fwrite_double(const char* fname, double* data, const int& dsize);
void Vec_fread_double(const char* fname, double* data, const int& dsize);

#endif
