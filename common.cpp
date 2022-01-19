#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <ctime>
#include <sstream>

#include "common.h"

using namespace std;

// a random number function from Anders Sandvik
double ran_num() {
	double ran;
	unsigned long long ran64;
	ifstream ifseed("seed.in");
	ifseed >> ran64;
	ifseed.close();
	unsigned long long irmax, mul64, add64;
	double dmu64;
	irmax = 9223372036854775807;
	mul64 = 2862933555777941757;
	add64 = 1013904243;
	dmu64 = 0.5 / double(irmax);
	ran64 = ran64 * mul64 + add64;
	ran = dmu64 * double(ran64);
	// refresh seed
	ran64 = (ran64 * mul64) / 5 + 5265361;
	ofstream ofseed("seed.in");
	ofseed << ran64;
	ofseed.close();
	return ran;
}

// bitwise operations
int numOfBit1(const l_int& b)
{
	int a = b;
	int cnt = 0;
	while (a != 0)
	{
		++cnt;
		a &= (a - 1);
	}
	return cnt;
}
// find first n '1's of a 
void findBit1(const l_int& a, const int& n, int* b)
{
	int x = 0;
	int i = 0;
	while (x < n) {
		if (((a >> i) & 1) == 1) {
			b[x] = i;
			x++;
		}
		i++;
	}
}

void print_binary(const l_int& a, const int& n)
{
	cout << "  ";
	for (int ix = 0; ix < n; ix++) cout << ((a >> (n - ix - 1)) & 1);
	cout << "  ";
}

//
l_int Power(int& m, int& n) {
	l_int ans = 1;
	for (int i = 0; i < n; i++) {
		ans = ans * (long long)m;
	}
	return ans;
}
//
l_int Factorial(const int& m) {
	l_int ans = 1;
	for (int i = 1; i < m + 1; i++) {
		ans = ans * i;
	}
	return ans;
}
//
l_int Combination(const int& m, const int& n) {
	l_int  ans = 1;
	int minmn = min(n, m - n);
	int maxmn = max(n, m - n);
	for (int i = 0; i < minmn; i++) {
		ans = ans * (maxmn + 1 + i);
	}
	ans = ans / Factorial(minmn);
	return ans;
}

//
void NormalizedCopy(const double* f, double* g, const l_int& Dim) {
	double res = 0;
	for (l_int i = 0; i < Dim; i++)
		res += f[i] * f[i];
	res = sqrt(res);
	for (l_int j = 0; j < Dim; j++)
		g[j] = f[j] / res;
}

//
void SetToZero(l_int* tmp, l_int length) {
	for (l_int x = 0; x < length; x++) tmp[x] = 0;
}
void SetToZero(double* f, const l_int dim) {
	for (l_int i = 0; i < dim; i++) f[i] = 0;
}

// ===============================================================================
// Matrix evd (use mkl lapacke function) 
void MatrixEvd(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w) 
{
	//LAPACKE_dsyev(matrix_layout, jobz, uplo, n, a, lda, w);
    //
    LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w);

    /*
    char range = 'A'; // all eigenvalues 
    lapack_int* isuppz = new lapack_int[2 * n];
    double abstol = 0;
    LAPACKE_dsyevr(matrix_layout, jobz, range, uplo,
        n, a, lda, NULL, NULL, NULL, NULL,
        abstol, &n, w, a, n, isuppz);
    */
}

// caution! Fortran interface gives different results!
void MatrixEvd_Fortran(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w) 
{
    // dsyev
/*
    int info, lwork;
    double wkopt;
    double* work;
    lwork = -1;
    dsyev(&jobz, &uplo, &n, a, &lda, w, &wkopt, &lwork, &info );
    lwork = (int)wkopt;
    cout << "lwork = " << lwork; 
    work = (double*)malloc( lwork*sizeof(double) );
    // Solve eigenproblem
    dsyev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info );
*/

    // dsyevr 
    /*
    int il, iu, ldz, info, lwork, liwork;
    ldz = lda;
    double abstol, vl, vu;
    int iwkopt;
    int* iwork;
    double wkopt;
    double* work;
    int isuppz[n];

    abstol = -1;
    il = 1;
    iu = n;
    lwork = -1;
    liwork = -1;
    dsyevr( "Vectors", "Indices", "Upper", &n, a, &lda, &vl, &vu, &il, &iu,
                        &abstol, &n, w, a, &ldz, isuppz, &wkopt, &lwork, &iwkopt, &liwork,
                        &info );
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    liwork = iwkopt;
    iwork = (int*)malloc( liwork*sizeof(int) );
    cout << "lwork = " << lwork << ", liwork = " << liwork << endl; 
    // Solve eigenproblem
    dsyevr( "Vectors", "Indices", "Upper", &n, a, &lda, &vl, &vu, &il, &iu,
                        &abstol, &n, w, a, &ldz, isuppz, work, &lwork, iwork, &liwork,
                        &info );
    */

    // dsyevd
    int info, lwork, liwork;
    int* iwork;
    double* work;
/*
    int iwkopt;
    double wkopt;
    lwork = -1;
    liwork = -1;
    dsyevd(&jobz, &uplo, &n, a, &lda, w, &wkopt, &lwork, &iwkopt,
            &liwork, &info );
    
    lwork = (int)wkopt;
    liwork = iwkopt;
*/
    lwork = 2*n*n + 6*n + 1;
    work = (double*)malloc( lwork*sizeof(double) );
    liwork = 5*n + 3;
    iwork = (int*)malloc( liwork*sizeof(int) );
    cout << "lwork = " << lwork << ", liwork = " << liwork << endl; 
    // Solve eigenproblem
    dsyevd(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork,
                        &liwork, &info );

    /* Check for convergence */
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }

    // dsyevr 

}

// Matrix svd (calling lapacke matrix svd)
void MatrixSvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt) {
	double* superb = new double[min(m, n)];
	LAPACKE_dgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

//
double VN_entropy(const l_int& dim, double* wf) {
	/*	double* log_wf = new double[dim];
		vdLn(dim, wf, log_wf);
		double aux = cblas_ddot(dim, wf, 1, log_wf, 1);
		delete[] log_wf;
		return -aux;
	*/
	// in case log(0) = nan might be a problem
	double aux = 0;
	for (l_int i = 0; i < dim; i++) {
		if (wf[i] > 1e-32) {
			aux += -wf[i] * log(wf[i]);
		}
	}
	return aux;
}

void GetParaFromInput_int(const char* fname, const char* string_match, int& para) {
	FILE* f_in = fopen(fname, "r");
	char testchar[40], line[80];
	sprintf(testchar, string_match);
	int len = strlen(testchar);
	while (fgets(line, 200, f_in) != NULL) {
		if (strncmp(line, testchar, len) == 0)
		{
			char* p = strtok(line, "=");
			stringstream ss;
			p = strtok(NULL, "=");
			ss << p;
			ss >> para;
			std::cout << "GetParaFromInput: " << string_match << " " << para << std::endl;
			break;
		}
	}
	fclose(f_in);
	f_in = NULL;
}

void GetParaFromInput_real(const char* fname, const char* string_match, double& para) {
	FILE* f_in = fopen(fname, "r");
	char testchar[40], line[80];
	sprintf(testchar, string_match);
	int len = strlen(testchar);
	while (fgets(line, 200, f_in) != NULL) {
		if (strncmp(line, testchar, len) == 0)
		{
			char* p = strtok(line, "=");
			stringstream ss;
			p = strtok(NULL, "=");
			ss << p;
			ss >> para;
			std::cout << "GetParaFromInput: " << string_match << " " << para << std::endl;
			break;
		}
	}
	fclose(f_in);
	f_in = NULL;
}

void GetParaFromInput_char(const char* fname, const char* string_match, char& para) {
	FILE* f_in = fopen(fname, "r");
	char testchar[40], line[80];
	sprintf(testchar, string_match);
	int len = strlen(testchar);
	while (fgets(line, 200, f_in) != NULL) {
		if (strncmp(line, testchar, len) == 0)
		{
			char* p = strtok(line, "=");
			stringstream ss;
			p = strtok(NULL, "=");
			ss << p;
			ss >> para;
			std::cout << "GetParaFromInput: " << string_match << " " << para << std::endl;
			break;
		}
	}
	fclose(f_in);
	f_in = NULL;
}

void Vec_fwrite_double(const char* fname, double* data, const int& dsize)
{
	FILE* f_out;
	f_out = fopen(fname, "wb");
	fwrite(data, sizeof(double), dsize, f_out);
	fclose(f_out);
}

void Vec_fread_double(const char* fname, double* data, const int& dsize)
{
	FILE* f_in;
	f_in = fopen(fname, "rb");
	fread(data, sizeof(double), dsize, f_in);
	fclose(f_in);
}