/*****************************************************************************
	
	FILE
		blas.h - C++ wrappers for http://www.netlib.org/blas/

	SYNOPSIS
		#include "lapack/blas.h"

	DESCRIPTION
		Convenience routines for calling BLAS.

	SEE ALSO
		http://www.netlib.org/blas/blast-forum/

*****************************************************************************/
#pragma once
#ifndef LAPACK_BLAS_H
#define LAPACK_BLAS_H
#include <stdexcept>
#include <sstream>

namespace blas {

// Handy constants

const int izero = 0;
const int ione  = 1;
const int itwo  = 2;

const double dzero = 0;
const double done  = 1;

const char DIAG_U = 'U';	// unit triangular
const char DIAG_N = 'N';	// not unit triangular

const char SIDE_R = 'R';	// right
const char SIDE_L = 'L';	// left

const char ROW_MAJOR = 'T';
const char COLUMN_MAJOR = 'N';
const char TRANSPOSE = 'N';

const char TRANS_N = 'T'; // row major
const char TRANS_T = 'N'; // column major
const char TRANS_R = 'R'; // conjugate
const char TRANS_C = 'C'; // conjugate transpose

const char UPLO_U = 'U';  // upper...
const char UPLO_L = 'L';  // ...lower triangular

class GB; // general band
class GE; // general rectangular
class SB; // symmetric band
class SP; // symmetric packed
class SY; // symmetric
class TB; // triangular band
class TP; // triangular packed
class TR; // triangular
class US; // unstructured sparse

inline 
std::string blas_error(const char* name, int info)
{
	std::ostringstream err;

	err << "blas_error: " << name << " problem with argument: " << info;

	return err.str();
};

extern "C" void xerbla_(const char* name, const int* info);

template<class T>
class vector {
	T* x_;
	int n_, incr_, off_;
public:
	vector(double* x, int n, int incr = 1, int off = 0)
		: x_(x), n_(n), incr_(incr), off_(off)
	{ }
	T* data(void)
	{
		return x_;
	}
	const T* data(void) const
	{
		return x_;
	}
	const int& size(void) const
	{
		return n_;
	}
	const int& incr(void) const
	{
		return incr_;
	}
	const int& off(void) const
	{
		return off_;
	}
	T& operator[](int i)
	{
		return x_[off_ + i*incr_];
	}
	T operator[](int i) const
	{
		return x_[off_ + i*incr_];
	}
};

#pragma warning(push)
#pragma warning(disable: 4100)
template<char t> inline int off(int i, int n);
// offet to i-th row
template<> inline int off<ROW_MAJOR>(int i, int n)
{
	return i*n;
}
// offet to i-th column
template<> inline int off<COLUMN_MAJOR>(int i, int n)
{
	return i;
}

template<char t> inline int incr(int m);
template<> inline int incr<ROW_MAJOR>(int m)
{
	return 1;
}
template<> inline int incr<COLUMN_MAJOR>(int m)
{
	return m;
}
#pragma warning(pop)

template<class T, char t = ROW_MAJOR>
class matrix {
	T* x_;
	int m_, n_;
	char t_;
public:
	matrix(double* x, int m, int n)
		: x_(x), m_(m), n_(n), t_(t)
	{ }
	T* data(void)
	{
		return x_;
	}
	const T* data(void) const
	{
		return x_;
	}
	const int& rows(void) const
	{
		return m_;
	}
	const int& columns(void) const
	{
		return n_;
	}
	const char& major(void) const
	{
		return t_;
	}
	// rows
	vector<T> operator[](int i)
	{
		return vector<T>(x_, n_, incr<t>(m_), off<t>(i, n_));
	}
	const vector<T> operator[](int i) const
	{
		return vector<T>(x_, n_, incr<t>(m_), off<t>(i, n_));
	}
};

// Computes dot product x' y
extern "C" double ddot_(const int* n, const double* x, const int* incrx, const double* y, const int* incry);
template<class T>
inline double dot(const vector<T>& x, const vector<T>& y)
{
	ensure (x.size() == y.size());

	return ddot_(&x.size(), x.data() + x.off(), &x.incr(), y.data() + y.off(), &y.incr());
}

// Computes a matrix-vector product using a general matrix
// y = alpha*trans(A)*x + beta*y
extern "C" void dgemv_(const char* trans, const int* m, const int* n, 
	const double* alpha, const double *a, const int* lda, const double *x,
	const int* incx, const double* beta, double *y, const int* incy);

template<class T, char t>
inline void gemv(T alpha, const matrix<T,t>& A, const vector<T>& x, T beta, vector<T>& y)
{
	dgemv_(&A.major(), &A.rows(), &A.columns(), &alpha, A.data(), &A.rows(), x.data(), &x.incr(), &beta, y.data() + y.off(), &y.incr());
}

// Computes a matrix-vector product using a triangular matrix
extern "C" void dtrmv_(const char* uplo, const char* trans, const char* diag,
	const int* n, const double *a, const int* lda, double *x, const int* incx);
inline void trmv(char uplo, char diag, int n, double *a, double *x, int incx = 1)
{
	char trans = n < 0 ? n = -n, TRANS_T : TRANS_N;

	dtrmv_(&uplo, &trans, &diag, &n, a, &n, x, &incx);
}

// Computes a scalar-matrix-matrix product and adds the result to 
// scalar-matrix product: c = alpha trans(a)*trans(b) + beta c
extern "C" void dgemm_(const char* transa, const char* transb, 
	const int* m, const int* n, const int* k, const double* alpha,
	const double *a, const int* lda, const double *b, const int* ldb,
	const double* beta, double *c, const int* ldc );

// multiply a (m x k) and b (k x n) returning c (m x n)
inline void gemm(int m, const double *a, int k, const double *b,
	int n, double *c)
{
	char transa = m < 0 ? m = -m, TRANS_T : TRANS_N;
	char transb = k < 0 ? k = -k, TRANS_T : TRANS_N;

	dgemm_(&transa, &transb, &n, &m, &k, &done, b, &n, a, &k,
		&dzero, c, &n);
}
template<class T, char t1, char t2, char t3>
inline void gemm(T alpha, const matrix<T, t1>& A, const matrix<T, t2>& B, T beta, matrix<T, t3>& C)
{
	dgemm_(&A.major(), &B.major(), &A.rows(), &A.columns(), &B.columns(), &alpha, A.data(), &A.rows(), B.data(), &B.columns(), &beta, C.data(), &C.columns());
}

} // namespace blas
#endif /* LAPACK_BLAS_H */
