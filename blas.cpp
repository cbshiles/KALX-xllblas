// blas.cpp - BLAS routines from MKL
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
#include "xllblas.h"
#include "blas.h"

using namespace xll;
using namespace xml;

#ifndef EXCEL12
static AddInX xai_blas(
	ArgsX(CATEGORY)
	.Documentation(
		_T("The BLAS (Basic Linear Algebra Subprograms) are routines that provide ")
		_T("standard building blocks for performing basic vector and matrix operations. ")
		_T("The Level 1 BLAS perform scalar, vector and vector-vector operations, ")
		_T("the Level 2 BLAS perform matrix-vector operations, and the Level 3 BLAS ")
		_T("perform matrix-matrix operations. Because the BLAS are efficient, portable, ")
		_T("and widely available, they are commonly used in the development of high quality ")
		_T("linear algebra software, LAPACK for example. ")
	)
);
#endif // EXCEL12

#ifdef _DEBUG

using namespace blas;

extern "C" void xerbla_(const char* name, const int* info)
{
	std::string err;
	err = blas_error(name, *info);
	//throw std::runtime_error(blas_error(name, *info));
}

void
test_matrix(void)
{
	double a[] = {1, 2, 3,
		          4, 5, 6};
	matrix<double> A(a, 2, 3);
	ensure (A.rows() == 2);
	ensure (A.columns() == 3);
	ensure (A.major() == ROW_MAJOR);
	ensure (A[0][0] == 1);
	ensure (A[0][1] == 2);
	ensure (A[0][2] == 3);
	ensure (A[1][0] == 4);
	ensure (A[1][1] == 5);
	ensure (A[1][2] == 6);

	vector<double> r0 = A[0];
	ensure (r0.size() == 3);
	ensure (r0.incr() == 1);
	ensure (r0.off() == 0);
	ensure (r0[0] == 1);
	ensure (r0[1] == 2);
	ensure (r0[2] == 3);

	vector<double> r1 = A[1];
	ensure (r1.size() == 3);
	ensure (r1.incr() == 1);
	ensure (r1.off() == 3);
	ensure (r1[0] == 4);
	ensure (r1[1] == 5);
	ensure (r1[2] == 6);

	// 1, 4
	// 2, 5
	// 3, 6
	matrix<double,COLUMN_MAJOR> A_(a, 3, 2);
	ensure (A_.rows() == 3);
	ensure (A_.columns() == 2);
	ensure (A_.major() == COLUMN_MAJOR);
	ensure (A_[0][0] == 1);   
	ensure (A_[0][1] == 4);
	ensure (A_[1][0] == 2);
	ensure (A_[1][1] == 5);
	ensure (A_[2][0] == 3);
	ensure (A_[2][1] == 6);

	vector<double> r0_ = A_[0];
	ensure (r0_.size() == A_.columns());
	ensure (r0_.incr() == A_.rows());
	ensure (r0_.off() == 0);
	ensure (r0_[0] == 1);
	ensure (r0_[1] == 4);

	vector<double> r1_ = A_[1];
	ensure (r1_.size() == A_.columns());
	ensure (r1_.incr() == A_.rows());
	ensure (r1_.off() == 1);
	ensure (r1_[0] == 2);
	ensure (r1_[1] == 5);

	vector<double> r2_ = A_[2];
	ensure (r2_.size() == A_.columns());
	ensure (r2_.incr() == A_.rows());
	ensure (r2_.off() == 2);
	ensure (r2_[0] == 3);
	ensure (r2_[1] == 6);
}

void
test_dot(void)
{
	double a[] = {1, 2, 3};
	double b[] = {4, 5, 6};
	ensure (1*4 + 2*5 + 3*6 == dot<double>(vector<double>(a, dimof(a)), vector<double>(b, dimof(b))));
	ensure (2*4 + 3*6 == dot<double>(vector<double>(a, 2, 1, 1), vector<double>(b, 2, 2)));
}


void
test_gemv()
{
	double A[] = {1, 2,
		          3, 4};
	double x[] = {5, 
		          6};
	double y_[2] = {0, 0};
	vector<double> y(y_, 2);

	// y = 1*Ax + 0*y
	gemv<double>(1, matrix<double>(A, 2, 2), vector<double>(x, 2), 0, y);
	ensure (y[0] == 1*5 + 2*6);
	ensure (y[1] == 3*5 + 4*6);

	y[0] = y[1] = 0;
	gemv<double>(1, matrix<double, COLUMN_MAJOR>(A, 2, 2), vector<double>(x, 2), 0, y);
	ensure (y[0] == 1*5 + 3*6);
	ensure (y[1] == 2*5 + 4*6);

}

void
test_gemm(void)
{
	double 
	a[] = {1, 2,
	       3, 4},
	b[] = {5, 6,
	       7, 8},
	c[4];
	matrix<double, TRANSPOSE> C(c, 2, 2);
//	gemm(2, a, 2, b, 2, c);
	gemm<double>(1, matrix<double>(a, 2, 2), matrix<double>(b, 2, 2), 0, C);
/*	ensure (C[0][0] == 1*5 + 2*7);
	ensure (C[0][1] == 1*6 + 2*8);
	ensure (C[1][0] == 1*6 + 2*8);
	ensure (C[1][1] == 1*6 + 2*8);
*/}

int
xll_test_blas(void)
{
	try {
		test_matrix();
		test_dot();
		test_gemv();
		test_gemm();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return 1;
}
static Auto<Open> xao_test_blas(xll_test_blas);

#endif // _DEBUG