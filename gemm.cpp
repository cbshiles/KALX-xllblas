// gemm.cpp - BLAS routines from MKL
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
#include "xllblas.h"

using namespace xll;

static AddInX X_(xai_blas_gemm)(
	FunctionX(XLL_FPX, _T("?xll_blas_gemm"), _T("GEMM"))
	.Arg(XLL_FPX, _T("A"), _T("is the first matrix."))
	.Arg(XLL_FPX, _T("B"), _T("is the second matrix. "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Return the matrix product of A and B."))
	.Documentation(
		_T("This calls the BLAS function <codeInline>DGEMM</codeInline>.")
	)
);
xfp* WINAPI
xll_blas_gemm(const xfp* pa, const xfp* pb)
{
#pragma XLLEXPORT
	static FPX c;

	try {
		ensure (pa->columns == pb->rows);
		c.reshape(pa->rows, pb->columns);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pa->rows, pb->columns, pa->columns,
			1, pa->array, pa->columns, pb->array, pb->columns, 0, c.array(), c.columns());
	}
	catch (std::exception& ex) {
		XLL_ERROR(ex.what());

		return 0;
	}

	return c.get();
}

