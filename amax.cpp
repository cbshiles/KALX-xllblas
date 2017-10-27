// amax.cpp - BLAS routines from MKL
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
#include "xllblas.h"
#include "blas.h"

using namespace xll;

static AddInX X_(xai_blas_iamax)(
	FunctionX(XLL_WORDX, _T("?xll_blas_iamax"), _T("IAMAX"))
	.Arg(XLL_FPX, _T("Array"), _T("is the array for which you want the index of the largest absolute value."))
	.Arg(XLL_WORDX, _T("Offset?"), _T("is the optional offset to use into Array."))
	.Arg(XLL_WORDX, _T("Increment?"), _T("is the optional increment for the elements of Array. "))
	.Category(CATEGORY)
	.FunctionHelp(_T("Finds the index of the element of Array that has the largest absolute value."))
	.Documentation(
		_T("This calls the BLAS function <codeInline>IDAMAX</codeInline>.")
	)
);
xword WINAPI
xll_blas_iamax(xfp* pa, xword off, xword inc)
{
#pragma XLLEXPORT

	if (inc == 0)
		inc = 1;

	if (inc >= size(*pa))
		return static_cast<xword>(-1); //???return XLOPER error???

	return static_cast<xword>(cblas_idamax(size(*pa) - off, pa->array + off, inc));
}
