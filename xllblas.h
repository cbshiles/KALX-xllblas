// blas.h - Header file for BLAS routines.
// Copyright (c) 2011 KALX, LLC. All rights reserved. No warranty is made.
#include "mkl_cblas.h"
//#define EXCEL12
#include "xll/xll.h"

#ifndef CATEGORY
#define CATEGORY _T("BLAS")
#endif
#ifndef BLAS_PREFIX
#define BLAS_PREFIX CATEGORY _T(".")
#endif

typedef xll::traits<XLOPERX>::xcstr xcstr;
typedef xll::traits<XLOPERX>::xword xword;
typedef xll::traits<XLOPERX>::xfp xfp;
