use cuda::ffi::runtime::{cudaStream_t};
use libc::{c_void, c_int};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum cusparseStatus_t {
  Success = 0,
  NotInitialized = 1,
  AllocFailed = 2,
  InvalidValue = 3,
  ArchMismatch = 4,
  MappingError = 5,
  ExecutionFailed = 6,
  InternalError = 7,
  MatrixTypeNotSupported = 8,
  ZeroPivot = 9,
}

#[repr(C)]
struct cusparseContext;
pub type cusparseHandle_t = *mut cusparseContext;

#[repr(C)]
struct cusparseMatDescr;
pub type cusparseMatDescr_t = *mut cusparseMatDescr;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum cusparsePointerMode_t {
  Host = 0,
  Device = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum cusparseDirection_t {
  Row = 0,
  Column = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum cusparseOperation_t {
  NonTranspose = 0,
  Transpose = 1,
  ConjugateTranspose = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum cusparseDiagType_t {
  NonUnit = 0,
  Unit = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum cusparseFillMode_t {
  Lower = 0,
  Upper = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum cusparseIndexBase_t {
  Zero = 0,
  One = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum cusparseMatrixType_t {
  General = 0,
  Symmetric = 1,
  Hermitian = 2,
  Triangular = 3,
}

#[link(name = "cusparse", kind = "dylib")]
extern "C" {
  pub fn cusparseCreate(handle: *mut cusparseHandle_t) -> cusparseStatus_t;
  pub fn cusparseDestroy(handle: cusparseHandle_t) -> cusparseStatus_t;
  pub fn cusparseGetVersion(handle: cusparseHandle_t, version: *mut c_int) -> cusparseStatus_t;
  pub fn cusparseSetStream(handle: cusparseHandle_t, stream: cudaStream_t) -> cusparseStatus_t;

  pub fn cusparseGetPointerMode(handle: cusparseHandle_t, mode: *mut cusparsePointerMode_t) -> cusparseStatus_t;
  pub fn cusparseSetPointerMode(handle: cusparseHandle_t, mode: cusparsePointerMode_t) -> cusparseStatus_t;

  pub fn cusparseCreateMatDescr(descr_a: *mut cusparseMatDescr_t) -> cusparseStatus_t;
  pub fn cusparseDestroyMatDescr(descr_a: cusparseMatDescr_t) -> cusparseStatus_t;
  pub fn cusparseSetMatType(descr_a: cusparseMatDescr_t, mat_ty: cusparseMatrixType_t) -> cusparseStatus_t;
  pub fn cusparseSetMatFillMode(descr_a: cusparseMatDescr_t, fill_mode: cusparseFillMode_t) -> cusparseStatus_t;
  pub fn cusparseSetMatDiagType(descr_a: cusparseMatDescr_t, diag_ty: cusparseDiagType_t) -> cusparseStatus_t;
  pub fn cusparseSetMatIndexBase(descr_a: cusparseMatDescr_t, base: cusparseIndexBase_t) -> cusparseStatus_t;

  pub fn cusparseSnnz(
      handle: cusparseHandle_t,
      dir_a: cusparseDirection_t,
      m: c_int, n: c_int,
      descr_a: cusparseMatDescr_t,
      a: *const f32, lda: c_int,
      nnz_per_row_col: *mut c_int,
      nnz_total: *mut c_int,
  ) -> cusparseStatus_t;
  pub fn cusparseSdense2csr(
      handle: cusparseHandle_t,
      m: c_int, n: c_int,
      descr_a: cusparseMatDescr_t,
      a: *const f32, lda: c_int,
      nnz_per_row: *const c_int,
      a_val: *mut f32,
      a_row_ptr: *mut c_int,
      a_col_ind: *mut c_int,
  ) -> cusparseStatus_t;

  pub fn cusparseSgemvi(
      handle: cusparseHandle_t,
      trans_a: cusparseOperation_t,
      m: c_int, n: c_int,
      alpha: *const f32,
      a: *const f32, lda: c_int,
      nnz: c_int,
      x_val: *const f32,
      x_ind: *const c_int,
      beta: *const f32,
      y: *mut f32,
      idx_base: cusparseIndexBase_t,
      p_buffer: *mut c_void,
  ) -> cusparseStatus_t;

  pub fn cusparseScsrmm(
      handle: cusparseHandle_t,
      trans_a: cusparseOperation_t,
      m: c_int, n: c_int, k: c_int,
      nnz: c_int,
      alpha: *const f32,
      descr_a: cusparseMatDescr_t,
      a_val: *const f32,
      a_row_ptr: *const c_int,
      a_col_ind: *const c_int,
      b: *const f32, ldb: c_int,
      beta: *const f32,
      c: *mut f32, ldc: c_int,
  ) -> cusparseStatus_t;
}
