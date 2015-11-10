#![feature(optin_builtin_traits)]

extern crate cuda;
extern crate libc;

use ffi::*;

use cuda::runtime::{CudaStream};
use std::ptr::{null_mut};

pub mod ffi;

pub type CusparseResult<T> = Result<T, cusparseStatus_t>;

#[derive(Clone, Copy, Debug)]
pub enum CusparsePointerMode {
  Host,
  Device,
}

impl CusparsePointerMode {
  #[inline]
  fn to_ffi(self) -> cusparsePointerMode_t {
    match self {
      CusparsePointerMode::Host   => cusparsePointerMode_t::Host,
      CusparsePointerMode::Device => cusparsePointerMode_t::Device,
    }
  }
}

pub struct CusparseHandle {
  pub ptr: cusparseHandle_t,
}

impl !Send for CusparseHandle {}

impl CusparseHandle {
  pub fn create() -> CusparseResult<CusparseHandle> {
    let mut handle: cusparseHandle_t = null_mut();
    match unsafe { cusparseCreate(&mut handle as *mut _) } {
      cusparseStatus_t::Success => Ok(CusparseHandle{ptr: handle}),
      e => Err(e),
    }
  }

  pub fn set_stream(&self, stream: &CudaStream) -> CusparseResult<()> {
    match unsafe { cusparseSetStream(self.ptr, stream.ptr) } {
      cusparseStatus_t::Success => Ok(()),
      e => Err(e),
    }
  }

  pub fn set_pointer_mode(&self, mode: CusparsePointerMode) -> CusparseResult<()> {
    match unsafe { cusparseSetPointerMode(self.ptr, mode.to_ffi()) } {
      cusparseStatus_t::Success => Ok(()),
      e => Err(e),
    }
  }
}

impl Drop for CusparseHandle {
  fn drop(&mut self) {
    match unsafe { cusparseDestroy(self.ptr) } {
      cusparseStatus_t::Success => {}
      e => {
        panic!("PANIC: failed to destroy CusparseHandle: cusparse status: {:?}", e);
      }
    }
  }
}

pub struct CusparseMatrixDesc {
  pub ptr: cusparseMatDescr_t,
}

impl CusparseMatrixDesc {
  pub fn create() -> CusparseResult<CusparseMatrixDesc> {
    let mut descr: cusparseMatDescr_t = null_mut();
    let mut desc = match unsafe { cusparseCreateMatDescr(&mut descr as *mut _) } {
      cusparseStatus_t::Success => CusparseMatrixDesc{ptr: descr},
      e => { return Err(e); }
    };
    match unsafe { cusparseSetMatType(desc.ptr, cusparseMatrixType_t::General) } {
      cusparseStatus_t::Success => {},
      e => { return Err(e); }
    }
    match unsafe { cusparseSetMatIndexBase(desc.ptr, cusparseIndexBase_t::Zero) } {
      cusparseStatus_t::Success => {},
      e => { return Err(e); }
    }
    Ok(desc)
  }
}

impl Drop for CusparseMatrixDesc {
  fn drop(&mut self) {
    match unsafe { cusparseDestroyMatDescr(self.ptr) } {
      cusparseStatus_t::Success => {}
      e => {
        panic!("PANIC: failed to destroy CusparseMatrixDesc: cusparse status: {:?}", e);
      }
    }
  }
}
