fn main() {
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l dylib=cusparse");
  //println!("cargo:rustc-flags=-l dylib=cusparse");

  println!("cargo:rustc-flags=-l dylib=cusparse");
}
