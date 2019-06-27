/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef TDL_INCLUDE_DNN_SHIFT_H
#define TDL_INCLUDE_DNN_SHIFT_H

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"

namespace triton{
namespace dnn{

class shift {

public:
  enum type {
    FPROP
  };

private:
  void set_ld(const std::vector<int32_t>& shapes,
              std::vector<int32_t>& ld);

public:

  shift(int B, int NC,
       int D, int H, int W,
       int T, int R, int S, int NF,
       const std::vector<int32_t> &shift_h, const std::vector<int32_t> &shift_w,
       std::string a_ty = "fp32", std::string b_ty = "fp32",
       type ty = FPROP, bool bias = false);

  // look-up table
  void build_deltas();
  void build_masks();

  // accessors
  size_t a_size();
  size_t b_size();
  size_t c_size();
  std::vector<int32_t> c_shapes();

  // device function
  void init(driver::stream *stream, driver::cu_module *module);
  void enqueue(driver::stream *stream, driver::kernel *kernel,
               driver::buffer *a, driver::buffer *b, driver::buffer *c,
               size_t TM, size_t TN, size_t nthreads);

  // utils
  size_t get_nflops();

  // source
  void src(std::ostream &os);

  // cpu_ref
  template<class IN_DTYPE, class OUT_DTYPE>
  void cpu_ref(OUT_DTYPE* O,
                  const IN_DTYPE* I,
                  const IN_DTYPE* F)
  {
    OUT_DTYPE acc;
    for(int32_t p = 0; p < AH_; ++p)
    for(int32_t q = 0; q < AW_; ++q)
    for(int32_t bs = 0; bs < NB_; ++bs)
    for(int32_t k = 0; k < NF_; ++k)
    {
      acc = 0;
      for(int32_t c = 0; c < NC_; ++c){
        int32_t h = p + shift_h_[c];
        int32_t w = q + shift_w_[c];
        bool in_bounds = (h >= 0 && w >= 0 && h < AH_ && w < AW_);
        IN_DTYPE a = in_bounds?I[bs + w*NB_ + h*NB_*AW_ + c*NB_*AH_*AW_]:0;
        IN_DTYPE b = F[k + c*NF_];
        acc = std::fma(a, b, acc);
      }
      O[bs + q*NB_ + p*NB_*AW_ + k*NB_*AH_*AW_] = acc;
    }
  }

private:
  // image size
  int32_t NB_;
  int32_t NC_;
  int32_t AD_;
  int32_t AH_;
  int32_t AW_;
  // filter size
  int32_t BD_;
  int32_t BH_;
  int32_t BW_;
  int32_t NF_;
  // activation size
  int32_t CD_;
  int32_t CH_;
  int32_t CW_;
  // equivalent matmul
  int32_t M_;
  int32_t N_;
  int32_t K_;
  // shapes
  std::vector<int32_t> shapes_a_;
  std::vector<int32_t> shapes_b_;
  std::vector<int32_t> shapes_c_;
  // memory strides
  std::vector<int32_t> ld_a_;
  std::vector<int32_t> ld_b_;
  std::vector<int32_t> ld_c_;
  // shift values
  std::vector<int32_t> shift_h_;
  std::vector<int32_t> shift_w_;
  // look-up tables
  std::vector<int32_t> h_deltas_;
  std::vector<int32_t> h_masks_;
  driver::buffer* d_deltas_;
  driver::buffer* d_masks_;
  // data types
  std::string a_ty_;
  std::string b_ty_;
  // convolution type
  type ty_;
  bool bias_;
};

}
}

#endif