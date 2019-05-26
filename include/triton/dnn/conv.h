#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"

namespace triton{
namespace dnn{

class conv {
public:
  enum type {
    FPROP,
    BPROP,
    WGRAD
  };

private:
  void set_ld(const std::vector<int32_t>& shapes,
                    std::vector<int32_t>& ld);

  std::tuple<int32_t, int32_t, int32_t, int32_t>
      unpack(int32_t ltrs, bool flip, int32_t EBD, int32_t EBH, int32_t EBW);

public:

  conv(int B, int NC,
       int D, int H, int W,
       int T, int R, int S, int NF,
       int stride_d, int stride_h, int stride_w,
       int pad_d, int pad_h, int pad_w,
       int upsample_d, int upsample_h, int upsample_w,
       type ty = FPROP, bool bias = false);

  // accessors
  size_t a_size();
  size_t b_size();
  size_t c_size();
  std::vector<int32_t> c_shapes();

  // initialize
  void build_b_deltas();
  void build_deltas();
  void build_masks();
  void init(driver::stream *stream, driver::cu_module *module);
  std::array<size_t, 3> get_grid(size_t TM, size_t TN);
  void set_arg(driver::kernel *kernel,
               driver::buffer *a, driver::buffer *b, driver::buffer *c,
               driver::buffer *bias);
  void enqueue(driver::stream *stream, driver::kernel *kernel,
               driver::buffer *a, driver::buffer *b, driver::buffer *c,
               driver::buffer *bias,
               size_t TM, size_t TN, size_t nthreads);

  // utilities
  size_t get_nflops();
  std::vector<unsigned> default_params();

  // source
  std::string src(){
    std::string BS    = b_trans_ ? "[TN,TK]"      : "[TK, TN]";
    std::string bcb0  = b_trans_ ? "[:, newaxis]" : "[newaxis, :]";
    std::string bcb1  = b_trans_ ? "[newaxis, :]" : "[:, newaxis]";
    std::string ldb0  = b_trans_ ? "*ldb_s"       : "";
    std::string useb  = b_trans_ ? "trans(b)"     : "b";
    std::string flipr = b_trans_ ? ""             : "BH - 1 -";
    std::string flips = b_trans_ ? ""             : "BW - 1 -";
    std::string upar = ty_ == WGRAD ? "stride_h * ": "";
    std::string upas = ty_ == WGRAD ? "stride_w * ": "";
    std::string upah = ty_ == WGRAD ? "": "*stride_h";
    std::string upaw = ty_ == WGRAD ? "": "*stride_w";
    std::vector<std::string> crs = {"c", "r", "s"};
    std::vector<std::string> rsc = {"r", "s", "c"};
    std::vector<std::string> ax  = b_trans_ ? crs : rsc;
    std::vector<std::string> redax;
    if(b_trans_)
      redax = {"NC", "BH", "BW"};
    else
      redax = {"BH", "BW", "NC"};
    std::string inc_pb = b_lut_ ? "db" + bcb1 : "TK" + ldb0;
    std::string inc_pdb = b_trans_ ? "incd" : "TK";
    std::string a_delta_mem = is_a_deltas_cst ? "__constant__" : "";
    std::string b_delta_mem = is_b_deltas_cst_? "__constant__" : "";
    std::string masks_mem = is_mask_cst_? "__constant__" : "";

    std::string res =
        R"(
  const tunable int32 TM = {16, 32, 64};
  const tunable int32 TN = {16, 32, 64};
  const tunable int32 TK = {8};
  )";
  if(is_a_deltas_cst)
    res += "__constant__ int32* delta = alloc_const int32[" + std::to_string(h_a_deltas_.size()) + "];\n";
  if(b_lut_ && is_b_deltas_cst_)
    res += "__constant__ int32* b_delta = alloc_const int32[" + std::to_string(h_b_deltas_.size()) + "];\n";
  if(is_mask_cst_)
    res += "__constant__ int32* masks = alloc_const int32[" + std::to_string(h_masks_.size()) + "];\n";
  res += R"(

   void conv(read_only restrict fp32 *a,
             read_only restrict fp32 *b,
             fp32 *c,
             fp32 *bias,
             int32 M, int32 N, int32 K,
             int32 AH, int32 AW,
             int32 BH, int32 BW,
             int32 CH, int32 CW,
             int32 NC,
             int32 lda_n, int32 lda_c, int32 lda_d, int32 lda_h, int32 lda_w,
             int32 ldb_c, int32 ldb_t, int32 ldb_r, int32 ldb_s, int32 ldb_k,
             int32 ldc_n, int32 ldc_k, int32 ldc_m, int32 ldc_p, int32 ldc_q,
             int32 pad_h, int32 pad_w,
             int32 stride_h, int32 stride_w,
             int32 upsample_h, int32 upsample_w,
             int32 off_uh, int32 off_uw)";
  if(!is_a_deltas_cst)
    res += ", int32* delta";
  if(b_lut_ && !is_b_deltas_cst_)
    res += ", int32* b_delta";
  if(!is_mask_cst_)
    res += ", int32* masks";
   res += R"(){
    int32 rxa[TM] = get_global_range[TM](0);
    int32 rb0[TN] = get_global_range[TN](1);
    int32 rka[TK] = 0 ... TK;
    int32 rkb[TK] = 0 ... TK;
    fp32 C[TM, TN] = 0;
    int32 ldlut = )" + std::to_string(Luts_) + R"(;
    int32 rabh[TM] = rxa / CW;
    int32 raw[TM] = rxa % CW;
    int32 rab[TM] = rabh / CH;
    int32 rah[TM] = rabh % CH;
    raw = (off_uw + raw)*upsample_w;
    rah = (off_uh + rah)*upsample_h;
    raw = (raw)" + upah + R"( - pad_w)/upsample_w;
    rah = (rah)" + upaw + R"( - pad_h)/upsample_h;
    int32 ra0[TM] = rab*lda_n + rah*lda_h + raw*lda_w;
    int32 ra)" + ax[0] + ax[1] + "[TK] = rka / " + redax[2] + R"(;
    int32 ra)" + ax[2] + "[TK] = rka %  " + redax[2] + R"(;
    int32 ra)" + ax[0] + "[TK] = ra" + ax[0] + ax[1] + " / " + redax[1] + R"(;
    int32 ra)" + ax[1] + "[TK] = ra" + ax[0] + ax[1] + " % " + redax[1] + R"(;
    rar = )" + flipr + R"( rar;
    ras = )" + flips + R"( ras;
    rar = )" + upar + R"( rar;
    ras = )" + upas + R"( ras;
    int32 ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
    fp32* pa[TM, TK] = a + ra1[newaxis, :] + ra0[:, newaxis];)";
  if(b_lut_){
   res += R"(
    int32 rb)" + ax[0] + ax[1] + "[TK] = rkb / " + redax[2] + R"(;
    int32 rb)" + ax[2] + "[TK] = rkb %  " + redax[2] + R"(;
    int32 rb)" + ax[0] + "[TK] = rb" + ax[0] + ax[1] + " / " + redax[1] + R"(;
    int32 rb)" + ax[1] + "[TK] = rb" + ax[0] + ax[1] + " % " + redax[1] + R"(;
    rbr = rbr*upsample_h + off_uh;
    rbs = rbs*upsample_w + off_uw;
    int32 rb1[TK] = rbc*ldb_c + rbr*ldb_r + rbs*ldb_s;
    )" + b_delta_mem + R"( int32* pdb[TK] = b_delta + rkb + off_uw*ldlut + off_uh*ldlut*upsample_w;
    int32 db[TK] = *pdb;)";
  }
  else{
  res += R"(
    int32 rb1[TK] = rkb)" + ldb0 + ";";
  }
  res += R"(
    fp32* pb)" + BS + " = b + rb1" + bcb1 + " + rb0" + bcb0 + R"(*ldb_k;
    )" + a_delta_mem + R"( int32* pincd[TK] = delta + rka;
    )" + a_delta_mem + R"( int32* pda[TK]  = delta + ldlut + rka + off_uw*ldlut + off_uh*ldlut*upsample_w;
    int32 da[TK] = *pda;
    int32 incd[TK] = *pincd;
    int32 maskh[TM] = pad_h + min(rah, 0) + max(rah + BH - AH, 0);
    int32 maskw[TM] = pad_w + min(raw, 0) + max(raw + BW - AW, 0);
    )" + masks_mem + R"( int32* pm[TM] = masks + ldlut + maskw*ldlut + maskh*ldlut*(2*pad_w + 1) + off_uw*ldlut*(2*pad_w+1)*(2*pad_h+1) + off_uh*ldlut*(2*pad_w+1)*(2*pad_h+1)*upsample_w;
    )" + a_delta_mem + R"( int32* pincm[TM] = delta;
    int32 incm[TM] = *pincm;
    int32 maska0[TM] = *pm;
    int32 maska1[TK] = 1 << rka;
    int1 checka[TM, TK] = (maska0[:, newaxis] & maska1[newaxis, :]) > 0;
    int1 checkb0[TN] = rb0 < N;
    int1 checkb)" + BS + " = checkb0" + bcb0 + R"(;
    fp32 a[TM, TK] = checka ? *pa : 0;
    fp32 b)" + BS + R"( = checkb ? *pb : 0;
    for(int32 k = K; k > 0; k = k - TK){
      C = dot(a, )" + useb + R"(, C);
      pa = pa + da[newaxis, :];
      pb = pb + )" + inc_pb + R"(;
      checkb = checkb && (k > TK);
      @checkb b = *pb;
      pda = pda + incd;)";
  if(b_lut_){
    res += R"(
      pdb = pdb + )" + inc_pdb + R"(;
      db = *pdb;)";
  }
    res += R"(
      pincd = pincd + incd;
      da = *pda;
      incd = *pincd;
      pm = pm + incm;
      pincm = pincm + incm;
      incm = *pincm;)";
  if((ty_ == FPROP) && (NC_ % TK_ != 0)){
    res += R"(
      rka = rka + TK;
      int32 rkac[TK] = rka / (BH*BW);
      int1 checka1[TK] = rkac < NC;)";
  }
  else{
    res += R"(
      int1 checka1[TK] = k > TK;)";
  }
    res += R"(
      maska0 = *pm;
      checka = (maska0[:, newaxis] & maska1[newaxis, :]) > 0;
      checka = checka && checka1[newaxis,:];
      a = checka ? *pa : 0;
    }
    int32 rxc[TM] = get_global_range[TM](0);
    int32 rc1[TN] = get_global_range[TN](1);
    int32 rcn[TM] = rxc / (CH*CW);
    int32 rcpq[TM] = rxc % (CH*CW);
    int32 rcp[TM] = rcpq / CW;
    int32 rcq[TM] = rcpq % CW;
    rcp = rcp * upsample_h + off_uh;
    rcq = rcq * upsample_w + off_uw;
    int1 checkc1[TN] = rc1 < N;
    int32 rc0[TM] = rcn * ldc_n + rcp * ldc_p + rcq * ldc_q;
    fp32* pc[TM, TN]  = c + rc1[newaxis, :]*ldc_k + rc0[:, newaxis];)";
  if(bias_ && ty_==FPROP){
    res += R"(
    fp32* pbias[TN] = bias + rc1;
    fp32 bias[TN] = checkc1 ? *pbias : 0;
    C = C + bias[newaxis, :];)";
  }
    res += R"(
    int1 checkc0[TM] = rxc < M;
    int1 checkc[TM, TN]  = checkc0[:, newaxis] && checkc1[newaxis, :];
    @checkc *pc = C;
  })";
    return res;
  }

  // cpu check
  template<class IN_DTYPE, class OUT_DTYPE>
  void cpu_xprop(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B);

  template<class IN_DTYPE, class OUT_DTYPE>
  void cpu_wgrad(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B);

  template<class IN_DTYPE, class OUT_DTYPE>
  void cpu_ref(OUT_DTYPE* C,  IN_DTYPE* A, IN_DTYPE* B);

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
  // striding
  int32_t stride_d_;
  int32_t stride_h_;
  int32_t stride_w_;
  // padding
  int32_t pad_d_;
  int32_t pad_h_;
  int32_t pad_w_;
  // upsampling
  int32_t upsample_d_;
  int32_t upsample_h_;
  int32_t upsample_w_;
  // equivalent matmul
  int32_t M_;
  int32_t N_;
  int32_t K_;
  // helpers
  int32_t Fs_;
  int32_t TK_;
  int32_t Luts_;
  // memory strides for A
  std::vector<int32_t> shapes_a_;
  std::vector<int32_t> ld_a_;
  // memory strides for B
  std::vector<int32_t> shapes_b_;
  std::vector<int32_t> ld_b_;
  // memory stride for C
  std::vector<int32_t> shapes_c_;
  std::vector<int32_t> ld_c_;
  // constant memory
  std::vector<int32_t> h_a_deltas_;
  std::vector<int32_t> h_b_deltas_;
  std::vector<int32_t> h_masks_;
  driver::buffer* d_a_deltas_;
  driver::buffer* d_b_deltas_;
  driver::buffer* d_masks_;
  bool is_a_deltas_cst;
  bool is_b_deltas_cst_;
  bool is_mask_cst_;
  // type
  type ty_;
  bool bias_;
  bool b_trans_;
  bool b_lut_;
  // axis index
  int32_t a_inner_idx_;
  int32_t a_outer_idx_;
  int32_t a_pix_idx_;
  int32_t b_inner_idx_;
  int32_t b_outer_idx_;
  int32_t b_pix_idx_;
  int32_t c_outer_0_idx_;
  int32_t c_outer_1_idx_;
  int32_t c_pix_idx;
};

}
}
