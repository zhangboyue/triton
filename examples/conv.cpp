#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

std::string src =
R"(
const tunable int32 TM = {16, 32, 64};
const tunable int32 TN = {16, 32, 64};
const tunable int32 TK = {8};

__constant__ int32* delta = alloc_const int32[18];

void matmul(read_only restrict fp32 *a,
          read_only restrict fp32 *b,
          fp32 *c,
          int32 M, int32 N, int32 K,
          int32 AN, int32 AH, int32 AW,
          int32 CN, int32 CK, int32 CP, int32 CQ,
          int32 AC, int32 AR, int32 AS,
          int32 lda_n, int32 lda_c, int32 lda_h, int32 lda_w,
          int32 ldc_n, int32 ldc_k, int32 ldc_p, int32 ldc_q,
          int32 bound){
    int32 rxa[TM] = get_global_range[TM](0);
    int32 rb0[TN] = get_global_range[TN](1);
    int32 rka[TK] = 0 ... TK;
    int32 rb1[TK] = 0 ... TK;
    fp32 C[TM, TN] = 0;
    int32 ranh[TM] = rxa / CQ;
    int32 raw[TM] = rxa % CQ;
    int32 ran[TM] = ranh / CP;
    int32 rah[TM] = ranh % CP;
    int32 ra0[TM] = ran*lda_n + rah*lda_h + raw*lda_w;
    int32 racr[TK] = rka / AS;
    int32 ras[TK] = rka % AS;
    int32 rac[TK] = racr / AR;
    int32 rar[TK] = racr % AR;
    int32 ra1[TK] = rac*lda_c + rar*lda_h + ras*lda_w;
    fp32* pa[TM, TK] = a + ra1[newaxis, :] + ra0[:, newaxis];
    fp32* pb[TN, TK] = b + rb1[newaxis, :]*CK + rb0[:, newaxis];
    fp32 a[TM, TK] = *pa;
    fp32 b[TN, TK] = *pb;
    __constant__ int32* pincd[TK] = delta + rka;
    __constant__ int32* pd[TK] = delta + 9 + rka;
    int32 d[TK] = *pd;
    int32 incd[TK] = *pincd;
    for(int32 k = K; k > 0;){
      C = dot(a, trans(b), C);
      k = k - TK;
      pb = pb + TK*CK;
      pa = pa + d[newaxis, :];
      int1 checka[TM, TK] = k > bound;
      int1 checkb[TN, TK] = k > bound;
      @checka a = *pa;
      @checkb b = *pb;
      pd = pd + incd;
      pincd = pincd + incd;
      d = *pd;
      incd = *pincd;
      if(k > bound)
        continue;
      int1 checka0[TM] = rxa < M;
      int1 checka1[TK] = rka < k;
      int1 checkb0[TN] = rb0 < N;
      int1 checkb1[TK] = rb1 < k;
      checka = checka0[:, newaxis] && checka1[newaxis, :];
      checkb = checkb0[:, newaxis] && checkb1[newaxis, :];
      a = checka ? *pa : 0;
      b = checkb ? *pb : 0;
    }
    int32 rxc[TM] = get_global_range[TM](0);
    int32 rc1[TN] = get_global_range[TN](1);
    int32 rcn[TM] = rxc / (CP*CQ);
    int32 rcpq[TM] = rxc % (CP*CQ);
    int32 rc0[TM] = rcn * ldc_n + rcpq;
    fp32* pc[TM, TN]  = c + rc1[newaxis, :]*ldc_k + rc0[:, newaxis];
    int1 checkc0[TM] = rxc < M;
    int1 checkc1[TN] = rc1 < N;
    int1 checkc[TM, TN]  = checkc0[:, newaxis] && checkc1[newaxis, :];
    @checkc *pc = C;
})";



int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  // initialize just-in-time compiler
  triton::jit jit(context);
  // initialization
  int32_t AN = 4, CK = 32;
  int32_t AD = 1, AH = 24, AW = 240;
  int32_t BC = 16, BT = 1, BR = 3, BS = 3;
  int32_t pad_d = 0, pad_h = 0, pad_w = 0;
  int32_t stride_d = 1, stride_h = 1, stride_w = 1;
  int32_t upsample_d = 1, upsample_h = 1, upsample_w = 1;
  int32_t CM = (AD*upsample_d - BT + 1 + 2*pad_d + stride_d - 1)/stride_d;
  int32_t CP = (AH*upsample_h - BR + 1 + 2*pad_h + stride_h - 1)/stride_h;
  int32_t CQ = (AW*upsample_w - BS + 1 + 2*pad_w + stride_w - 1)/stride_w;
  // equivalent matmul dimensions
  int32_t M = AN*CM*CP*CQ;
  int32_t N = CK;
  int32_t K = BC*BT*BR*BS;
  std::vector<float> hc(AN*CP*CQ*CK);
  std::vector<float> rc(AN*CP*CQ*CK);
  std::vector<float> ha(AN*BC*AH*AW);
  std::vector<float> hb(BC*BR*BS*CK);
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*4);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*4);
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*4);
  triton::driver::stream* stream = triton::driver::stream::create(context);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  // memory strides for data
  int32_t stride_i_w = 1;
  int32_t stride_i_h = AW*stride_i_w;
  int32_t stride_i_d = AH*stride_i_h;
  int32_t stride_i_c = AD*stride_i_d;
  int32_t stride_i_n = BC*stride_i_c;
  // memory strides for filters
  int32_t stride_f_k = 1;
  int32_t stride_f_s = CK*stride_f_k;
  int32_t stride_f_r = BS*stride_f_s;
  int32_t stride_f_t = BR*stride_f_r;
  int32_t stride_f_c = BT*stride_f_t;
  // memory stride for activations
  int32_t stride_o_q = 1;
  int32_t stride_o_p = CQ*stride_o_q;
  int32_t stride_o_m = CP*stride_o_p;
  int32_t stride_o_k = CM*stride_o_m;
  int32_t stride_o_n = CK*stride_o_k;
  std::vector<int> h_delta = build_conv_lut(8, stride_i_d, stride_i_h, stride_i_w, stride_i_c, BT, BR, BS);
  // benchmark a given convolution kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned TK = jit.get_int("TK");
    // initialize constant memory
    triton::driver::buffer* delta = jit.get_buffer("delta");
    stream->write(delta, false, 0, h_delta.size()*4, h_delta.data());
    stream->synchronize();
    // launch info
    unsigned nthreads = info.num_threads;
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, 1};
    // fast bounds-checking
    unsigned lasti = (grid[0]*TM - 1)*TM + TM - 1;
    unsigned lastj = (grid[1]*TN - 1)*TN + TN - 1;
    unsigned lastk = TK - 1;
    bool AT = false;
    bool BT = true;
    unsigned last_safe_a = (AT==false)?(M*K - 1 - lasti)/M - lastk : M*K - 1 - lasti*K - lastk;
    unsigned last_safe_b =  (BT==true)?(N*K - 1 - lastj)/N - lastk : N*K - 1 - lastj*K - lastk;
    int32_t bound = std::max<unsigned>(1, std::max(K - last_safe_a, K - last_safe_b));
    // set arguments
    kernel->setArg(0, da);
    kernel->setArg(1, db);
    kernel->setArg(2, dc);
    kernel->setArg(3, M);
    kernel->setArg(4, N);
    kernel->setArg(5, K);
    kernel->setArg(6, AN);
    kernel->setArg(7, AH);
    kernel->setArg(8, AW);
    kernel->setArg(9, AN);
    kernel->setArg(10, CK);
    kernel->setArg(11, CP);
    kernel->setArg(12, CQ);
    kernel->setArg(13, BC);
    kernel->setArg(14, BR);
    kernel->setArg(15, BS);
    kernel->setArg(16, stride_i_n);
    kernel->setArg(17, stride_i_c);
    kernel->setArg(18, stride_i_h);
    kernel->setArg(19, stride_i_w);
    kernel->setArg(20, stride_o_n);
    kernel->setArg(21, stride_o_k);
    kernel->setArg(22, stride_o_p);
    kernel->setArg(23, stride_o_q);
    kernel->setArg(24, bound);
    // dry run
    stream->enqueue(kernel, grid, {nthreads, 1, 1});
    stream->synchronize();
    // benchmark
    double ts = bench([&](){stream->enqueue(kernel, grid, {nthreads, 1, 1});},
                      [&](){ stream->synchronize(); }, *context->device());
    ts = ts * 1e-9;
    double tflops = 2.*M*N*K / ts * 1e-12;
    return tflops;
  };
  // run
  std::vector<unsigned> params = {
    16, 2, 64,
    32, 2, 64,
    16, 8, 2, 2,
    8, 8,
    4
  };
//  jit.autotune(src, benchmark);
  jit.add_module(src, params);
  triton::driver::kernel* kernel = jit.get_function("matmul");
  triton::jit::launch_information info = jit.get_launch_info("matmul");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;
  stream->read(dc, true, 0, hc);
  cpp_conv_nchw(BC, AN, CK, AD, AH, AW, BT, BR, BS, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, CM, CP, CQ, rc, ha, hb);
  for(size_t i = 0; i < M*N; i++)
    if(std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;
}
