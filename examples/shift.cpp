#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

// K = channels
// M = batch * height * width
// N = number of feature maps

const char* src =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {8};

__constant__ int32* delta = alloc_const int32[256];

void shift(restrict read_only fp32 *a, restrict read_only fp32 *b, fp32 *c,
           int32 M, int32 N, int32 K){
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 C[TM, TN] = 0;
  fp32* pxa[TM, TK] = a + rxa[:, newaxis];
  fp32* pb[TN, TK] = b + rkb[newaxis, :]*K + ryb[:, newaxis];
  __constant__ int32* pd[TK] = delta + rka;
  for(int32 k = K; k > 0; k = k - TK){
    int32 delta[TK] = *pd;
    fp32 *pa[TM, TK] = pxa + delta[newaxis, :];
    fp32 a[TM, TK] = *pa;
    fp32 b[TN, TK] = *pb;
    C = dot(a, trans(b), C);
    pb = pb + TK*K;
    pd = pd + TK;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  fp32* pc[TM, TN] = c + ryc[newaxis, :]*M + rxc[:, newaxis];
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  @checkc *pc = C;
}
)";

std::vector<int32_t> shift_deltas(int32_t TK,
                                  // strides
                                  int32_t stride_w, int32_t stride_h, int32_t stride_c,
                                  // shift
                                  int32_t C,
                                  const std::vector<int32_t>& shift_h,
                                  const std::vector<int32_t>& shift_w) {
  std::vector<int32_t> res(C);
  for(unsigned c = 0; c < C; c++){
    res[c] = c*stride_c;
    res[c] += shift_h[c]*stride_h;
    res[c] += shift_w[c]*stride_w;
  }
  return res;
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  // initialize just-in-time compiler
  triton::jit jit(context);
  // initialization
  int32_t BS = 1, F = 32;
  int32_t H = 24, W = 240;
  int32_t C = 64;
  // equivalent matmul dimensions
  int32_t M = BS*H*W;
  int32_t N = F;
  int32_t K = C;
  std::vector<float> hc(BS*H*W*F);
  std::vector<float> rc(BS*H*W*F);
  std::vector<float> ha(BS*C*H*W);
  std::vector<float> hb(C*F);
  // strides
  int32_t stride_i_n = 1;
  int32_t stride_i_w = N*stride_i_n;
  int32_t stride_i_h = W*stride_i_w;
  int32_t stride_i_c = H*stride_i_h;
  // random shifts
  std::vector<int32_t> shift_h(C);
  std::vector<int32_t> shift_w(C);
  for(int32_t c = 0; c < C; c++){
    shift_h[c] = 0;
    shift_w[c] = 0;
  }
  // initialize buffers
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
  std::vector<int32_t> h_delta = shift_deltas(8, stride_i_w, stride_i_h, stride_i_c, C, shift_h, shift_w);

  // benchmark a given matrix multiplication kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    // initialize constant memory
    triton::driver::buffer* delta = jit.get_buffer("delta");
    stream->write(delta, false, 0, h_delta.size()*4, h_delta.data());
    stream->synchronize();
    // set argument
    kernel->setArg(0, da);
    kernel->setArg(1, db);
    kernel->setArg(2, dc);
    kernel->setArg(3, M);
    kernel->setArg(4, N);
    kernel->setArg(5, K);
    // dry run
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, 1};
    stream->enqueue(kernel, grid, {nthreads, 1, 1});
    stream->synchronize();
    // benchmark
    double ts = bench([&](){stream->enqueue(kernel, grid, {nthreads, 1, 1});},
                      [&](){ stream->synchronize(); }, *context->device());
    ts = ts * 1e-9;
    double tflops = 2.*M*N*K / ts * 1e-12;
    return tflops;
  };

  // shift
  std::vector<unsigned> params = {
    16, 2, 64,
    32, 2, 64,
    16, 8, 2, 2,
    8, 8,
    4
  };
  jit.add_module("shift", src, params);
  triton::driver::kernel* kernel = jit.get_function("shift");
  triton::jit::launch_information info = jit.get_launch_info("shift");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;

}
