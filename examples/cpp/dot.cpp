#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

const char* src =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {8};
const tunable int32 GZ = {2};

void matmul(restrict read_only fp32 *A, restrict read_only fp32 *B, fp32 *C,
           int32 M, int32 N, int32 K){
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rz[1] = get_global_range[1](2);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 c[TM, TN] = 0;
  fp32* pa[TM, TK] = A + rka[newaxis, :]*M + rxa[:, newaxis];
  fp32* pb[TN, TK] = B + rkb[newaxis, :]*K + ryb[:, newaxis];
  fp32 a[TM, TK] = *pa;
  fp32 b[TN, TK] = *pb;
  int32 last_a = ((M*K - 1) - (TM*TK + 1)) / M;
  int32 last_b = ((N*K - 1) - (TN*TK + 1)) / N;
  last_a = last_a / TK * TK;
  last_b = last_b / TK * TK;
  int32 bound = K - max(last_a, last_b);
  for(int32 k = K; k > bound; k = k - TK){
    c = dot(a, trans(b), c);
    pa = pa + TK*M;
    pb = pb + TK*K;
    a = *pa;
    b = *pb;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  for(int32 k = bound; k > 0; k = k - 1){
    int1 checka[TM, 1] = rxc[:, newaxis] < M;
    int1 checkb[TN, 1] = ryc[:, newaxis] < N;
    fp32* pa[TM, 1] = A + (K-k)*M + rxc[:, newaxis];
    fp32* pb[TN, 1] = B + (K-k)*K + ryc[:, newaxis];
    fp32 a[TM, 1] = checka ? *pa : 0;
    fp32 b[TN, 1] = checkb ? *pb : 0;
    c = dot(a, trans(b), c);
  }
  fp32* pc[TM, TN] = C + ryc[newaxis, :]*M + rxc[:, newaxis];
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  @checkc *pc = c;
}
)";

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::jit jit(context);

  // matrix multiplication parameters
  int32_t M = 512, N = 512, K = 512;
  std::vector<float> hc(M*N);
  std::vector<float> rc(M*N);
  std::vector<float> ha(M*K);
  std::vector<float> hb(K*N);
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


  // benchmark a given matrix multiplication kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, 1};
    // set argument
    kernel->setArg(0, da);
    kernel->setArg(1, db);
    kernel->setArg(2, dc);
    kernel->setArg(3, M);
    kernel->setArg(4, N);
    kernel->setArg(5, K);
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


  // just-in-time compile source-code
  std::vector<unsigned> params = {
    16, 2, 64, 16, 2, 64, 16, 8, 2, 2, 8, 8, 8
  };
//  jit.autotune("matmul",src, benchmark);
  jit.add_module("matmul", src, params);
  triton::driver::kernel* kernel = jit.get_function("matmul");
  triton::jit::launch_information info = jit.get_launch_info("matmul");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;
  stream->read(dc, true, 0, hc);
  simple_gemm<float,false,true>(rc, ha, hb, M, N, K);
  for(size_t i = 0; i < M*N; i++)
    if(std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;
}
