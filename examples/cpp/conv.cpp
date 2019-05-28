#include <cstring>
#include <cstdio>
#include "triton/runtime/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/dnn/conv.h"
#include "triton/tools/bench.hpp"

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::jit jit(context);
  triton::dnn::conv::type ty = triton::dnn::conv::FPROP;
  // initialization
  int32_t B = 64, NF = 64;
  int32_t D = 1, H = 8, W = 8;
  int32_t NC = 3, T = 1, R = 3, S = 3;
  int32_t pad_d = 0, pad_h = 0, pad_w = 0;
  int32_t stride_d = 1, stride_h = 1, stride_w = 1;
  int32_t upsample_d = 1, upsample_h = 1, upsample_w = 1;
  triton::dnn::conv configuration(128, 256, 1, 14, 14, 1, 5, 5, 512, 1, 1, 1, 0, 0, 0, 1, 1, 1, triton::dnn::conv::FPROP, 0);
//  triton::dnn::conv configuration(B, NC, D, H, W, T, R, S, NF, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, upsample_d, upsample_h, upsample_w, ty);
  // convolution configuration
  std::vector<float> hc(configuration.c_size());
  std::vector<float> rc(configuration.c_size());
  std::vector<float> ha(configuration.a_size());
  std::vector<float> hb(configuration.b_size());
  srand(0);
  for(size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand()/RAND_MAX;
  for(size_t i = 0; i < hc.size(); i++)
    hc[i] = 0;
  rc = hc;
  triton::driver::buffer* dc = triton::driver::buffer::create(context, hc.size()*4);
  triton::driver::buffer* da = triton::driver::buffer::create(context, ha.size()*4);
  triton::driver::buffer* db = triton::driver::buffer::create(context, hb.size()*4);
  triton::driver::stream* stream = triton::driver::stream::create(context);
  stream->write(da, true, 0, ha);
  stream->write(db, true, 0, hb);
  stream->write(dc, true, 0, hc);
  stream->synchronize();
  // benchmark a given convolution kernel
  auto benchmark = [&](triton::driver::kernel* kernel,
                       triton::jit::launch_information info) {
    configuration.init(stream, (triton::driver::cu_module*)kernel->module());
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    unsigned GZ = jit.get_int("GZ");
    configuration.enqueue(stream, kernel, da, db, dc, nullptr, TM, TN, GZ, nthreads);
    stream->synchronize();
    double ts = triton::tools::bench([&](){ configuration.enqueue(stream, kernel, da, db, dc, nullptr, TM, TN, GZ, nthreads); },
                      [&](){ stream->synchronize(); }, context->device());
    return configuration.get_nflops() / ts * 1e-3;
  };
  std::string src = configuration.src();
  triton::jit::tune_res_t best = jit.autotune("conv", src.c_str(), benchmark);
  jit.add_module("conv", src.c_str(), best.params);
//  jit.add_module("conv", src.c_str(), configuration.default_params());
  triton::driver::kernel* kernel = jit.get_function("conv");
  triton::jit::launch_information info = jit.get_launch_info("conv");
  std::cout << "Performance: " << benchmark(kernel, info) << " TFLOPS " << std::endl;
  stream->read(dc, true, 0, hc);
  configuration.cpu_ref(rc.data(), ha.data(), hb.data());
  for(size_t i = 0; i < hc.size(); i++){
    if(std::isnan(hc[i]) || std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-4){
      std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "Pass!" << std::endl;
}
