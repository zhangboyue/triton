#include <sstream>
#include "triton/dnn/base.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"

namespace triton{
namespace dnn{




void base::set_ld(const std::vector<int32_t>& shapes,
                  std::vector<int32_t>& ld) {
  size_t size = shapes.size();
  ld.resize(size);
  ld[size - 1] = 1;
  for(int i = size - 1; i >= 1; i--)
    ld[i - 1] = shapes[i] * ld[i];
}


base::base(const std::string& name)
  : name_(name) { }

void base::enqueue(driver::stream *stream, std::vector<driver::buffer *> args) {
  static std::map<base*, std::unique_ptr<triton::jit>, cmp_recompile> m_jit;
  bool autotune = false;
  driver::context* ctx = stream->context();
  triton::jit* jit;
  /* the current template has not already been compiled */
  if(m_jit.find(this) == m_jit.end()) {
    jit = m_jit.emplace(this->clone(), new triton::jit(ctx)).first->second.get();
    std::ostringstream oss;
    triton_c_src(oss);
    std::string src = oss.str();
    auto benchmark = [&](triton::driver::kernel* kernel,
                         triton::jit::launch_information info) {
      // launch info
      unsigned nthreads = info.num_threads;
      init_impl(stream, (triton::driver::cu_module*)kernel->module());
      enqueue_impl(stream, kernel, args, info.global_range_size, nthreads);
      stream->synchronize();
      double ts = triton::tools::bench([&](){ enqueue_impl(stream, kernel, args, info.global_range_size, nthreads); },
                        [&](){ stream->synchronize(); }, ctx->device());
      return num_flops() / ts * 1e-3;
    };
    // auto-tune and save result
    if(autotune) {
      triton::jit::tune_res_t best = jit->autotune(name_.c_str(), src.c_str(), benchmark);
      jit->add_module(name_.c_str(), src.c_str(), best.params);
    }
    else {
      jit->add_module(name_.c_str(), src.c_str(), jit->get_valid(name_.c_str(), src.c_str()));
    }
    triton::driver::kernel* kernel = jit->get_function(name_.c_str());
    init_impl(stream, (triton::driver::cu_module*)kernel->module());
  }
  /* retrieved compiled template */
  else
    jit = m_jit.at(this).get();

  /* get launch parameters */
  driver::kernel* kernel = jit->get_function(name_.c_str());
  triton::jit::launch_information info = jit->get_launch_info(name_.c_str());
  /* launch */
  enqueue_impl(stream, kernel, args,
               info.global_range_size, info.num_threads);
}

}
}