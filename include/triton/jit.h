#ifndef TDL_INCLUDE_JIT_H
#define TDL_INCLUDE_JIT_H

#include <string>
#include <memory>
#include "llvm/IR/LLVMContext.h"
#include "triton/ir/context.h"
#include "triton/ir/print.h"
#include "triton/driver/module.h"
#include "triton/driver/kernel.h"
#include "triton/codegen/selection.h"
#include "triton/codegen/tune.h"
#include "triton/codegen/shmem_allocation.h"
#include "triton/codegen/shmem_liveness.h"
#include "triton/codegen/shmem_info.h"
#include "triton/codegen/shmem_barriers.h"
#include "triton/codegen/target.h"
#include "triton/codegen/vectorize.h"
#include <functional>

namespace llvm {
  class Module;
}

namespace triton {

namespace codegen{
class tune;
}

namespace ir {
class module;
class context;
class metaparameter;
}

class jit {
public:
  struct launch_information{
    std::vector<unsigned> global_range_size;
    unsigned num_threads;
  };
  typedef std::function<double(driver::kernel*, launch_information)> benchmark_t;

  struct passes_wrapper {
    passes_wrapper(codegen::target* target)
                    : shmem_liveness(&shmem_info),
                      shmem_allocation(&shmem_liveness, &shmem_info),
                      shmem_barriers(&shmem_allocation, &shmem_info),
                      vectorize(&tune),
                      selection(&shmem_allocation, &tune, &shmem_info, target),
                      target_(target) { }

    void init(ir::module &module) {
      if(target_->is_gpu()){
        shmem_info.run(module);
        shmem_liveness.run(module);
        shmem_allocation.run();
        shmem_barriers.run(module);
      }
//      vectorize.run(module);
      ir::print(module, std::cout);
    }

    codegen::tune tune;
    codegen::shmem_info shmem_info;
    codegen::shmem_liveness shmem_liveness;
    codegen::shmem_allocation shmem_allocation;
    codegen::shmem_barriers shmem_barriers;
    codegen::vectorize vectorize;
    codegen::selection selection;
    codegen::target* target_;
  };

private:
  std::string compute_data_layout(bool is_64bit = true, bool use_short_pointers = true);
  std::unique_ptr<llvm::Module> make_llvm_module(triton::ir::module &module, passes_wrapper &passes);
  std::unique_ptr<ir::module> make_triton_module(const std::string &src);

public:
  jit(driver::context* context);
  void autotune(const std::string &src, benchmark_t benchmark);
  void add_module(ir::module &module, const std::vector<unsigned>& params = {});
  void add_module(const std::string &src, const std::vector<unsigned>& params = {});
  driver::kernel* get_function(const std::string &name);
  launch_information get_launch_info(const std::string &name);
  unsigned get_int(const std::string &name);

private:
  std::vector<driver::module*> modules_;
  driver::context* driver_context_;
  llvm::LLVMContext llvm_context_;
  ir::context triton_context_;
  std::map<std::string, launch_information> launch_info_map_;
  std::map<std::string, unsigned> global_ints_;
  std::unique_ptr<triton::codegen::target> target_;
};


}

#endif