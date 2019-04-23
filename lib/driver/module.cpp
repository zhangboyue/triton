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

#include <iostream>
#include <fstream>
#include <memory>
#include <fstream>
#include "llvm/IR/IRBuilder.h"
#include "triton/driver/module.h"
#include "triton/driver/context.h"
#include "triton/driver/error.h"
#include "triton/tools/sys/getenv.hpp"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Linker/Linker.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/OrcMCJITReplacement.h"
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include "llvm/Transforms/Utils/Cloning.h"

namespace triton
{
namespace driver
{

/* ------------------------ */
//         Base             //
/* ------------------------ */

void module::init_llvm() {
  static bool init = false;
  if(!init){
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    init = true;
  }
}

module::module(driver::context* ctx, CUmodule mod, bool has_ownership)
  : polymorphic_resource(mod, has_ownership), ctx_(ctx) {
}

module::module(driver::context* ctx, cl_program mod, bool has_ownership)
  : polymorphic_resource(mod, has_ownership), ctx_(ctx) {
}

module::module(driver::context* ctx, host_module_t mod, bool has_ownership)
  : polymorphic_resource(mod, has_ownership), ctx_(ctx) {
}

driver::context* module::context() const {
  return ctx_;
}

module* module::create(driver::context* ctx, llvm::Module *src) {
  switch(ctx->backend()){
    case CUDA: return new cu_module(ctx, src);
    case OpenCL: return new ocl_module(ctx, src);
    case Host: return new host_module(ctx, src);
    default: throw std::runtime_error("unknown backend");
  }
}

void module::compile_llvm_module(llvm::Module* module, const std::string& triple,
                                        const std::string &proc, std::string layout,
                                        llvm::SmallVectorImpl<char> &buffer,
                                        const std::string& features,
                                        file_type_t ft) {
  init_llvm();
  // debug
  llvm::legacy::PassManager pm;
  pm.add(llvm::createPrintModulePass(llvm::outs()));
  pm.add(llvm::createVerifierPass());
  pm.run(*module);
  // create machine
  module->setTargetTriple(triple);
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  llvm::TargetMachine *machine = target->createTargetMachine(module->getTargetTriple(), proc, features, opt,
                                                             llvm::Reloc::PIC_, llvm::None, llvm::CodeGenOpt::Aggressive);
  // set data layout
  if(layout.empty())
    module->setDataLayout(machine->createDataLayout());
  else
    module->setDataLayout(layout);
  // emit machine code
  for (llvm::Function &f : module->functions())
    f.addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::legacy::PassManager pass;
  llvm::raw_svector_ostream stream(buffer);
  // convert triton file type to llvm file type
  auto ll_file_type = [&](module::file_type_t type){
    if(type == Object)
      return llvm::TargetMachine::CGFT_ObjectFile;
    return llvm::TargetMachine::CGFT_AssemblyFile;
  };
  // emit
  machine->addPassesToEmitFile(pass, stream, nullptr, ll_file_type(ft));
  pass.run(*module);
}


/* ------------------------ */
//        Host              //
/* ------------------------ */

host_module::host_module(driver::context * context, llvm::Module* src): module(context, host_module_t(), true) {
  init_llvm();
  // host info
//  std::string triple = llvm::sys::getDefaultTargetTriple();
//  std::string cpu = llvm::sys::getHostCPUName();
//  llvm::SmallVector<char, 0> buffer;
//  module::compile_llvm_module(src, triple, cpu, "", buffer, "", Assembly);

  // create kernel wrapper
  llvm::LLVMContext &ctx = src->getContext();
  llvm::Type *void_ty = llvm::Type::getVoidTy(ctx);
  llvm::Type *args_ty = llvm::Type::getInt8PtrTy(ctx)->getPointerTo();
  llvm::Type *int32_ty = llvm::Type::getInt32Ty(ctx);
  llvm::FunctionType *main_ty = llvm::FunctionType::get(void_ty, {args_ty, int32_ty, int32_ty, int32_ty}, false);
  llvm::Function* main = llvm::Function::Create(main_ty, llvm::Function::ExternalLinkage, "main", src);
  llvm::Function* fn = src->getFunction("matmul");
  llvm::FunctionType *fn_ty = fn->getFunctionType();
  std::vector<llvm::Value*> fn_args(fn_ty->getNumParams());
  std::vector<llvm::Value*> ptrs(fn_args.size() - 3);
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", main);
  llvm::IRBuilder<> ir_builder(ctx);
  ir_builder.SetInsertPoint(entry);
  for(unsigned i = 0; i < ptrs.size(); i++)
    ptrs[i] = ir_builder.CreateGEP(main->arg_begin(), ir_builder.getInt32(i));
  for(unsigned i = 0; i < ptrs.size(); i++){
    llvm::Value* addr = ir_builder.CreateBitCast(ir_builder.CreateLoad(ptrs[i]), fn_ty->getParamType(i)->getPointerTo());
    fn_args[i] = ir_builder.CreateLoad(addr);
  }
  fn_args[fn_args.size() - 3] = main->arg_begin() + 1;
  fn_args[fn_args.size() - 2] = main->arg_begin() + 2;
  fn_args[fn_args.size() - 1] = main->arg_begin() + 3;
  ir_builder.CreateCall(fn, fn_args);
  ir_builder.CreateRetVoid();


  // create execution engine
  auto cloned = llvm::CloneModule(*src);
  for(llvm::Function& fn: cloned->functions())
    hst_->functions[fn.getName()] = &fn;
  llvm::EngineBuilder builder(std::move(cloned));
  builder.setErrorStr(&hst_->error);
  builder.setMCJITMemoryManager(llvm::make_unique<llvm::SectionMemoryManager>());
  builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
  builder.setEngineKind(llvm::EngineKind::JIT);
  builder.setUseOrcMCJITReplacement(true);
  hst_->engine = builder.create();
}

/* ------------------------ */
//         OpenCL           //
/* ------------------------ */

ocl_module::ocl_module(driver::context * context, llvm::Module* src): module(context, cl_program(), true) {
  init_llvm();
  llvm::SmallVector<char, 0> buffer;
  module::compile_llvm_module(src, "amdgcn-amd-amdhsa-amdgizcl", "gfx902", "", buffer, "code-object-v3", Object);
  std::ofstream output("/tmp/tmp.o", std::ios::binary);
  std::copy(buffer.begin(), buffer.end(), std::ostreambuf_iterator<char>(output));
  system("ld.lld-8 /tmp/tmp.o -shared -o /tmp/tmp.o");
  std::ifstream input("/tmp/tmp.o", std::ios::in | std::ios::binary );
  std::vector<unsigned char> in_buffer(std::istreambuf_iterator<char>(input), {});
  size_t sizes[] = {in_buffer.size()};
  const unsigned char* data[] = {(unsigned char*)in_buffer.data()};
  cl_int status;
  cl_int err;
  *cl_ = dispatch::clCreateProgramWithBinary(*context->cl(), 1, &*context->device()->cl(), sizes, data, &status, &err);
  check(status);
  check(err);
  try{
  dispatch::clBuildProgram(*cl_, 1, &*context->device()->cl(), NULL, NULL, NULL);
  }
  catch(...){
  char log[2048];
  dispatch::clGetProgramBuildInfo(*cl_, *context->device()->cl(), CL_PROGRAM_BUILD_LOG, 1024, log, NULL);
  std::cout << log << std::endl;
  throw;
  }
}


/* ------------------------ */
//         CUDA             //
/* ------------------------ */

std::string cu_module::compile_llvm_module(llvm::Module* module) {
  // set data layout
  std::string layout = "e";
  bool is_64bit = true;
  bool use_short_pointers = true;
  if (!is_64bit)
    layout += "-p:32:32";
  else if (use_short_pointers)
    layout += "-p3:32:32-p4:32:32-p5:32:32";
  layout += "-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  // create
  llvm::SmallVector<char, 0> buffer;
  module::compile_llvm_module(module, "nvptx64-nvidia-cuda", "sm_52", layout, buffer, "", Assembly);
  return std::string(buffer.begin(), buffer.end());
}



cu_module::cu_module(driver::context * context, llvm::Module* ll_module): cu_module(context, compile_llvm_module(ll_module)) { }

cu_module::cu_module(driver::context * context, std::string const & source) : module(context, CUmodule(), true), source_(source){
  std::cout << source << std::endl;
  cu_context::context_switcher ctx_switch(*context);
  // JIT compile source-code
  CUjit_option opt[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER};
  unsigned int errbufsize = 8096;
  std::string errbuf(errbufsize, 0);
  void* optval[] = {(void*)(uintptr_t)errbufsize, (void*)errbuf.data()};
  try{
    dispatch::cuModuleLoadDataEx(&*cu_, source_.data(), 2, opt, optval);
  }catch(exception::cuda::base const &){
    std::cerr << "Compilation Failed! Log: " << std::endl;
    std::cerr << errbuf << std::endl;
    throw;
  }
}

cu_buffer* cu_module::symbol(const char *name) const{
  CUdeviceptr handle;
  size_t size;
  dispatch::cuModuleGetGlobal_v2(&handle, &size, *cu_, name);
  return new cu_buffer(ctx_, handle, false);
}


}
}

