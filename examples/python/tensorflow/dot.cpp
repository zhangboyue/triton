#include <iostream>

#include "triton/driver/buffer.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/runtime/jit.h"

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;


const char* src =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {8};
const tunable int32 GZ = {1};

void matmul(restrict read_only fp32 *A, restrict read_only fp32 *B, fp32 *C,
           int32 M, int32 N, int32 K,
           int32 lda, int32 ldb, int32 ldc,
           int32 *locks, int32 grid0, int32 grid1) {
  int32 rxa[TM] = get_global_range[TM](0);
  int32 ryb[TN] = get_global_range[TN](1);
  int32 rz = get_global_range[1](2);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 c[TM, TN] = 0;
  int32 div = K / GZ;
  int32 rem = K % GZ;
  K = select(rz < rem, div - 1, div);
  int32 offk = select(rz < rem, rz*(div + 1), rz*div + rem);
  fp32* pa[TM, TK] = A + (offk + rka[newaxis, :])*lda + rxa[:, newaxis];
  fp32* pb[TN, TK] = B + (offk + rkb[newaxis, :])*ldb + ryb[:, newaxis];
  fp32 a[TM, TK] = *pa;
  fp32 b[TN, TK] = *pb;
  int32 last_a = ((M*K - 1) - (TM*TK + 1)) / lda;
  int32 last_b = ((K*N - 1) - (TN*TK + 1)) / ldb;
  last_a = last_a / TK * TK;
  last_b = last_b / TK * TK;
  int32 bound = K - max(last_a, last_b);
  for(int32 k = K; k > bound; k = k - TK){
    c = dot(a, trans(b), c);
    pa = pa + TK*lda;
    pb = pb + TK*ldb;
    a = *pa;
    b = *pb;
  }
  int32 rxc[TM] = get_global_range[TM](0);
  int32 ryc[TN] = get_global_range[TN](1);
  for(int32 k = bound; k > 0; k = k - 1){
    int1 checka[TM, 1] = rxc[:, newaxis] < M;
    int1 checkb[TN, 1] = ryc[:, newaxis] < N;
    fp32* pa[TM, 1] = A + (offk + K - k)*lda + rxc[:, newaxis];
    fp32* pb[TN, 1] = B + (offk + K - k)*ldb + ryc[:, newaxis];
    fp32 a[TM, 1] = checka ? *pa : 0;
    fp32 b[TN, 1] = checkb ? *pb : 0;
    c = dot(a, trans(b), c);
  }
  int32 ridx = get_range_id(0);
  int32 ridy = get_range_id(1);
  fp32* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  int32 *plock = locks + ridx + ridy*grid0;
  while(__atomic_cas(plock, 0, 1));
  int32 *pcount = plock + grid0*grid1;
  int32 count = *pcount;
  int32 countp1 = select(count == GZ - 1, 0, count + 1);
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  if(count == 0) {
    @checkc *pc = c;
    *pcount = countp1;
  }
  else {
    @checkc *pc = c + *pc;
    *pcount = countp1;
  }
  __atomic_cas(plock, 1, 0);
}
)";

REGISTER_OP("Dot")
    .Input("a: T")
    .Input("b: T")
    .Input("locks: int32")
    .Output("c: T")
    .Attr("T: {float}")
;

class BlockSparseGemmOp : public OpKernel {
 public:
  explicit BlockSparseGemmOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context){
    // get device/stream
    GPUDevice device =  context->eigen_device<GPUDevice>();
    triton::driver::cu_stream sstream(device.stream(), false);
    triton::driver::context* ctx = sstream.context();
    triton::driver::stream* stream = &sstream;
    // get inputs
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& locks = context->input(2);
    // get shapes
    const int32_t M = a.dim_size(0);
    const int32_t N = b.dim_size(0);
    const int32_t K = a.dim_size(1);
    // allocate output
    Tensor* c = nullptr;
    TensorShape out_shape({(int64)M, (int64)N});
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &c));
    // return early if possible
    if (out_shape.num_elements() == 0)
      return;
    // initialize default compute device
    triton::jit jit(ctx);
    // matrix multiplication parameters
    triton::driver::cu_buffer da(ctx, (CUdeviceptr)a.flat<float>().data(), false);
    triton::driver::cu_buffer db(ctx, (CUdeviceptr)b.flat<float>().data(), false);
    triton::driver::cu_buffer dc(ctx, (CUdeviceptr)c->flat<float>().data(), false);
    triton::driver::cu_buffer dlocks(ctx, (CUdeviceptr)locks.flat<int32_t>().data(), false);
    // just-in-time compile source-code
    jit.add_module("matmul", src, {16, 2, 64, 16, 2, 64, 16, 8, 2, 2, 8, 8, 8, 1});
    triton::driver::kernel* kernel = jit.get_function("matmul");
    triton::jit::launch_information info = jit.get_launch_info("matmul");
    // launch info
    unsigned TM = info.global_range_size[0];
    unsigned TN = info.global_range_size[1];
    unsigned nthreads = info.num_threads;
    unsigned GZ = jit.get_int("GZ");
    std::array<size_t, 3> grid = {(M + TM - 1)/TM, (N + TN - 1)/TN, GZ};
    // set argument
    kernel->setArg(0, *da.cu());
    kernel->setArg(1, *db.cu());
    kernel->setArg(2, *dc.cu());
    kernel->setArg(3, M);
    kernel->setArg(4, N);
    kernel->setArg(5, K);
    kernel->setArg(6, M);
    kernel->setArg(7, N);
    kernel->setArg(8, M);
    kernel->setArg(9, *dlocks.cu());
    kernel->setArg(10, grid[0]);
    kernel->setArg(11, grid[1]);
    stream->enqueue(kernel, grid, {nthreads, 1, 1});
  }

private:
};

REGISTER_KERNEL_BUILDER(Name("Dot").Device(DEVICE_GPU).TypeConstraint<float>("T"), BlockSparseGemmOp);
