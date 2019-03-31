#include <cstring>
#include <cstdio>
#include "common.hpp"
#include "triton/jit.h"
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"

std::vector<int> deltas_;
int nlut = 0;

std::string src =
R"(
__constant__ int32* delta = alloc_const int32[)" + std::to_string(deltas_.size()) + R"(];

const tunable int32 TM;
const tunable int32 TN;
const tunable int32 TK;

void (read_only restrict fp32 *a,
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
    __constant__ int32* pd[TK] = delta + )" + std::to_string(nlut) + R"( + rka;
    int32 d[TK] = *pd;
    int32 incd[TK] = *pincd;
    for(int32 k = K; k > 0;){
      C = dot(a, b, C);
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
  triton::jit jit(context);
}
