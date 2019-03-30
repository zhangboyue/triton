#ifndef TDL_INCLUDE_CODEGEN_OPTIMIZE_TRANS_H
#define TDL_INCLUDE_CODEGEN_OPTIMIZE_TRANS_H

#include <tuple>
#include <vector>
#include <set>

namespace triton {

namespace ir {
  class module;
}

namespace codegen{

class optimize_trans {
public:
  optimize_trans() {}
  void run(ir::module &mod);
};


}
}

#endif
