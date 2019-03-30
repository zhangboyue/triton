#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/codegen/optimize_trans.h"

namespace triton {
namespace codegen{

//copy-to-shared(trans(x)) -> trans(x)
//trans(copy-to-shared(x)) -> trans(x)
void optimize_trans::replace_cts(ir::trans_inst* trans, ir::value* value,
                                 std::vector<ir::instruction*>& to_delete,
                                 ir::builder& builder){
  if(auto cts = dynamic_cast<ir::copy_to_shared_inst*>(value)){
    builder.set_insert_point(trans);
    ir::value *new_trans = builder.create_trans(trans->get_operand(0), trans->get_name());
    trans->replace_all_uses_with(new_trans);
    cts->replace_all_uses_with(new_trans);
    to_delete.push_back(cts);
    to_delete.push_back(trans);
  }
  if(auto phi = dynamic_cast<ir::phi_node*>(value))
  for(unsigned n = 0; n < phi->get_num_incoming(); n++){
    replace_cts(trans, phi->get_incoming_value(n), to_delete, builder);
  }
}


void optimize_trans::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  std::vector<ir::instruction*> to_delete;
  // iterate
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction* i: block->get_inst_list()){
    // filter transposition
    if(auto trans = dynamic_cast<ir::trans_inst*>(i)) {
      auto users = trans->get_users();
      auto ops = trans->ops();
      if(users.size() > 1 || ops.size() > 1)
        continue;
      ir::user* user = *users.begin();
      ir::value* op = *ops.begin();
      std::cout << "op: " << typeid(*op).name() << std::endl;
      std::cout << "user: " << typeid(*user).name() << std::endl;
      //copy-to-shared(trans(x)) -> trans(x)
      replace_cts(trans, user, to_delete, builder);
      //trans(copy-to-shared(x)) -> trans(x)
      replace_cts(trans, op, to_delete, builder);
    }
  }
  std::cout << "optimizing trans " << to_delete.size() << std::endl;
  // erase dead code
  for(ir::instruction* i: to_delete)
    i->erase_from_parent();
}

}
}
