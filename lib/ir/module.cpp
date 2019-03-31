#include "triton/ir/basic_block.h"
#include "triton/ir/module.h"
#include "triton/ir/type.h"
#include "triton/ir/constant.h"
#include "triton/ir/function.h"

namespace triton{
namespace ir{

/* Module */
module::module(const std::string &name, context &ctx)
  : name_(name), context_(ctx), builder_(ctx) {
  sealed_blocks_.insert(nullptr);
}

ir::builder& module::get_builder() {
  return builder_;
}

ir::context& module::get_context() {
  return context_;
}

void module::set_value(const std::string& name, ir::basic_block *block, ir::value *value){
  values_[val_key_t{name, block}] = value;
}

void module::set_value(const std::string& name, ir::value *value){
  return set_value(name, builder_.get_insert_block(), value);
}

void module::set_const(const std::string& name){
  const_.insert(name);
}

void module::set_continue_fn(std::function<ir::value*()> fn) {
  continue_fn_ = fn;
}

std::function<ir::value*()> module::get_continue_fn() {
  return continue_fn_;
}

ir::phi_node* module::make_phi(ir::type *ty, unsigned num_values, ir::basic_block *block){
  basic_block::iterator insert = block->get_first_non_phi();
  if(insert != block->end()){
    builder_.set_insert_point(insert);
  }
  ir::phi_node *res = builder_.create_phi(ty, num_values);
  if(insert != block->end())
    builder_.set_insert_point(block);
  return res;
}

ir::value *module::try_remove_trivial_phis(ir::phi_node *&phi){
  // find non-self references
  std::set<ir::value*> non_self_ref;
  std::copy_if(phi->ops().begin(), phi->ops().end(), std::inserter(non_self_ref, non_self_ref.begin()),
               [phi](ir::value* op){ return  op != phi && op; });
  // non-trivial
  if(non_self_ref.size() != 1)
    return phi;
  // unique value or self-reference
  ir::value *same = *non_self_ref.begin();
  std::set<ir::user*> users = phi->get_users();
  phi->replace_all_uses_with(same);
  phi->erase_from_parent();
  for(ir::user* u: users)
  if(auto *uphi = dynamic_cast<ir::phi_node*>(u))
    if(uphi != phi)
      try_remove_trivial_phis(uphi);
  return same;
}


ir::value *module::add_phi_operands(const std::string& name, ir::phi_node *&phi){
  // already initialized
  if(phi->get_num_operands())
    return phi;
  ir::basic_block *block = phi->get_parent();
  for(ir::basic_block *pred: block->get_predecessors()){
    ir::value *value = get_value(name, pred);
    phi->add_incoming(value, pred);
  }
  return phi;
}

ir::value *module::get_value_recursive(const std::string& name, ir::basic_block *block) {
  ir::value *result;
  bool is_const = const_.find(name) != const_.end();
  auto &preds = block->get_predecessors();
  ir::type *ty = get_scope().types.at(name);
  if(block)
  if(!is_const && sealed_blocks_.find(block) == sealed_blocks_.end()){
    incomplete_phis_[block][name] = make_phi(ty, 1, block);
    result = (ir::value*)incomplete_phis_[block][name];
  }
  else if(preds.size() <= 1){
    bool has_pred = preds.size();
    result = get_value(name, has_pred?preds.front():nullptr);
  }
  else{
    result = make_phi(ty, 1, block);
    set_value(name, block, result);
    result = add_phi_operands(name, (ir::phi_node*&)result);
  }
  if(auto *phi = dynamic_cast<ir::phi_node*>(result))
    result = try_remove_trivial_phis(phi);
  set_value(name, block, result);
  return result;
}

ir::value *module::get_value(const std::string& name, ir::basic_block *block) {
  ir::basic_block* save_block = builder_.get_insert_block();
  ir::basic_block::iterator save_pt = builder_.get_insert_point();
  val_key_t key(name, block);
  if(values_.find(key) != values_.end()){
    return values_.at(key);
  }
  ir::value *result = get_value_recursive(name, block);
  builder_.set_insert_point(save_block);
  if(save_pt != save_block->end())
    builder_.set_insert_point(save_pt);
  return result;
}

ir::value *module::get_value(const std::string& name) {
  return get_value(name, builder_.get_insert_block());
}

const std::string& module::get_name() {
  return name_;
}

void module::seal_block(ir::basic_block *block){
  for(auto &x: incomplete_phis_[block]){
    add_phi_operands(x.first, x.second);
    if(get_value(x.first) == x.second)
      set_value(x.first, try_remove_trivial_phis(x.second));
  }
  sealed_blocks_.insert(block);
  incomplete_phis_[block].clear();
}

/* functions */
function *module::get_or_insert_function(const std::string &name, function_type *ty) {
  function *&fn = (function*&)symbols_[name];
  if(fn == nullptr)
    return fn = function::create(ty, global_value::external, name, this);
  return fn;
}


}
}
