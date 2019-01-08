#include "ir/basic_block.h"
#include "ir/module.h"
#include "ir/type.h"
#include "ir/constant.h"
#include "ir/function.h"

namespace tdl{
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

ir::phi_node* module::make_phi(ir::type *ty, unsigned num_values, ir::basic_block *block){
  basic_block::iterator insert = block->get_first_non_phi();
  if(*insert)
    builder_.set_insert_point(insert);
  ir::phi_node *res = builder_.create_phi(ty, num_values);
  if(*insert)
    builder_.set_insert_point(block);
  return res;
}

ir::value *module::try_remove_trivial_phis(ir::phi_node *&phi){
  // find non-self references
  std::vector<ir::value*> non_self_ref;
  std::copy_if(phi->ops().begin(), phi->ops().end(), std::back_inserter(non_self_ref),
               [phi](ir::value* op){ return  op != phi; });
  // non-trivial
  if(non_self_ref.size() > 1)
    return phi;
  // unique value or self-reference
  ir::value *same = non_self_ref[0];
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
  ir::basic_block *block = phi->get_parent();
  for(ir::basic_block *pred: block->get_predecessors()){
    ir::value *value = get_value(name, pred);
    phi->add_incoming(value, pred);
  }
  return try_remove_trivial_phis(phi);
}

ir::value *module::get_value_recursive(const std::string& name, ir::basic_block *block) {
  ir::value *result;
  auto &preds = block->get_predecessors();
  if(sealed_blocks_.find(block) == sealed_blocks_.end()){
    ir::value *pred = get_value(name, preds.front());
    incomplete_phis_[block][name] = make_phi(pred->get_type(), 1, block);
    result = (ir::value*)incomplete_phis_[block][name];
  }
  else if(preds.size() <= 1){
    bool has_pred = preds.size();
    result = get_value(name, has_pred?preds.front():nullptr);
  }
  else{
    ir::value *pred = get_value(name, preds.front());
    result = make_phi(pred->get_type(), 1, block);
    set_value(name, block, result);
    add_phi_operands(name, (ir::phi_node*&)result);
  }
  set_value(name, block, result);
  return result;
}

ir::value *module::get_value(const std::string& name, ir::basic_block *block) {
  val_key_t key(name, block);
  if(values_.find(key) != values_.end()){
    return values_.at(key);
  }
  return get_value_recursive(name, block);
}

ir::value *module::get_value(const std::string& name) {
  return get_value(name, builder_.get_insert_block());
}

void module::seal_block(ir::basic_block *block){
  for(auto &x: incomplete_phis_[block])
    add_phi_operands(x.first, x.second);
  sealed_blocks_.insert(block);
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