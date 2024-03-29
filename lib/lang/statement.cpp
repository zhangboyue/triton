#include "triton/lang/expression.h"
#include "triton/lang/statement.h"
#include "triton/lang/declaration.h"
#include "triton/ir/constant.h"
#include "triton/ir/module.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/builder.h"
#include "triton/ir/type.h"

namespace triton{

namespace lang{

/* Helpers */
inline bool is_terminator(ir::value* x) {
  return x && dynamic_cast<ir::terminator_inst*>(x);
}


/* Statements */
ir::value* compound_statement::codegen(ir::module* mod) const{
  mod->add_new_scope();
  if(items_)
    items_->codegen(mod);
  mod->pop_scope();
  return nullptr;
}

/* Expression statement */
ir::value* expression_statement::codegen(ir::module *mod) const{
  ir::builder &builder = mod->get_builder();
  ir::basic_block *block = builder.get_insert_block();
  if(pred_) {
    // check that it is an assignment
    assignment_expression *assignment = dynamic_cast<assignment_expression*>(expr_);
    assert(assignment);
    // generate mask
    ir::value *pred = pred_->codegen(mod);
    ir::mask_inst *mask = (ir::mask_inst*)builder.create_mask(pred);
    // generate expression
    unsigned szbegin = block->get_inst_list().size();
    ir::value *expr = expr_->codegen(mod);
    ir::basic_block::iterator begin = block->begin();
    std::advance(begin, szbegin);
    // set mask
    ir::type *ty = expr->get_type();
    for(auto it = begin; it != builder.get_insert_point(); it++)
      (*it)->set_mask_pred(mask->get_result(0));
//    if(auto *itn = dynamic_cast<ir::instruction*>(expr))
//      itn->set_mask_pred(mask->get_result(0));
    if(ty->is_void_ty())
      return expr;
    // merge with psi
    ir::psi_inst *psi = (ir::psi_inst*)builder.create_merge(mask->get_result(0), expr,
                                                                  mask->get_result(1), ir::undef_value::get(ty));
    std::string name = ((named_expression*)assignment->lvalue())->id()->name();
    mod->set_value(name, psi);
    return psi;
  }
  return expr_->codegen(mod);
}

/* For statement */
ir::value* iteration_statement::codegen(ir::module *mod) const{
  ir::builder &builder = mod->get_builder();
  ir::context &ctx = mod->get_context();
  ir::basic_block *current_bb = builder.get_insert_block();
  ir::function *fn = current_bb->get_parent();
  ir::basic_block *loop_bb = ir::basic_block::create(ctx, "loop", fn);
  ir::basic_block *next_bb = ir::basic_block::create(ctx, "postloop", fn);
  mod->set_continue_fn([&](){
    if(exec_)
      exec_->codegen(mod);
    ir::value *cond = explicit_cast(builder, stop_->codegen(mod), ir::type::get_int1_ty(ctx));
    return builder.create_cond_br(cond, loop_bb, next_bb);
  });
  init_->codegen(mod);
  ir::value *cond = explicit_cast(builder, stop_->codegen(mod), ir::type::get_int1_ty(ctx));
  builder.create_cond_br(cond, loop_bb, next_bb);
//  builder.create_br(loop_bb);
  builder.set_insert_point(loop_bb);
  if(!is_terminator(statements_->codegen(mod)))
    mod->get_continue_fn()();
  ir::basic_block *stop_bb = builder.get_insert_block();
  mod->seal_block(stop_bb);
  mod->seal_block(loop_bb);
  mod->seal_block(builder.get_insert_block());
  mod->seal_block(next_bb);
  builder.set_insert_point(next_bb);
  return nullptr;
}

/* While statement */
ir::value* while_statement::codegen(ir::module* mod) const{
  ir::builder &builder = mod->get_builder();
  ir::context &ctx = mod->get_context();
  ir::basic_block *current_bb = builder.get_insert_block();
  ir::function *fn = current_bb->get_parent();
  ir::basic_block *loop_bb = ir::basic_block::create(ctx, "loop", fn);
  ir::basic_block *next_bb = ir::basic_block::create(ctx, "postloop", fn);
  mod->set_continue_fn([&](){
    ir::value *cond = explicit_cast(builder, cond_->codegen(mod), ir::type::get_int1_ty(ctx));
    return builder.create_cond_br(cond, loop_bb, next_bb);
  });
  ir::value *cond = explicit_cast(builder, cond_->codegen(mod), ir::type::get_int1_ty(ctx));
  builder.create_cond_br(cond, loop_bb, next_bb);
  builder.set_insert_point(loop_bb);
  if(!is_terminator(statements_->codegen(mod)))
    mod->get_continue_fn()();
  ir::basic_block *stop_bb = builder.get_insert_block();
  mod->seal_block(stop_bb);
  mod->seal_block(loop_bb);
  mod->seal_block(builder.get_insert_block());
  mod->seal_block(next_bb);
  builder.set_insert_point(next_bb);
  return nullptr;
}

/* Selection statement */
ir::value* selection_statement::codegen(ir::module* mod) const{
  ir::builder &builder = mod->get_builder();
  ir::context &ctx = mod->get_context();
  ir::function *fn = builder.get_insert_block()->get_parent();
  ir::value *cond = cond_->codegen(mod);
  ir::basic_block *then_bb = ir::basic_block::create(ctx, "then", fn);
  ir::basic_block *else_bb = else_value_?ir::basic_block::create(ctx, "else", fn):nullptr;
  ir::basic_block *endif_bb = ir::basic_block::create(ctx, "endif", fn);
  mod->seal_block(then_bb);
  if(else_value_)
    mod->seal_block(else_bb);

  // Branch
  if(else_value_)
    builder.create_cond_br(cond, then_bb, else_bb);
  else
    builder.create_cond_br(cond, then_bb, endif_bb);
  // Then
  builder.set_insert_point(then_bb);
  if(!is_terminator(then_value_->codegen(mod)))
    builder.create_br(endif_bb);
  // Else
  if(else_value_){
    builder.set_insert_point(else_bb);
    if(!is_terminator(else_value_->codegen(mod)))
      builder.create_br(endif_bb);
  }
  // Endif
  mod->seal_block(endif_bb);
  builder.set_insert_point(endif_bb);
  return nullptr;
}

/* Continue statement */
ir::value* continue_statement::codegen(ir::module *mod) const{
  return mod->get_continue_fn()();
}

}

}
