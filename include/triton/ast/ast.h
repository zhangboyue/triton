#ifndef TDL_INCLUDE_AST_H
#define TDL_INCLUDE_AST_H

#include "parser.hpp"
#include <cassert>
#include <vector>
#include <string>
#include <iostream>


namespace triton{


namespace ir{
  class function;
  class value;
  class type;
  class builder;
  class module;
}

namespace ast{

// Enumerations
enum ASSIGN_OP_T{
  ASSIGN,
  INPLACE_MUL, INPLACE_DIV, INPLACE_MOD,
  INPLACE_ADD, INPLACE_SUB,
  INPLACE_LSHIFT, INPLACE_RSHIFT,
  INPLACE_AND, INPLACE_XOR,
  INPLACE_OR
};

enum BIN_OP_T{
  MUL, DIV, MOD,
  ADD, SUB,
  LEFT_SHIFT, RIGHT_SHIFT,
  LT, GT,
  LE, GE,
  EQ, NE,
  AND, XOR, OR,
  LAND, LOR
};

enum UNARY_OP_T{
  INC, DEC,
  PLUS, MINUS,
  ADDR, DEREF,
  COMPL, NOT
};

enum TYPE_T{
  VOID_T,
  UINT1_T, UINT8_T, UINT16_T, UINT32_T, UINT64_T,
  INT1_T, INT8_T, INT16_T, INT32_T, INT64_T,
  FLOAT32_T, FLOAT64_T
};

enum STORAGE_SPEC_T{
  CONST_T,
  TUNABLE_T,
  KERNEL_T,
  READONLY_T, WRITEONLY_T,
};

class pointer;
class identifier;
class constant;

// AST
class node {
protected:
  static ir::value* explicit_cast(ir::builder &builder, ir::value *src, ir::type *dst_ty);
  static void implicit_broadcast(ir::module *mod, ir::value *&lhs, ir::value *&rhs);
  static void implicit_broadcast(ir::module *mod, ir::value *&arg, ir::type *ty);
  static void implicit_cast(ir::builder &builder, ir::value *&lhs, ir::value *&rhs,
                            bool &is_float, bool &is_ptr, bool &is_int, bool &is_signed);
public:
  virtual ir::value* codegen(ir::module *) const { return nullptr; }
};

template<class T>
class list: public node {
public:
  list(const T& x): values_(1, x) {}

  node* append(const T& x){
    values_.push_back(x);
    return this;
  }

  ir::value* codegen(ir::module * mod) const{
    for(T x: values_){
      x->codegen(mod);
    }
    return nullptr;
  }

  const std::vector<T> &values() const
  { return values_; }

private:
  std::vector<T> values_;
};

enum slice_enum_t{
  ALL,
  NEWAXIS
};

class slice: public node{
public:
  slice(slice_enum_t type)
    : type_(type){}

  slice_enum_t type() const{
    return type_;
  }

public:
  const slice_enum_t type_;
};

class named_expression;

class expression: public node{
public:
  virtual ir::value* codegen(ir::module *) const = 0;
  named_expression *lvalue() const { return lvalue_; }

protected:
  named_expression *lvalue_;
};

class postfix_expression: public expression{

};

class builtin_expression: public node{

};


class get_global_range: public builtin_expression{
public:
  get_global_range(node *size, node *axis): size_((constant*)size), axis_((constant*)axis) { }
  ir::value* codegen(ir::module *) const;

private:
  const constant* size_;
  const constant* axis_;
};

class matmul_expression: public builtin_expression{
public:
  matmul_expression(node* A, node *B, node *C):
    A_((expression*)A), B_((expression*)B), C_((expression*)C) { }
  ir::value* codegen(ir::module *) const;

private:
  const expression *A_;
  const expression *B_;
  const expression *C_;
};


class indexing_expression: public postfix_expression{
public:
  indexing_expression(node *id, node *slices)
    : id_((const identifier*)id), slices_((const list<slice*>*)slices) {}

  ir::value* codegen(ir::module *) const;

private:
  const identifier* id_;
  const list<slice*>* slices_;
};

class named_expression: public expression {
public:
  named_expression(node *id): id_((const identifier*)id) { lvalue_ = this; }
  const identifier *id() const { return id_; }
  ir::value* codegen(ir::module * mod) const;

private:
  const identifier *id_;
};

class binary_operator: public expression{
private:
  ir::value* llvm_op(ir::module *mod, ir::builder &bld, ir::value *lhs, ir::value *rhs, const std::string &name) const;

public:
  binary_operator(BIN_OP_T op, node *lhs, node *rhs)
    : op_(op), lhs_((expression*)lhs), rhs_((expression*)rhs) {
  }
  ir::value* codegen(ir::module *) const;

private:
  const BIN_OP_T op_;
  const expression *lhs_;
  const expression *rhs_;
};


class constant: public expression{
public:
  constant(int value): value_(value) { }
  ir::value* codegen(ir::module *mod) const;
  int value() const;

private:
  const int value_;
};

class constant_range: public expression {
public:
  constant_range(node *first, node *last)
    : first_((constant*)first), last_((constant*)last) { }

  ir::value* codegen(ir::module *mod) const;

private:
  constant *first_;
  constant *last_;
};

class string_literal: public expression{
public:
  string_literal(char *&value): value_(value) { }
  ir::value* codegen(ir::module *mod) const;

public:
  std::string value_;
};

class unary_operator: public expression{
private:
  ir::value *llvm_op(ir::builder &builder, ir::value *arg, const std::string &name) const;

public:
  unary_operator(UNARY_OP_T op, node *arg)
      : op_(op),
        arg_((expression*)arg) {
    if(op == DEREF)
      this->lvalue_ = arg_->lvalue();
  }

  UNARY_OP_T get_op() const { return op_; }
  ir::value* codegen(ir::module *mod) const;

private:
  const UNARY_OP_T op_;
  const expression *arg_;
};

class type_name;
class cast_operator: public expression{
private:
  ir::value *llvm_op(ir::builder &builder, ir::type *T, ir::value *arg, const std::string &name) const;

public:
  cast_operator(node *T, node *arg):
    T_((type_name*)T),
    arg_((expression*)arg) { }

  ir::value* codegen(ir::module *mod) const;

public:
  const type_name *T_;
  const expression *arg_;
};

class conditional_expression: public expression{
private:
  ir::value *llvm_op(ir::builder &builder,
                       ir::value *cond, ir::value *true_value, ir::value *false_value,
                       const std::string &name) const;

public:
  conditional_expression(node *cond, node *true_value, node *false_value)
    : cond_((expression*)cond),
      true_value_((expression*)true_value),
      false_value_((expression*)false_value) { }

  ir::value* codegen(ir::module *mod) const;

public:
  const expression *cond_;
  const expression *true_value_;
  const expression *false_value_;
};

class assignment_expression: public expression{
public:
  assignment_expression(node *lvalue, ASSIGN_OP_T op, node *rvalue)
    : lvalue_((named_expression*)lvalue), op_(op), rvalue_((expression*)rvalue) { }

  ir::value* codegen(ir::module *mod) const;

public:
  ASSIGN_OP_T op_;
  const expression *lvalue_;
  const expression *rvalue_;
};


class initializer;
class declaration_specifier;

class declaration: public node{
public:
  declaration(node *spec, node *init)
    : spec_((declaration_specifier*)spec), init_((list<initializer*>*)init) { }

  ir::value* codegen(ir::module * mod) const;

public:
  const declaration_specifier *spec_;
  const list<initializer*> *init_;
};

class statement: public node{

private:
  expression *pred_;
};

class expression_statement: public statement{
public:
  expression_statement(node *expr, node *mask = nullptr)
    : expr_((expression*)expr), mask_((expression*)mask){ }

  ir::value* codegen(ir::module * mod) const;

private:
  expression *expr_;
  expression *mask_;
};

class compound_statement: public statement{
  typedef list<declaration*>* declarations_t;
  typedef list<statement*>* statements_t;

public:
  compound_statement(node* decls, node* statements)
    : decls_((declarations_t)decls), statements_((statements_t)statements) {}

  ir::value* codegen(ir::module * mod) const;

private:
  declarations_t decls_;
  statements_t statements_;
};

class selection_statement: public statement{
public:
  selection_statement(node *cond, node *if_value, node *else_value = nullptr)
    : cond_(cond), then_value_(if_value), else_value_(else_value) { }

  ir::value* codegen(ir::module *mod) const;

public:
  const node *cond_;
  const node *then_value_;
  const node *else_value_;
};

class iteration_statement: public statement{
public:
  iteration_statement(node *init, node *stop, node *exec, node *statements)
    : init_(init), stop_(stop), exec_(exec), statements_(statements)
  { }

  ir::value* codegen(ir::module *mod) const;

private:
  const node *init_;
  const node *stop_;
  const node *exec_;
  const node *statements_;
};

// Jump

class jump_statement: public statement{
public:
  using statement::statement;
};

class continue_statement: public jump_statement{
public:
  ir::value* codegen(ir::module *mod) const;
};

class no_op: public statement { };

// Types
class declaration_specifier: public node{
public:
  using node::node;
  virtual ir::type* type(ir::module *mod) const = 0;
  virtual std::vector<STORAGE_SPEC_T> storage() const = 0;
};

class typed_declaration_specifier: public declaration_specifier {
public:
  typed_declaration_specifier(TYPE_T ty): ty_(ty){ }
  ir::type* type(ir::module *mod) const;
  std::vector<STORAGE_SPEC_T> storage() const;

private:
  const TYPE_T ty_;
};

class storage_declaration_specifier: public declaration_specifier {
public:
  storage_declaration_specifier(STORAGE_SPEC_T storage_spec, node *decl_spec)
    : storage_spec_(storage_spec), decl_spec_((declaration_specifier*)decl_spec) {}
  ir::type* type(ir::module *mod) const;
  std::vector<STORAGE_SPEC_T> storage() const;

private:
  const STORAGE_SPEC_T storage_spec_;
  const declaration_specifier* decl_spec_;
};

class declarator;
class parameter: public node {
public:
  parameter(node *spec, node *decl)
    : spec_((declaration_specifier*)spec),
      decl_((declarator*)decl) { }

  ir::type* type(ir::module *mod) const;
  std::vector<STORAGE_SPEC_T> storage() const;
  const identifier* id() const;

public:
  const declaration_specifier *spec_;
  const declarator *decl_;
};

/* Declarators */
class declarator: public node{
  virtual ir::type* type_impl(ir::module *mod, ir::type *type) const = 0;

public:
  declarator(node *lhs)
    : lhs_((declarator*)lhs), ptr_(nullptr){ }

  ir::type* type(ir::module *mod, ir::type *type) const;

  const identifier* id() const {
    return (const identifier*)lhs_;
  }

  declarator *set_ptr(node *ptr){
    ptr_ = (pointer*)ptr;
    return this;
  }

protected:
  declarator *lhs_;
  pointer *ptr_;
};

class identifier: public declarator {
  ir::type* type_impl(ir::module *mod, ir::type *type) const;

public:
  identifier(char *&name): declarator(this), name_(name) { }
  const std::string &name() const;

private:
  std::string name_;
};

class pointer: public declarator{
private:
  ir::type* type_impl(ir::module *mod, ir::type *type) const;

public:
  pointer(node *id): declarator(id) { }
};

class tile: public declarator{
private:
  ir::type* type_impl(ir::module *mod, ir::type *type) const;

public:
  tile(node *id, node *shapes)
    : declarator(id), shapes_((list<expression*>*)(shapes)) { }

public:
  const list<expression*>* shapes_;
};

class function: public declarator{
private:
  ir::type* type_impl(ir::module *mod, ir::type *type) const;

public:
  function(node *id, node *args)
    : declarator(id), args_((list<parameter*>*)args) { }

  void bind_parameters(ir::module *mod, ir::function *fn) const;

public:
  const list<parameter*>* args_;
};


class initializer : public declarator{
private:
  ir::type* type_impl(ir::module * mod, ir::type *type) const;

public:
  initializer(node *decl, node *init)
  : declarator((node*)((declarator*)decl)->id()),
    decl_((declarator*)decl), expr_((expression*)init){ }

  void set_specifier(const declaration_specifier *spec);
  ir::value* codegen(ir::module *) const;

public:
  const declaration_specifier *spec_;
  const declarator *decl_;
  const expression *expr_;
};


class type_name: public node{
public:
  type_name(node *spec, node * decl)
    : spec_((declaration_specifier*)spec), decl_((declarator*)decl) { }

  ir::type *type(ir::module *mod) const;

public:
  const declaration_specifier *spec_;
  const declarator *decl_;
};

/* Function definition */
class function_definition: public node{
public:
  function_definition(node *spec, node *header, node *body)
    : spec_((declaration_specifier*)spec), header_((function *)header), body_((compound_statement*)body) { }

  ir::value* codegen(ir::module * mod) const;

public:
  const declaration_specifier *spec_;
  const function *header_;
  const compound_statement *body_;
};

/* Translation Unit */
class translation_unit: public node{
public:
  translation_unit(node *item)
    : decls_(item) { }

  translation_unit *add(node *item) {
    decls_.append(item);
    return this;
  }

  ir::value* codegen(ir::module * mod) const;

private:
  list<node*> decls_;
};

}

}

#endif