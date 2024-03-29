#ifndef TRITON_INCLUDE_LANG_DECLARATION_H
#define TRITON_INCLUDE_LANG_DECLARATION_H

#include "node.h"
#include <string>


namespace triton{


namespace ir{
  class function;
  class value;
  class type;
  class builder;
  class module;
}

namespace lang{

class expression;
class pointer;
class identifier;
class constant;
class compound_statement;
class initializer;
class declaration_specifier;


class declaration: public block_item{
public:
  declaration(node *spec, node *init)
    : spec_((declaration_specifier*)spec), init_((list<initializer*>*)init) { }

  ir::value* codegen(ir::module * mod) const;

public:
  const declaration_specifier *spec_;
  const list<initializer*> *init_;
};

// Types
class declaration_specifier: public node{
public:
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
protected:
  typedef std::vector<STORAGE_SPEC_T> storage_spec_vec_t;
  typedef const storage_spec_vec_t& storage_spec_vec_const_ref_t;

public:
  virtual ir::type* type_impl(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t storage) const = 0;

public:
  declarator(node *lhs)
    : lhs_((declarator*)lhs), ptr_(nullptr){ }

  ir::type* type(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t storage) const;

  const identifier* id() const {
    return (const identifier*)lhs_;
  }

  declarator *set_ptr(node *ptr){
    ptr_ = (pointer*)ptr;
    return this;
  }

  void set_addr_space(unsigned addr_space){
    addr_space_ = addr_space;
  }

protected:
  declarator *lhs_;
  pointer *ptr_;
  unsigned addr_space_;
};

class identifier: public declarator {
  ir::type* type_impl(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t storage) const;

public:
  identifier(char *&name): declarator(this), name_(name) { }
  const std::string &name() const;

private:
  std::string name_;
};

class pointer: public declarator{
private:
  ir::type* type_impl(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t storage) const;

public:
  pointer(node *id): declarator(id) { }
};

class tile: public declarator{
private:
  ir::type* type_impl(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t storage) const;

public:
  tile(node *id, node *shapes)
    : declarator(id), shapes_((list<expression*>*)(shapes)) { }

public:
  const list<expression*>* shapes_;
};

class function: public declarator{
private:
  ir::type* type_impl(ir::module *mod, ir::type *type, storage_spec_vec_const_ref_t storage) const;

public:
  function(node *id, node *args)
    : declarator(id), args_((list<parameter*>*)args) { }

  void bind_parameters(ir::module *mod, ir::function *fn) const;
  unsigned get_num_args() const { return args_->values().size(); }
  parameter* get_arg(unsigned i) const { return args_->values().at(i); }

public:
  const list<parameter*>* args_;
};


class initializer : public declarator{
private:
  ir::type* type_impl(ir::module * mod, ir::type *type, storage_spec_vec_const_ref_t storage) const;

public:
  initializer(node *decl, node *init)
  : declarator((node*)((declarator*)decl)->id()),
    decl_((declarator*)decl), expr_((expression*)init){ }

  void set_specifier(const declaration_specifier *spec);
  ir::value* codegen(ir::module *) const;

public:
  const declaration_specifier *spec_;
  declarator *decl_;
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

}

}

#endif
