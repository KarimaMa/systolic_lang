import numpy as np

# spaces per indent for cstrings 
INDENT = "  "

# from a github gist by victorlei
def extclass(cls):
  return lambda f: (setattr(cls,f.__name__,f) or f)
# ------------------------------------------------------------------------
# Grammatical Forms (Classes)

"""
BNF for our grammar
scalarT ::= np.int64 | np.float64 
tensorT ::= ( scalarT, int* )
expr ::=   Add( expr, expr )
       |   Access( iexpr*, str )
       |   Product( expr, expr )
       |   SumR( iexpr*, expr )
texpr ::=  Tensor ( str )
iexpr ::=  IndexConst( str )
       |   IAdd( iexpr, iexpr )
stmt  ::=  TConstruct( str ) | TAssign( str, str ) | Gen( str, iexpr*, expr )
prog  ::=  stmt* 

"""

class expr:
  def __init__(self):
    assert False, "Do not try to instantiate abstract expressions"

class iexpr:
  def __init__(self):
    assert False, "Do not try to instantiate abstract index expressions"

class texpr:
  def __init__(self):
    assert False, "Do not try to instantiate abstract tensor expressions"

class stmt:
  def __init__(self):
    assert False, "Do not try to instantiate abstract tensor expressions"

class Update:
  def __init__(self):
    assert False, "Do not try to instantiate abstract expression types"

class IndexConst(iexpr):
  def __init__(self, name):
    assert type(name) is str, "expected string"
    self.values = [name]

class Add(expr):
  def __init__(self, lhs, rhs):
    assert isinstance(lhs, expr), "expected expr on lhs of Add"
    assert isinstance(rhs, expr), "expected expr on rhs of Add"
    self.values = [lhs, rhs]

class IAdd(iexpr):
  def __init__(self, lhs, rhs):
    assert isinstance(lhs, iexpr), "expected iexpr on lhs of Add"
    assert isinstance(rhs, iexpr), "expected iexpr on rhs of Add"
    self.values = [lhs, rhs]

class Tensor(texpr):
  def __init__(self, t):
    #assert isinstance(t, np.ndarray), "expected Tensor const to be numpy array"
    assert type(t) is str, "expected Tensor name to be a string"
    self.values = [t]

class Access(expr):
  def __init__(self, index_exprs, t):
    assert type(index_exprs) is list, "expected index_exprs to be a list"
    for i,ie in enumerate(index_exprs):
      assert isinstance(ie, iexpr), f"expected elem {i} of index_exprs to be an iexpr"
    assert type(t) is str, "expected tensor reference t to be a str"
    self.values = [index_exprs, t]

class Product(expr):
  def __init__(self, lhs, rhs):
    assert isinstance(lhs, expr), "expected expr on lhs of Product"
    assert isinstance(rhs, expr), "expected expr on rhs of Product"
    self.values = [lhs, rhs]

class SumR(expr, Update):
  def __init__(self, r_vars, term):
    assert type(r_vars) is list or type(r_vars) is tuple, "expected r_vars to be a list or tuple"
    for i, rv in enumerate(r_vars):
      assert isinstance(rv, iexpr), f"expected elem {i} of r_vars to be an iexpr"
    assert isinstance(term, expr), "expected term of SumR to be an expr"
    self.values = [r_vars, term]

class Gen(texpr):
  def __init__(self, name, lhs_index_exprs, rhs):
    assert type(lhs_index_exprs) is list, "expected lhs_index_exprs to be a list"
    for i,ie in enumerate(lhs_index_exprs):
      assert isinstance(ie, iexpr), f"expected elem {i} of lhs_index_exprs to be an iexpr"
    assert isinstance(rhs, expr), "expected rhs to be an expr"
    self.values = [name, lhs_index_exprs, rhs]

class TConstruct(stmt):
  def __init__(self, t):
    assert type(t) is str, "expected tensor name to be a string"
    self.values = [t]

class TAssign(stmt):
  def __init__(self, tlhs, trhs):
    assert type(tlhs) is str, "expected lhs tensor reference to be a str"
    assert type(trhs) is str, "expected rhs tensor reference to be a str"
    self.values = [tlhs, trhs]

# ------------------------------------------------------------------------
# Visit() Pass
@extclass(IndexConst)
def visit(self, env_map):
  e = self.values[0] #char name
  e_val = env_map[e]
  assert type(e_val) is int, "Expected IndexConst to evaluate to an integer value"
  return e_val

@extclass(Add)
def visit(self, env_map):
  e1 = self.values[0].visit(env_map)
  e2 = self.values[1].visit(env_map)
  return e1 + e2

@extclass(IAdd)
def visit(self, env_map):
  e1 = self.values[0].visit(env_map)
  e2 = self.values[1].visit(env_map)
  return e1 + e2

@extclass(Tensor)
def visit(self, env_map):
  return env_map[self.values[0]][1]

@extclass(Access)
def visit(self, env_map):
  tensor = env_map[self.values[1]][1]
  # array of index expressions, which are either IndexConst or Add
  index_exprs = self.values[0] 
  index_values = tuple([index_exprs[i].visit(env_map) for i in range(len(index_exprs))])
  return tensor[index_values] 

@extclass(Product)
def visit(self, env_map):
  e1 = self.values[0].visit(env_map)
  e2 = self.values[1].visit(env_map)
  return e1 * e2

@extclass(SumR)
def visit(self, env_map):
  e = self.values[1]
  r = 0
  rvar_names = [rv.values[0] for rv in self.values[0]]
  reductions = [(env_map['ranges'][rname], rname) for rname in rvar_names]
 
  def subvisit(i, reductions, env_map):
    r = 0
    rdom, rname = reductions[i]
    for val in range(rdom[0], rdom[1]):
      env_map[rname] = val
      if i == (len(reductions) - 1):
        r += e.visit(env_map)
      else:
        r += subvisit(i+1, reductions, env_map)
    return r

  return subvisit(0, reductions, env_map)

@extclass(TConstruct)
def visit(self, env_map):
  name = self.values[0]
  tensorType = env_map[name][0]
  if tensorType[0] is np.int64:
    newArray = np.zeros(tensorType[1], np.int64)
  elif tensorType[0] is np.float64:
    newArray = np.zeros(tensorType[1], np.float64)
  
  env_map[name] += (newArray,)

@extclass(TAssign)
def visit(self, env_map):
  rhs_tensor_ref = env_map[self.values[1]][1]
  env_map[self.values[0]][1] = rhs_tensor_ref

@extclass(Gen)
def visit(self, env_map):
  name = self.values[0]
  # array of dimension sizes
  dim_sizes = env_map[name][0][1]
  
  # array of dimension names, should be type index const for now
  lhs_index_exprs = self.values[1]
  rhs = self.values[2]
  def subGen(lhs, lhs_index_exprs, dim_id, dim_sizes, rhs, env_map):
    dim_size = dim_sizes[dim_id]
    for val in range(dim_size):
      dim_name = lhs_index_exprs[dim_id].values[0]
      env_map[dim_name] = val
      if dim_id == (len(lhs_index_exprs) - 1): # all indices have been determined
        lhs_indices = tuple(lhs_index_exprs[i].visit(env_map) for i in range(len(lhs_index_exprs)))
        lhs[lhs_indices] = rhs.visit(env_map)
      else:
        subGen(lhs, lhs_index_exprs, dim_id+1, dim_sizes, rhs, env_map)

  lht = env_map[name][1]
  subGen(lht, lhs_index_exprs, 0, dim_sizes, rhs, env_map)


# ------------------------------------------------------------------------
# typecheck() Pass
@extclass(IndexConst)
def typecheck(self, env_map):
  e = self.values[0] #char name
  assert type(e) is str, "Expected IndexConst reference to be a string"
  return int # just assuming for now that it evaluates to int?? FIX THIS

@extclass(Add)
def typecheck(self, env_map):
  e1type = self.values[0].typecheck(env_map)
  e2type = self.values[0].typecheck(env_map)
  assert e1type is np.float64 or e1type is np.int64, "Expected lhs of Add to be int64 or float64"
  assert e2type is np.float64 or e2type is np.int64, "Expected rhs of Add to be int64 or float64"

  if e1type is np.float64 or e2type is np.float64:
    return np.float64
  return np.int64

@extclass(IAdd)
def typecheck(self, env_map):
  e1type = self.values[0].typecheck(env_map)
  e2type = self.values[1].typecheck(env_map)
  assert e1type is int, "Expected lhs of IAdd to be int"
  assert e2type is int, "Expected rhs of IAdd to be int"
  return int

@extclass(Tensor)
def typecheck(self, env_map):
  tensor_type = env_map[self.values[0]][0]
  tensor_dtype = tensor_type[0]
  tensor = env_map[self.values[0][1]]
  if tensor_dtype is np.int64:
    assert tensor.dtype == np.int64, f"Tensor {self.values[0]} type does not match referenced array type"
    return tensor_type
  elif tensor_dtype is np.float64:
    assert tensor.dtype == np.float64, f"Tensor {self.values[0]} type does not match referenced array type"
    return tensor_type
  else:
    assert False, f"Expected tensor {self.values[0]} to have type np.int64 or np.float64"

@extclass(Access)
def typecheck(self, env_map):
  tensor = env_map[self.values[1]][1]
  tensorType = env_map[self.values[1]][0]
  index_exprs = self.values[0] 
  assert(len(index_exprs) == tensor.ndim), "Expected length of index_exprs to match number of tensor dimensions"
  index_types = tuple([index_exprs[i].typecheck(env_map) for i in range(len(index_exprs))])
  for i, t in enumerate(index_types):
    assert t is int, f"Expected index {i} into tensor to be an int"

  return tensorType[0]

@extclass(Product)
def typecheck(self, env_map):
  e1type = self.values[0].typecheck(env_map)
  e2type = self.values[0].typecheck(env_map)
  assert e1type is np.float64 or e1type is np.int64, f"Expected lhs of Product to be int64 or float64 {e1type}"
  assert e2type is np.float64 or e2type is np.int64, f"Expected rhs of Product to be int64 or float64 {e2type}"

  if e1type is np.float64 or e2type is np.float64:
    return np.float64
  return np.int64

@extclass(SumR)
def typecheck(self, env_map):
  e = self.values[1]
  rvar_names = [rv.values[0] for rv in self.values[0]]
  reductions = [(env_map['ranges'][rname], rname) for rname in rvar_names]
  for i, (rdom, rname) in enumerate(reductions):
    assert isinstance(rdom, tuple), f"expected r_dom {rname} of SumR to be a tuple"
    assert len(rdom) == 2, f"expected r_dom {rname} of SumR to contain two values"
    for v in rdom:
      assert type(v) is int, f"expected values in r_dom {rname} of SumR to be ints"

  return e.typecheck(env_map)

@extclass(TConstruct)
def typecheck(self, env_map):
  name = self.values[0]
  tensorDType = env_map[name][0][0]
  assert tensorDType is np.int64 or tensorDType is np.float64, "Invalid tensor type in constructor"
  return None

@extclass(TAssign)
def typecheck(self, env_map):
  rhs_tensor_type = env_map[self.values[1]][0]
  lhs_tensor_type = env_map[self.values[0]][0]
  assert rhs_tensor_type == lhs_tensor_type, "Expected tensor types to match in tensor assignment"
  return None

@extclass(Gen)
def typecheck(self, env_map):
  name = self.values[0]
  rhs = self.values[2]

  # array of dimension sizes
  gen_dtype = env_map[name][0][0]
  dim_sizes = env_map[name][0][1]
  assert type(dim_sizes) is list or type(dim_sizes) is tuple, "expected dim_sizes to be a list"
  for i,ie in enumerate(dim_sizes):
    assert type(ie) is int, "expected elem {i} of dim_sizes to be an int"

  rhs_type = rhs.typecheck(env_map)
  assert gen_dtype == rhs_type, f"Gen {name} type {gen_dtype} does not match rhs type {rhs_type}"
  return gen_dtype

# ------------------------------------------------------------------------
# dump_cstr() Pass
@extclass(IndexConst)
def cstr(self, env_map):
  e = self.values[0] #char name
  return e

@extclass(IAdd)
def cstr(self, env_map):
  e1 = self.values[0].cstr(env_map)
  e2 = self.values[1].cstr(env_map)
  return "{} + {}".format(e1, e2)

@extclass(Add)
def cstr(self, env_map):
  e1 = self.values[0].cstr(env_map)
  e2 = self.values[1].cstr(env_map)
  return "{} + {}".format(e1, e2)

@extclass(Tensor)
def cstr(self, env_map):
  return self.values[0] 

@extclass(Access)
def cstr(self, env_map):
  # should be a tensor
  tname = self.values[1]
  tensor = env_map[tname][1]
  # array of index expressions, which are either IndexConst or Add
  # assuming everything is 2D, 3D, or 4D
  index_exprs = self.values[0] 
  assert(len(index_exprs) == tensor.ndim), "expected length of index_exprs to match number of tensor dimensions"
  access_index_str = ",".join([i.cstr(env_map) for i in index_exprs])
  access_str = "{}[{}]".format(tname, access_index_str)
  return access_str

@extclass(Product)
def cstr(self, env_map):
  e1 = self.values[0].cstr(env_map)
  e2 = self.values[1].cstr(env_map)
  return "{} * {}".format(e1, e2)

@extclass(SumR)
def cstr(self, env_map):
  rvar_names = [rv.values[0] for rv in self.values[0]]
  reductions = [(env_map['ranges'][rname], rname) for rname in rvar_names]
  for i, (rdom, rname) in enumerate(reductions):
    assert isinstance(rdom, tuple), f"expected r_dom {rname} of SumR to be a tuple"
    assert len(rdom) == 2, f"expected r_dom {rname} of SumR to contain two values"
    for v in rdom:
      assert type(v) is int, f"expected values in r_dom {rname} of SumR to be ints"

  def subcstr(i, reductions, env_map):
    rdom, rname = reductions[i]
    looplevel = env_map['looplevel']
    indent = ''.join([INDENT for i in range(looplevel)])
    loopstr = f""
    if i == 0:
      initial_val = "0.0"
      lhstr = env_map["lhstr"]
      loopstr += f"{indent}{lhstr} = {initial_val};\n"

    if i < len(reductions)-1:
      env_map['looplevel'] += 1
      child_loop_str = subcstr(i+1, reductions, env_map)
      loopstr += (f"{indent}for (int {rname} = {rdom[0]}; {rname} < {rdom[1]}; {rname}++) {{\n"+
                  f"{child_loop_str}\n" +
                  f"{indent}}}")
    else:
      e = self.values[1]
      lhstr = env_map["lhstr"]    
      rhstr = e.cstr(env_map)
      loopstr += (f"{indent}for (int {rname} = {rdom[0]}; {rname} < {rdom[1]}; {rname}++) {{\n"+
                  f"{indent}{INDENT}{lhstr} += {rhstr};\n"+
                  f"{indent}}}")
    return loopstr

  return subcstr(0, reductions, env_map)


@extclass(TConstruct)
def cstr(self, env_map):
  name = self.values[0]
  tensorDtype = env_map[name][0][0]
  tensorSize = env_map[name][0][1]

  sizestr = "["+"][".join(str(i) for i in tensorSize)+"]"
  if tensorDtype is np.int64:
    return f"int {name}{sizestr};"
  elif tensorDtype is np.float64:
    return f"float {name}{sizestr};"
  else:
    assert False, "Invalid tensor type in constructor"

@extclass(Gen)
def cstr(self, env_map):
  name = self.values[0]
  # array of dimension sizes
  dim_sizes = env_map[name][0][1]
  assert type(dim_sizes) is list or type(dim_sizes) is tuple, "expected dim_sizes to be a list"
  for i,ie in enumerate(dim_sizes):
    assert type(ie) is int, "expected elem {i} of dim_sizes to be an int"

  # array of dimension names, should be type index const for now
  lhs_index_exprs = self.values[1]
  rhs = self.values[2]

  def subcstr(lhs_index_exprs, dim_id, dim_sizes, rhs, env_map):
    dim_size = dim_sizes[dim_id]
    loopvar = lhs_index_exprs[dim_id].cstr(env_map)
    looplevel = env_map['looplevel']
    indent = ''.join([INDENT for i in range(looplevel)])

    if dim_id == (len(lhs_index_exprs) - 1):
      gen_name = self.values[0]

      lhs_index_str = ",".join([index.cstr(env_map) for index in lhs_index_exprs])
      if isinstance(rhs, Update):
        env_map['lhstr'] = f"{gen_name}[{lhs_index_str}]"
        env_map['looplevel'] += 1
        rhs_update_str = rhs.cstr(env_map)
        loop_str = (f""+
        f"{indent}for (int {loopvar} = 0; {loopvar} < {dim_size}; {loopvar}++) {{\n"+
        f"{rhs_update_str}\n"+
        f"{indent}}}")
      else:
        rhs_expr_str = rhs.cstr(env_map)
        loop_str = (f""+
        f"{indent}for (int {loopvar} = 0; {loopvar} < {dim_size}; {loopvar}++) {{\n"+
        f"{indent}{INDENT}{gen_name}[{lhs_index_str}] = {rhs_expr_str};\n"+
        f"{indent}}}")
    else:
      env_map['looplevel'] += 1
      child_loop_str = subcstr(lhs_index_exprs, dim_id+1, dim_sizes, rhs, env_map)
      loop_str = (f"" + 
      f"{indent}for (int {loopvar} = 0; {loopvar} < {dim_size}; {loopvar}++) {{\n" +
      f"{child_loop_str}\n" +
      f"{indent}}}")
     
    return loop_str

  return subcstr(lhs_index_exprs, 0, dim_sizes, rhs, env_map)


# ------------------------------------------------------------------------
# Tests
rht = np.random.randint(0,10,(3,3,2))
t = Tensor('T')
ttype = (np.int64, (3,3,2))
rh_indices = [IndexConst('x'), IndexConst('y'), IndexConst('z')]
access = Access(rh_indices, 'T')
lh_indices = [IndexConst('x'), IndexConst('y'), IndexConst('z')]

copytype = (np.int64, (3,3,2))
Cconstr = TConstruct('CopyT')
gen = Gen('CopyT', lh_indices, access)
env_map = {'T':(ttype, rht), 'CopyT': (copytype, )}

gen_type = gen.typecheck(env_map)
print(f"gen type {gen_type}")

Cconstr.visit(env_map)
gen.visit(env_map)
print(f"{rht}")
print(env_map['CopyT'][1])


print("---------- Testing MATMUL ------------")
At = Tensor('A')
Bt = Tensor('B')
Atype = (np.int64, (2,2))
Btype = (np.int64, (2,2))
Ctype = (np.int64, (2,2))
A = np.random.randint(0,5,(2,2))
B = np.random.randint(0,5,(2,2))
print("A\n{}".format(A))
print("B\n{}".format(B))

Cconstr = TConstruct('C')
env_map = {'A': (Atype, A), 'B': (Btype, B), 'C': (Ctype,), \
          'ranges': {'i':(0,2), 'j':(0,2), 'k':(0,2)},\
          'looplevel': 0}
Cconstr.visit(env_map)

Aindices = [IndexConst('i'), IndexConst('k')]
Bindices = [IndexConst('k'), IndexConst('j')]
Aaccess = Access(Aindices, 'A')
Baccess = Access(Bindices, 'B')
matprod = Product(Aaccess, Baccess)
matsum = SumR([IndexConst('k')],  matprod)
lh_indices = [IndexConst('i'), IndexConst('j')]
matmul = Gen('C', lh_indices, matsum)

gen_type = matmul.typecheck(env_map)
print(f"gen type {gen_type}")

matmul.visit(env_map)
print("C\n{}".format(env_map['C'][1]))
print(Cconstr.cstr(env_map))
print(matmul.cstr(env_map))


print("---------- Testing composed tensor exprs ----------")
Atype = (np.int64, (3,3))
Btype = (np.int64, (3,2))
Ctype = (np.int64, (3,2))
Vtype = (np.int64, (3,))
A = np.random.randint(0,4,(3,3))
B = np.random.randint(0,4,(3,2))

print("A\n{}".format(A))
print("B\n{}".format(B))

env_map = {'A': (Atype, A), 'B': (Btype, B), 'C': (Ctype, ), 'V': (Vtype, ), \
          'ranges': {'i':(0,3), 'j':(0,2), 'k':(0,3), 'c':(0,2)}}

Cconstr.visit(env_map)
matmul.visit(env_map)

mat_indices = [IndexConst('r'), IndexConst('c')]
mat_access = Access(mat_indices, 'C')
rowsum = SumR([IndexConst('c')], mat_access)
vect_indices = [IndexConst('r')]
Vconstr = TConstruct('V')
vect = Gen('V', vect_indices, rowsum)

gen_type = vect.typecheck(env_map)
print(f"gen type {gen_type}")

Vconstr.visit(env_map)
vect.visit(env_map)
print("row sum of matmul\n{}".format(env_map['V'][1]))

env_map['looplevel'] = 0
print(Cconstr.cstr(env_map))
print(matmul.cstr(env_map))
# reset loop level for next loop nest
env_map['looplevel'] = 0
print(Vconstr.cstr(env_map))
print(vect.cstr(env_map))


print("---------- Testing ADD  ------------")


Aindices = [IndexConst('i'), IndexConst('j')]
Bindices = [IndexConst('i'), IndexConst('j')]
Cindices = [IndexConst('i'), IndexConst('j')]
Aaccess = Access(Aindices, 'A')
Baccess = Access(Bindices, 'B')
add = Add(Aaccess, Baccess)
matadd = Gen('C', Cindices, add)
# program = {}

Atype = (np.int64, (2,2))
Btype = (np.int64, (2,2))
Ctype = (np.int64, (2,2))

A = np.random.randint(0,5,(2,2))
B = np.random.randint(0,5,(2,2))

env_map = {'A': (Atype, A), 'B': (Btype, B), 'C': (Ctype, ),\
           'ranges':{'i':(0,2), 'j':(0,2)},\
           'looplevel': 0}

gen_type = matadd.typecheck(env_map)
print(f"gen type {gen_type}")

Cconstr.visit(env_map)
matadd.visit(env_map)

print("A\n{}".format(A))
print("B\n{}".format(B))
print("element wise add of A and B\n{}".format(env_map['C'][1]))

print(Cconstr.cstr(env_map))
print(matadd.cstr(env_map))


print("---------- Testing CONV ------------")

p = IndexConst('p')
q = IndexConst('q')
k = IndexConst('k')

h = IndexConst('h')
w = IndexConst('w')
c = IndexConst('c')

Xindices = [IAdd(h, p), IAdd(w, q), k]
Windices = [c, p, q, k]
Yindices = [h, w, c]
prod = Product(Access(Xindices, 'X'), Access(Windices, 'W'))
sums = SumR([p,q,k], prod)
conv = Gen('Y', Yindices, sums)

xsize = (3,3,2)
wsize = (1,2,2,2)
ysize = (2,2,1)
Xtype = (np.float64, xsize)
Wtype = (np.int64, wsize)
Ytype = (np.float64, ysize)

X = np.zeros(xsize)
X[...,0] = 1
X[...,1] = 2
X = np.random.randint(0, 3, xsize)

W = np.random.randint(0, 3, wsize)
Yconstr = TConstruct('Y')

env_map = {'X': (Xtype, X), 'W': (Wtype, W), 'Y': (Ytype, ),\
           'ranges':{'p':(0,2), 'q':(0,2), 'k':(0,2), 'c':(0,1), 'h':(0,2), 'w':(0,2)},\
           'looplevel': 0}
Yconstr.visit(env_map)
conv.visit(env_map)
print(f"X {X}")
print(f"W {W}")
print(f"Y {env_map['Y'][1]}")
print("----------------")
gen_type = conv.typecheck(env_map)
print(f"gen type {gen_type}")
print("----------------")
print(Yconstr.cstr(env_map))
print(conv.cstr(env_map))

