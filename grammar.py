import numpy as np
# from a github gist by victorlei
def extclass(cls):
  return lambda f: (setattr(cls,f.__name__,f) or f)
# ------------------------------------------------------------------------
# Grammatical Forms (Classes)
"""
BNF for our grammar
expr ::=   Add(expr, expr)
       |   Access( iexpr*, texpr )
       |   Product(expr, expr)
       |   SumR(iexpr, (int,int), texpr)
       |   Gen( int*, iexpr*, expr )
texpr ::=  Gen( int*, iexpr*, expr )
       |   Tensor(tensor)
iexpr ::=  IndexConst(string)
        |  IAdd(iexpr, iexpr)
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
  def __init__(self, index_exprs, tensor):
    assert type(index_exprs) is list, "expected index_exprs to be a list"
    for i,ie in enumerate(index_exprs):
      assert isinstance(ie, iexpr), "expected elem {i} of index_exprs to be an iexpr"
    assert isinstance(tensor, texpr), "expected tensor to be a texpr"
    self.values = [index_exprs, tensor]

class Product(expr):
  def __init__(self, lhs, rhs):
    assert isinstance(lhs, expr), "expected expr on lhs of Product"
    assert isinstance(rhs, expr), "expected expr on rhs of Product"
    self.values = [lhs, rhs]

class SumR(expr, Update):
  def __init__(self, r_var, term):
    assert isinstance(r_var, iexpr), "expected r_var of SumR to be an iexpr"
    assert isinstance(term, expr), "expected term of SumR to be an eexpr"
    self.values = [r_var, term]

class Gen(texpr):
  def __init__(self, name, lhs_index_exprs, rhs):
    assert type(lhs_index_exprs) is list, "expected lhs_index_exprs to be a list"
    for i,ie in enumerate(lhs_index_exprs):
      assert isinstance(ie, iexpr), "expected elem {i} of lhs_index_exprs to be an iexpr"
    assert isinstance(rhs, expr), "expected rhs to be an expr"
    self.values = [name, lhs_index_exprs, rhs]

# ------------------------------------------------------------------------
# Visit() Pass
@extclass(IndexConst)
def visit(self, env_map):
  e = self.values[0] #char name
  e_val = env_map[e]
  return e_val

@extclass(Add)
def visit(self, env_map):
  e1 = self.values[0].visit(env_map)
  e2 = self.values[1].visit(env_map)
  return e1 + e2

@extclass(Tensor)
def visit(self, env_map):
  return env_map[self.values[0]]

@extclass(Access)
def visit(self, env_map):
  # should be a tensor
  t = self.values[1].visit(env_map)
  # array of index expressions, which are either IndexConst or Add
  # assuming everything is 2D, 3D, or 4D
  index_exprs = self.values[0] 
  assert(len(index_exprs) == t.ndim), "expected length of index_exprs to match number of tensor dimensions"
  if t.ndim == 2:
    return t[index_exprs[0].visit(env_map), index_exprs[1].visit(env_map)]
  if t.ndim == 3:
    return t[index_exprs[0].visit(env_map), index_exprs[1].visit(env_map), index_exprs[2].visit(env_map)]
  if t.ndim == 4:
    return t[index_exprs[0].visit(env_map), index_exprs[1].visit(env_map), index_exprs[2].visit(env_map), index_exprs[3].visit(env_map)]

@extclass(Product)
def visit(self, env_map):
  e1 = self.values[0].visit(env_map)
  e2 = self.values[1].visit(env_map)
  return e1 * e2

@extclass(SumR)
def visit(self, env_map):
  r = 0.0
  r_var_name = self.values[0].values[0]
  r_dom = env_map['ranges'][r_var_name]
  assert isinstance(r_dom, tuple), "expected r_dom of SumR to be a tuple"
  assert len(r_dom) == 2, "expected r_dom of SumR to contain two values"
  for v in r_dom:
    assert type(v) is int, "expected values in r_dom of SumR to be ints"
  e = self.values[1]

  for val in range(r_dom[0], r_dom[1]):
    env_map[r_var_name] = val
    r += e.visit(env_map)
  return r

@extclass(Gen)
def visit(self, env_map):
  name = self.values[0]
  # array of dimension sizes
  dim_sizes = env_map['gen_sizes'][name]
  assert type(dim_sizes) is list, "expected dim_sizes to be a list"
  for i,ie in enumerate(dim_sizes):
    assert type(ie) is int, "expected elem {i} of dim_sizes to be an int"

  # array of dimension names, should be type index const for now
  lhs_index_exprs = self.values[1]
  rhs = self.values[2]
  def subGen(lhs, lhs_index_exprs, dim_id, dim_sizes, rhs, env_map):
    dim_size = dim_sizes[dim_id]
    for val in range(dim_size):
      dim_name = lhs_index_exprs[dim_id].values[0]
      env_map[dim_name] = val
      if dim_id == (len(lhs_index_exprs) - 1): # all indices have been determined
        if len(lhs_index_exprs) == 1:
          lhs[lhs_index_exprs[0].visit(env_map)] = rhs.visit(env_map)
        if len(lhs_index_exprs) == 2:
          lhs[lhs_index_exprs[0].visit(env_map), lhs_index_exprs[1].visit(env_map)] = rhs.visit(env_map)
        if len(lhs_index_exprs) == 3:
          lhs[lhs_index_exprs[0].visit(env_map), lhs_index_exprs[1].visit(env_map), lhs_index_exprs[2].visit(env_map)] = rhs.visit(env_map)
        if len(lhs_index_exprs) == 4:
          lhs[lhs_index_exprs[0].visit(env_map), lhs_index_exprs[1].visit(env_map), lhs_index_exprs[2].visit(env_map), lhs_index_exprs[3].visit(env_map)] = rhs.visit(env_map)
      else:
        subGen(lhs, lhs_index_exprs, dim_id+1, dim_sizes, rhs, env_map)
  t = np.zeros(dim_sizes)
  subGen(t, lhs_index_exprs, 0, dim_sizes, rhs, env_map)
  return t

# ------------------------------------------------------------------------
# dump_cstr() Pass
@extclass(IndexConst)
def cstr(self, env_map):
  e = self.values[0] #char name
  return e

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
  tname = self.values[1].cstr(env_map)
  t = self.values[1].visit(env_map)
  # array of index expressions, which are either IndexConst or Add
  # assuming everything is 2D, 3D, or 4D
  index_exprs = self.values[0] 
  assert(len(index_exprs) == t.ndim), "expected length of index_exprs to match number of tensor dimensions"
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
  r_var_name = self.values[0].values[0]
  r_dom = env_map['ranges'][r_var_name]
  assert isinstance(r_dom, tuple), "expected r_dom of SumR to be a tuple"
  assert len(r_dom) == 2, "expected r_dom of SumR to contain two values"
  for v in r_dom:
    assert type(v) is int, "expected values in r_dom of SumR to be ints"
  e = self.values[1]
  
  initial_val = "0.0"
  lhstr = env_map["lhstr"]
  rhstr = e.cstr(env_map)
  loop = (f"""
  for (int {r_var_name} = {r_dom[0]}; {r_var_name} < {r_dom[1]}; {r_var_name}++) {{
    {lhstr} += {rhstr}
  }}
  """)        
  return initial_val, loop

@extclass(Gen)
def cstr(self, env_map):
  name = self.values[0]
  # array of dimension sizes
  dim_sizes = env_map['gen_sizes'][name]
  assert type(dim_sizes) is list, "expected dim_sizes to be a list"
  for i,ie in enumerate(dim_sizes):
    assert type(ie) is int, "expected elem {i} of dim_sizes to be an int"

  # array of dimension names, should be type index const for now
  lhs_index_exprs = self.values[1]
  rhs = self.values[2]

  def subcstr(lhs_index_exprs, dim_id, dim_sizes, rhs, env_map):
    dim_size = dim_sizes[dim_id]
    loopvar = lhs_index_exprs[dim_id].cstr(env_map)
    if dim_id == (len(lhs_index_exprs) - 1):
      gen_name = self.values[0]

      lhs_index_str = ",".join([index.cstr(env_map) for index in lhs_index_exprs])
      if isinstance(rhs, Update):
        env_map['lhstr'] = f"{gen_name}[{lhs_index_str}]"
        rhs_initial, rhs_update_str = rhs.cstr(env_map)
        loop_str = f"""
        for (int {loopvar} = 0; {loopvar} < {dim_size}; {loopvar}++) {{
          {gen_name}[{lhs_index_str}] = {rhs_initial};
          {rhs_update_str}
        }}
        """
      else:
        rhs_expr_str = rhs.cstr(env_map)
        loop_str = f"""
        for (int {loopvar} = 0; {loopvar} < {dim_size}; {loopvar}++) {{
          {gen_name}[{lhs_index_str}] = {rhs_expr_str};
        }}
        """
    else:
      child_loop_str = subcstr(lhs_index_exprs, dim_id+1, dim_sizes, rhs, env_map)
      loop_str = f"""
      for (int {loopvar} = 0; {loopvar} < {dim_size}; {loopvar}++) {
        {child_loop_str}
      }
      """
    return loop_str

  return subcstr(lhs_index_exprs, 0, dim_sizes, rhs, env_map)


# ------------------------------------------------------------------------
# Tests
rht = np.random.randint(0,10,(3,3,2))
print(rht)
t = Tensor('T')

rh_indices = [IndexConst('x'), IndexConst('y'), IndexConst('z')]
access = Access(rh_indices, t)
lht_sizes = [2,3,2]
lh_indices = [IndexConst('x'), IndexConst('y'), IndexConst('z')]
gen = Gen('copy', lh_indices, access)
result = gen.visit({'T':rht, 'gen_sizes': {'copy':lht_sizes}})
print(result)

print("---------- Testing MATMUL ------------")
At = Tensor('A')
Bt = Tensor('B')

Aindices = [IndexConst('i'), IndexConst('k')]
Bindices = [IndexConst('k'), IndexConst('j')]
Aaccess = Access(Aindices, At)
Baccess = Access(Bindices, Bt)
matprod = Product(Aaccess, Baccess)
matsum = SumR(IndexConst('k'),  matprod)
lh_indices = [IndexConst('i'), IndexConst('j')]
matmul = Gen('matmul', lh_indices, matsum)

A = np.random.randint(0,5,(2,2))
B = np.random.randint(0,5,(2,2))
print("A\n{}".format(A))
print("B\n{}".format(B))
result = matmul.visit({'A': A, 'B': B, \
                      'gen_sizes': {'matmul': [2,2]}, \
                      'ranges': {'i':(0,2), 'j':(0,2), 'k':(0,2)}})
print("C\n{}".format(result))
print(matmul.cstr({'A': A, 'B': B, \
                   'gen_sizes': {'matmul': [2,2]}, \
                   'ranges': {'i':(0,2), 'j':(0,2), 'k':(0,2)}}))

print("---------- Testing composed tensor exprs ----------")
A = np.random.randint(0,5,(3,3))
B = np.random.randint(0,5,(3,2))
print("A\n{}".format(A))
print("B\n{}".format(B))
mat_indices = [IndexConst('r'), IndexConst('c')]
mat_access = Access(mat_indices, matmul)
rowsum = SumR(IndexConst('c'), mat_access)
vect_indices = [IndexConst('r')]
vect = Gen('vect', vect_indices, rowsum)
result = vect.visit({'A': A, 'B': B, \
                    'gen_sizes': {'vect': [3], 'matmul':[3,2]}, \
                    'ranges': {'i':(0,3), 'j':(0,2), 'k':(0,3), 'c':(0,2)}})

print("row sum of matmul\n{}".format(result))

print("---------- Testing ADD  ------------")
Aindices = [IndexConst('i'), IndexConst('j')]
Bindices = [IndexConst('i'), IndexConst('j')]
Cindices = [IndexConst('i'), IndexConst('j')]
Aaccess = Access(Aindices, At)
Baccess = Access(Bindices, Bt)
add = Add(Aaccess, Baccess)
matadd = Gen('add', Cindices, add)

A = np.random.randint(0,5,(2,2))
B = np.random.randint(0,5,(2,2))
result = matadd.visit({'A': A, 'B': B,\
                      'gen_sizes': {'add':[2,2]}, 'ranges':{'i':(0,2), 'j':(0,2)}})

print("A\n{}".format(A))
print("B\n{}".format(B))
print("element wise add of A and B\n{}".format(result))


print(matadd.cstr({'A': A, 'B': B,\
                  'gen_sizes': {'add':[2,2]}, 'ranges':{'i':(0,2), 'j':(0,2)}}))
print("---------- Testing CONV ------------")

