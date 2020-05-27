import numpy as np

class IndexConst:
    def __init__(self, name):
      self.values = [name]

    def visit(self, env_map):
      e = self.values[0] #char name
      e_val = env_map[e]
      return e_val

class Add:
  def __init__(self, lhs, rhs):
    self.values = [lhs, rhs]

  def visit(self, env_map):
    e1 = self.values[0].visit(env_map)
    e2 = self.values[1].visit(env_map)
    e1_val = env_map[e1]
    e2_val = env_map[e2]
    return e1_val + e2_val

class Tensor:
  def __init__(self, t):
    self.values = [t]

  def visit(self):
    return self.values[0]

class Access:
  def __init__(self, index_exprs, tensor):
    self.values = [index_exprs, tensor]

  def visit(self, env_map):
    # should be a tensor
    t = self.values[1].visit()
    # array of index expressions, which are either IndexConst or Add
    # assuming everything is 2D, 3D, or 4D
    index_exprs = self.values[0] 
    assert(len(index_exprs) == t.ndim)
    if t.ndim == 2:
      return t[index_exprs[0].visit(env_map), index_exprs[1].visit(env_map)]
    if t.ndim == 3:
      return t[index_exprs[0].visit(env_map), index_exprs[1].visit(env_map), index_exprs[2].visit(env_map)]
    if t.ndim == 4:
      return t[index_exprs[0].visit(env_map), index_exprs[1].visit(env_map), index_exprs[2].visit(env_map), index_exprs[3].visit(env_map)]


class Product:
  def __init__(self, lhs, rhs):
    self.values = [lhs, rhs]

  def visit(self, env_map):
    e1 = self.values[0].visit(env_map)
    e2 = self.values[1].visit(env_map)
    return e1 * e2

class SumR:
  def __init__(self, r_var, r_dom, expr):
    self.values = [r_var, r_dom, expr]

  def visit(self, env_map):
    r = 0.0
    r_var_name = self.values[0].values[0]
    r_dom = self.values[1]
    e = self.values[2]

    for val in range(r_dom[0], r_dom[1]):
      env_map[r_var_name] = val
      r += e.visit(env_map)

    return r


class Gen:
  def __init__(self, dim_sizes, lhs_index_exprs, rhs):
    self.values = [dim_sizes, lhs_index_exprs, rhs]

  def subGen(self, lhs, lhs_index_exprs, dim_id, dim_sizes, rhs, env_map):
    dim_size = dim_sizes[dim_id]
    for val in range(dim_size):
      dim_name = lhs_index_exprs[dim_id].values[0]
      env_map[dim_name] = val

      if dim_id == (len(lhs_index_exprs) - 1): # all indices have been determined
        if len(lhs_index_exprs) == 2:
          lhs[lhs_index_exprs[0].visit(env_map), lhs_index_exprs[1].visit(env_map)] = rhs.visit(env_map)
        if len(lhs_index_exprs) == 3:
          lhs[lhs_index_exprs[0].visit(env_map), lhs_index_exprs[1].visit(env_map), lhs_index_exprs[2].visit(env_map)] = rhs.visit(env_map)
        if len(lhs_index_exprs) == 4:
          lhs[lhs_index_exprs[0].visit(env_map), lhs_index_exprs[1].visit(env_map), lhs_index_exprs[2].visit(env_map), lhs_index_exprs[3].visit(env_map)] = rhs.visit(env_map)

      else:
        self.subGen(lhs, lhs_index_exprs, dim_id+1, dim_sizes, rhs, env_map)


  def visit(self, env_map):
    # array of dimension sizes
    dim_sizes = self.values[0]
    # array of dimension names, should be type index const for now
    lhs_index_exprs = self.values[1]
    rhs = self.values[2]

    t = np.zeros(dim_sizes)
    self.subGen(t, lhs_index_exprs, 0, dim_sizes, rhs, env_map)
    return t

rht = np.random.randint(0,10,(3,3,2))
print(rht)
t = Tensor(rht)
rh_indices = [IndexConst('x'), IndexConst('y'), IndexConst('z')]
access = Access(rh_indices, t)
lht_sizes = [2,3,2]

lh_indices = [IndexConst('x'), IndexConst('y'), IndexConst('z')]
gen = Gen(lht_sizes, lh_indices, access)
result = gen.visit({})

print(result)
print("---------- Testing MATMUL ------------")

A = np.random.randint(0,5,(2,2))
B = np.random.randint(0,5,(2,2))
print("A")
print(A)
print("B")
print(B)

At = Tensor(A)
Bt = Tensor(B)
Aindices = [IndexConst('i'), IndexConst('k')]
Bindices = [IndexConst('k'), IndexConst('j')]
Aaccess = Access(Aindices, At)
Baccess = Access(Bindices, Bt)
matprod = Product(Aaccess, Baccess)
matsum = SumR(IndexConst('k'), (0,2), matprod)

lht_sizes = [2,2]
lh_indices = [IndexConst('i'), IndexConst('j')]
gen = Gen(lht_sizes, lh_indices, matsum)
result = gen.visit({})
print("C")
print(result)


print("---------- Testing CONV ------------")

