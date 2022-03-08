from casadi import *

p = SX.sym('p')

x = SX.sym('x')
y = SX.sym('y')
z = SX.sym('z')

nlp = nlp = {'x' : vertcat(x, y, z),
             'f' : x ** p + 100 * z ** p,
             'g' : z + (1 - x) ** p - y,
             'p' : p
             }

opts = {"ipopt.linear_solver" : "Mumps"}
#-----------when HSL is available, we should also be able to run:
#opts = {"ipopt.linear_solver" : "MA27"}
#-----------or:
#opts = {"ipopt.linear_solver" : "MA57"}
#-----------or:
#opts = {"ipopt.linear_solver" : "HSL_MA86"}
#-----------or:
#opts = {"ipopt.linear_solver" : "HSL_MA97"}

#-----------the following advice comes from https://www.hsl.rl.ac.uk/ipopt/
#-----------"when using HSL_MA86 or HSL_MA97 ensure MeTiS ordering is 
#-----------compiled into Ipopt to maximize parallelism"

solver = nlpsol('solver', 'ipopt', nlp, opts)

res = solver(x0=[1, 2, 3], p = 2.5)
print(res)
res = solver(x0=[3, 2, 1], p = 2)
print(res)
