from casadi import *

x = SX.sym('x')
y = SX.sym('y')
z = SX.sym('z')

nlp = nlp = {'x' : vertcat(x, y, z),
             'f' : x ** 2 + 100 * z ** 2,
             'g' : z + (1 - x) ** 2 - y
             }

solver = nlpsol('solver', 'ipopt', nlp)

res = solver(x0=[1, 2, 3])

print(res)
