from casadi import *

opti = casadi.Opti()

p = opti.parameter(4)

x = opti.variable()
y = opti.variable()
z = opti.variable()

opti.minimize(p[2] * (x ** p[0] + p[1] * z ** p[0]))
opti.subject_to(z + (p[3] -  x) ** p[0] == y)
opti.subject_to(x + y + z == 1)
opti.subject_to(opti.x >= 0)
opts = {'calc_lam_p' : False,
        'ipopt.linear_solver' : 'Mumps',
        'ipopt.print_level' : 5, #default=5
        }

opti.solver('ipopt', opts)

P = np.array([2, 100, 1, 0])
opti.set_value(p, P)
opti.set_initial(opti.x, vertcat(100, 200, 300))
#opti.set_initial(y, 2)
#opti.set_initial(z, 3)

sol = dict()
sol[0] = opti.solve()

#-----------Below, opti.x is the full ipopt vector:
print(sol[0].value(opti.x))
#-----------Here it is one of the three variables:
print(sol[0].value(vertcat(x, y , z)))
#-----------The following will not work:
#print(sol[0].value([x, y , z]))

#-----------Next run: we need to set new values.
#-----------for parameters:
opti.set_value(p, P)

#-----------for variables: (opti.x is the full ipopt vector!!)
opti.set_initial(x, sol[0].value(x))
opti.set_initial(y, sol[0].value(y))
opti.set_initial(z, sol[0].value(z))

#-----------and a warm start for lam_g (prices)
lam_g0 = sol[0].value(opti.lam_g)
opti.set_initial(opti.lam_g, lam_g0)

#-----------The following will not work:
#opti.set_initial(opti.g, sol[0].value(opti.g))

#-----------and a quick change to parameters:
P1 = P
P1[3] = 1
opti.set_value(p, P1)
sol[1] = opti.solve()

for s in range(len(sol)):
    print(
        sol[s].value(opti.x), "\n",
        sol[s].value(opti.lam_g[0:2]), "\n",
    )



