
#==============================================================================
#-----------structure of x using latex notation:
#---x = [
#        x_{p0, t0, r0, s0}, x_{p0, t0, r0, s1}, x_{p0, t0, r0, s2},
#
#        x_{p0, t0, r1, s0}, x_{p0, t0, r1, s1}, x_{p0, t0, r1, s2},
#
#        x_{p0, t1, r0, s0}, x_{p0, t1, r0, s1}, x_{p0, t1, r0, s2},
#
#        x_{p0, t1, r1, s0}, x_{p0, t1, r1, s1}, x_{p0, t1, r1, s2},
#
#        x_{p1, t0, r0, s0}, x_{p1, t0, r0, s1}, x_{p0, t0, r0, s2},
#
#        x_{p1, t0, r1, s0}, x_{p1, t0, r1, s1}, x_{p0, t0, r1, s2},
#
#        x_{p1, t1, r0, s0}, x_{p1, t1, r0, s1}, x_{p0, t1, r0, s2},
#
#        x_{p1, t1, r1, s0}, x_{p1, t1, r1, s1}, x_{p0, t1, r1, s2},
#
#
#        x_{p2, t0, r0, s00}, x_{p2, t0, r0, s01}, x_{p2, t0, r0, s02},
#        x_{p2, t0, r0, s10}, x_{p2, t0, r0, s11}, x_{p2, t0, r0, s12},
#        x_{p2, t0, r0, s20}, x_{p2, t0, r0, s21}, x_{p2, t0, r0, s22},
#
#        x_{p2, t0, r1, s00}, x_{p2, t0, r1, s01}, x_{p2, t0, r1, s02},
#        x_{p2, t0, r1, s10}, x_{p2, t0, r1, s11}, x_{p2, t0, r1, s12},
#        x_{p2, t0, r1, s20}, x_{p2, t0, r1, s21}, x_{p2, t0, r1, s22},
#
#        x_{p2, t1, r0, s00}, x_{p2, t1, r0, s01}, x_{p2, t1, r0, s02},
#        x_{p2, t1, r0, s10}, x_{p2, t1, r0, s11}, x_{p2, t1, r0, s12},
#        x_{p2, t1, r0, s20}, x_{p2, t1, r0, s21}, x_{p2, t1, r0, s22},
#
#        x_{p2, t1, r1, s00}, x_{p2, t1, r1, s01}, x_{p2, t1, r1, s02},
#        x_{p2, t1, r1, s10}, x_{p2, t1, r1, s11}, x_{p2, t1, r1, s12},
#        x_{p2, t1, r1, s20}, x_{p2, t1, r1, s21}, x_{p2, t1, r1, s22},
#
#
#        x_{p3, t0, r0, s00}, x_{p3, t0, r0, s01}, x_{p3, t0, r0, s02},
#        x_{p3, t0, r0, s10}, x_{p3, t0, r0, s11}, x_{p3, t0, r0, s12},
#        x_{p3, t0, r0, s20}, x_{p3, t0, r0, s21}, x_{p3, t0, r0, s22},
#
#        x_{p3, t0, r1, s00}, x_{p3, t0, r1, s01}, x_{p3, t0, r1, s02},
#        x_{p3, t0, r1, s10}, x_{p3, t0, r1, s11}, x_{p3, t0, r1, s12},
#        x_{p3, t0, r1, s20}, x_{p3, t0, r1, s21}, x_{p3, t0, r1, s22},
#
#        x_{p3, t1, r0, s00}, x_{p3, t1, r0, s01}, x_{p3, t1, r0, s02},
#        x_{p3, t1, r0, s10}, x_{p3, t1, r0, s11}, x_{p3, t1, r0, s12},
#        x_{p3, t1, r0, s20}, x_{p3, t1, r0, s21}, x_{p3, t1, r0, s22},
#
#        x_{p3, t1, r1, s00}, x_{p3, t1, r1, s01}, x_{p3, t1, r1, s02},
#        x_{p3, t1, r1, s10}, x_{p3, t1, r1, s11}, x_{p3, t1, r1, s12},
#        x_{p3, t1, r1, s20}, x_{p3, t1, r1, s21}, x_{p3, t1, r1, s22},
#       ]
#
#==============================================================================
#---------------dicts
#-----------dimensions for each pol var: 0 : scalar; 1 : vector; 2 : matrix

d_dim = {
    "con": 1,
    "knx": 1,
    "lab": 1,
}
i_pol = {
    "con": 0,
    "knx": 1,
    "lab": 2,
}
i_reg = {
    "aus": 0,
    "qld": 1,
    "wld": 2,
}
i_sec = {
    "agr": 0,
    "for": 1,
    #"min": 2,
    #"man": 3,
    #"uty": 4,
    #"ctr": 5,
    #"com": 6,
    #"tps": 7,
    #"res": 8,
}
# Warm start
pol_S = {
    "con": 4,
    "lab": 1,
    "knx": KAP0,
    #"sav": 2,
    #"out": 6,
    #    "itm": 10,
    #    "ITM": 10,
    #    "SAV": 10,
    #"utl": 1,
    #    "val": -300,
}
#-----------dicts of index lists for locating variables in x:
#-------Dict for locating every variable for a given policy
d_pol_ind_x = dict()
for pk in i_pol.keys():
    p = i_pol[pk]
    d = d_dim[pk]
    stride = NTIM * NREG * NSEC ** d
    start = p * stride
    end = start + stride
    d_pol_ind_x[pk] = range(NVAR)[start : end : 1]

#-------Dict for locating every variable at a given time
d_tim_ind_x = dict()
for t in range(NTIM):
    indlist = []
    for pk in i_pol.keys():
        p = i_pol[pk]
        d = d_dim[pk]
        stride = NREG * NSEC ** d
        start = (p * NTIM + t) * stride
        end = start + stride
        indlist.extend(range(NVAR)[start : end : 1])
    d_tim_ind_x[t] = sorted(indlist)

#-----------the final one can be done with a slicer with stride NSEC ** d_dim
#-------Dict for locating every variable in a given region
d_reg_ind_x = dict()
for rk in i_reg.keys():
    r = i_reg[rk]
    indlist = []
    for t in range(NTIM):
        for pk in i_pol.keys():
            p = i_pol[pk]
            d = d_dim[pk]
            stride = NSEC ** d
            start = (p * NTIM * NREG + t * NREG + r) * stride
            end = start + stride
            indlist += range(NVAR)[start : end : 1]
    d_reg_ind_x[rk] = sorted(indlist)

#-------Dict for locating every variable in a given sector
d_sec_ind_x = dict()
for sk in i_sec.keys(): #comment
    s = i_sec[sk]
    indlist = []
    for rk in i_reg.keys():
        r = i_reg[rk]
        for t in range(NTIM):
            for pk in i_pol.keys():
                p = i_pol[pk]
                d = d_dim[pk]
                stride = 1
                start = (p * NTIM * NREG + t * NREG + r) * NSEC ** d + s
                end = start + stride
                indlist += range(NVAR)[start : end : 1]
    d_sec_ind_x[s] = sorted(indlist)

#-----------union of all the "in_x" dicts: those relating to indices of x
d_ind_x = d_pol_ind_x | d_tim_ind_x | d_reg_ind_x | d_sec_ind_x

#------------------------------------------------------------------------------
#-----------function for returning index subsets of x for a pair of dict keys
def sub_ind_x(key1,             # any key of d_ind_x
              key2,             # any key of d_ind_x
              d=d_ind_x,   # dict of index categories: pol, time, sec, reg
              ):
    val = np.array(list(set(d[key1]) & set(d[key2])))
    return val
j_sub_ind_x = jit(sub_ind_x)
# possible alternative: ind(ind(ind(range(len(X0)), key1),key2), key3)

#-----------function for intersecting two lists: returns indices as np.array
#def f_I2L(list1,list2):
#    return np.array(list(set(list1) & set(list2)))

