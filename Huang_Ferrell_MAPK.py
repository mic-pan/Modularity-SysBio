import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

def dx_MAPK(x,t,a=1000,k=150,d=150,l=0):
    KKK = x[0]
    E1 = x[1]
    KKK_E1 = x[2]
    KKKP = x[3]
    E2 = x[4]
    KKKP_E2 = x[5]
    KK = x[6]
    KK_KKKP = x[7]
    KKP = x[8]
    KKPase = x[9]
    KKP_KKPase = x[10]
    KKP_KKKP = x[11]
    KKPP = x[12]
    KKPP_KKPase = x[13]
    K = x[14]
    K_KKPP = x[15]
    KP = x[16]
    KPase = x[17]
    KP_KPase = x[18]
    KP_KKPP = x[19]
    KPP = x[20]
    KPP_KPase = x[21]
    
    v1a = a*KKK*E1 - d*KKK_E1
    v1b = k*KKK_E1 - l*KKKP*E1
    v2a = a*KKKP*E2 - d*KKKP_E2
    v2b = k*KKKP_E2 - l*KKK*E2
    v3a = a*KK*KKKP - d*KK_KKKP
    v3b = k*KK_KKKP - l*KKP*KKKP
    v4a = a*KKP*KKPase - d*KKP_KKPase
    v4b = k*KKP_KKPase - l*KK*KKPase
    v5a = a*KKP*KKKP - d*KKP_KKKP
    v5b = k*KKP_KKKP - l*KKPP*KKKP
    v6a = a*KKPP*KKPase - d*KKPP_KKPase
    v6b = k*KKPP_KKPase - l*KKP*KKPase
    v7a = a*K*KKPP - d*K_KKPP
    v7b = k*K_KKPP - l*KP*KKPP
    v8a = a*KP*KPase - d*KP_KPase
    v8b = k*KP_KPase - l*K*KPase
    v9a = a*KP*KKPP - d*KP_KKPP
    v9b = k*KP_KKPP - l*KPP*KKPP
    v10a = a*KPP*KPase - d*KPP_KPase
    v10b = k*KPP_KPase - l*KP*KPase

    d_KKK = -v1a +v2b
    d_E1 = -v1a + v1b
    d_KKK_E1 = v1a - v1b
    d_KKKP = v1b -v2a -v3a +v3b -v5a +v5b
    d_E2 = -v2a +v2b
    d_KKKP_E2 = v2a - v2b
    d_KK = -v3a +v4b
    d_KK_KKKP = v3a -v3b
    d_KKP = v3b -v4a -v5a +v6b
    d_KKPase = -v4a +v4b -v6a +v6b
    d_KKP_KKPase = v4a -v4b
    d_KKP_KKKP = v5a -v5b
    d_KKPP = v5b -v6a -v7a +v7b -v9a +v9b
    d_KKPP_KKPase = v6a -v6b
    d_K = -v7a +v8b
    d_K_KKPP = v7a - v7b
    d_KP = v7b -v8a -v9a +v10b
    d_KPase = -v8a +v8b -v10a +v10b
    d_KP_KPase = v8a - v8b
    d_KP_KKPP = v9a - v9b
    d_K_PP = v9b -v10a
    d_KPP_KPase = v10a - v10b

    dx = [
        d_KKK, d_E1, d_KKK_E1, d_KKKP, d_E2, d_KKKP_E2,
        d_KK, d_KK_KKKP, d_KKP, d_KKPase, d_KKP_KKPase,
        d_KKP_KKKP, d_KKPP, d_KKPP_KKPase, d_K,d_K_KKPP,
        d_KP, d_KPase, d_KP_KPase, d_KP_KKPP, d_K_PP,d_KPP_KPase,
    ]

    return dx

x0 = np.array([0.0]*22)
x0[1] = 3e-5
x0[0] = 3e-3
x0[6] = 1.2
x0[14] = 1.2
x0[4] = 3e-4
x0[9] = 3e-4
x0[17] = 0.12

def generate_activation_curve(E1_vals = np.logspace(-7,-1,num=100)):
    MKKKP = np.array(len(E1_vals)*[0.])
    MKKPP = np.array(len(E1_vals)*[0.])
    MKPP = np.array(len(E1_vals)*[0.])
    t = np.arange(0.,1000.,0.1)

    a = 1000
    d = 150
    k = 150
    DG_ATP = -50000
    R = 8.314
    T = 310
    D = (a*k/d)**2*np.exp(DG_ATP/R/T)
    l = np.sqrt(D)

    for i,E1 in enumerate(E1_vals):
        x0[1] = E1
        x = odeint(dx_MAPK,x0,t,args=(a,k,d,l))
        MKKKP[i] = x[-1,3]
        MKKPP[i] = x[-1,12]
        MKPP[i] = x[-1,20]

    MKKKP_ideal = np.array(len(E1_vals)*[0.])
    MKKPP_ideal = np.array(len(E1_vals)*[0.])
    MKPP_ideal = np.array(len(E1_vals)*[0.])

    for i,E1 in enumerate(E1_vals):
        x0[1] = E1
        x = odeint(dx_MAPK,x0,t)
        MKKKP_ideal[i] = x[-1,3]
        MKKPP_ideal[i] = x[-1,12]
        MKPP_ideal[i] = x[-1,20]
    
    return E1_vals, (MKKKP,MKKPP,MKPP), (MKKKP_ideal,MKKPP_ideal,MKPP_ideal)
