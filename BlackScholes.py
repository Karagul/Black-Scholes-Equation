import numpy as np 
from scipy.interpolate import interp1d

# Import submodules
from ..output.printers import printers
from ..RK import rk 
from ..instruments.options import *
from ..utils.utils import *

# Class vars
pr = printers()

#
class BlackScholes(object):
    '''Base class of a Black-Scholes solver'''

    params = {}
    S = None
    dS = None

    # Initialize
    def __init__(self, r=0.01, v=0.2, d=0.0, silent=False):
        '''Base settings'''

        self.setR(r) # risk free interest rate
        self.setV(v) # volatility
        self.setD(d) # annulized dividend
        self.setSilent(silent)

        if not self.getSilent():
            self.prtParameters()

    # Boundary Conditions
    def BCEuropeanCallDirichlet(self, V, t, op):
        V[0] = 0
        V[-1] = self.S[-1] - op.getStrike()*np.exp(-self.getR()*t/np.timedelta64(365, 'D'))

        return V

    def BCEuropeanPutDirichlet(self, V, t, op):
        V[0] = op.getStrike()*np.exp(-self.getR()*t/np.timedelta64(365, 'D'))
        V[-1] = 0

        return V

    def BCBinaryCallDirichlet(self, V, t, op):
        V[0] = 0
        V[-1] = 0

        return V

    def BCBinaryPutDirichlet(self, V, t, op):
        V[0] = 1
        V[-1] = 0

        return V

    applyBC = {
        'EuropeanCallDirichlet' : BCEuropeanCallDirichlet,
        'EuropeanPutDirichlet' : BCEuropeanPutDirichlet,
        'BinaryCallDirichlet' : BCBinaryCallDirichlet,
        'BinaryPutDirichlet' : BCBinaryPutDirichlet
    }

    #--
    def prtParameters(self):
        iterDict(self.params, 'Black Scholes initialized.')

    #--
    def setR(self, val):
        self.params['r'] = val

    def getR(self):
        return self.params['r']
    
    #--
    def setSilent(self, val):
        self.params['silent'] = val

    def getSilent(self):
        return self.params['silent']
    
    #--
    def setV(self, val):
        self.params['v'] = val

    def getV(self):
        return self.params['v']

    #--
    def setD(self, val):
        self.params['v'] = val

    def getD(self):
        return self.params['d']

    #--
    def solve(self, op):

        # grid
        upper = np.amax([op.getStrike(), op.getUnderlying()])
        self.S = np.linspace(0, 2*upper, 101)
        self.dS = np.gradient(self.S)

        op.setSGrid(self.S)

        # payoff along grid (Initial values of V)
        po = np.asarray([op.payoff(s) for s in self.S])

        # time stepping
        # init integrator
        intg = rk.lsRK(RKtype='LSRK5_4', dt=np.timedelta64(1, 'h'), silent=self.getSilent())

        t      = np.timedelta64(0, 'D')
        histT  = [t]
        V      = po
        histV  = np.asarray(V)

        steps  = np.abs(op.getExpiry()/intg.getDt())
        inc    = int(steps/50)
        n      = 0

        while t < op.getExpiry():
            if not self.getSilent():
                pr.prtInfo('Solving backward in time: ' + str(int(n/steps*100)) + ' %', end='\r')
            V = intg.step(self.rhs, V, t)

            # apply boundary conditions
            V = self.applyBC[op.getBCType()](self, V, t, op)

            t = t + intg.getDt()
            if not n%inc:
                histV = np.vstack((histV, np.asarray(V)))
                histT.append(t)
            n = n + 1

        if not self.getSilent():
            pr.prtInfo('Solving backward in time: 100 %')

        op.setPrice(interp1d(self.S, V, kind='cubic')(op.getUnderling()))
        op.setPriceOfS(V)
        return self.S, histV, np.asarray(histT)

    def dVdS(self, V):
        

