import numpy as np
import random
import math
from itertools import combinations 
import copy
# #########  Environment  ####################
class Hypothesis():
    def __init__(self, n, a, r):


        self.NoProcess = n
        self.NoHypo = 2 ** self.NoProcess
        self.n = 2 ** self.NoProcess  # number of hypothesis
        self.NFeatures = 2*self.NoProcess
        self.rho = r
        self.Pf = 0.8

        self.pcross = self.Pf * (1 - self.Pf)

        self.P12 = [self.Pf * self.Pf + self.rho * self.pcross,
                    (1 - self.rho) * self.pcross,
                    (1 - self.rho) * self.pcross,
                    (1 - self.Pf) * (1 - self.Pf) + self.rho * self.pcross]
        self.P3 = [self.Pf, 1 - self.Pf]

        self.normal = [self.Pf, 1-self.Pf]
        self.abnormal = [1-self.Pf, self.Pf]

##########Observation model
        self.Q = []
        for iter in range(2 ** (self.NoProcess)):
            NoTemp = bin(iter)[2:].zfill(self.NoProcess)
            BinIndex = [int(x) for x in NoTemp]
            TempList = []
            for iter_in in range(self.NoProcess):
                if BinIndex[iter_in] == 0:
                    TempList.append(self.normal)
                else:
                    TempList.append(self.abnormal)

            self.Q.append(TempList)
            
        self.Q = np.array(self.Q)
        print('check Q', np.shape(self.Q))

##########Hypothesis model

        self.p_fix = np.kron(self.P12, self.P3)
        self.p_fix = np.kron(self.p_fix, self.P12)
        
        self.p_update = np.zeros((self.NoProcess,self.NoProcess,2,2),dtype=float)
        
        for iter in range(self.NoProcess):
            for iter_in in range(self.NoProcess):
                self.p_update[iter,iter_in] = [[self.Pf, 1-self.Pf], [self.Pf, 1-self.Pf]]
                self.p_update[iter,iter] = [[1,0],[0,1]]
    
        update_cross =[ [self.P12[0]/self.Pf, self.P12[1]/self.Pf],[self.P12[2]/(1-self.Pf),self.P12[3]/(1-self.Pf)]]
        self.p_update[0,1] = np.matrix(update_cross)
        self.p_update[1,0] = np.matrix(update_cross)
        self.p_update[3,4] = np.matrix(update_cross)
        self.p_update[4,3] = np.matrix(update_cross)
        
        #        self.p_fix = np.kron(self.P12,self.p_fix)
        print('check prior', self.p_fix)


##########Actions model        
        self.Actions = []
        for iter in a:
            for iter_in in range(iter):
                comb = combinations(range(n), iter_in+1)
                self.Actions=self.Actions + list(comb)       


    def Update(self, y, u, CurrP):
        P = CurrP.copy()
        ui = u

        P = np.multiply(self.Q[:, ui, y], P)
        P = P / sum(P)
        self.p = P

        return P

    def Confidence(self, p_tmp):
        C = [0 for col in range(self.n)]
        for h in range(self.n):
            if 1 - p_tmp[h] == 0:
                p_tmp[h] = 1 - 10 ** (-12)
                
            if p_tmp[h] == 0:
                p_tmp[h] = 10 ** (-12)

            C[h] = p_tmp[h] * math.log10(p_tmp[h]/(1 - p_tmp[h]))
        r = sum(C)
        return r


    def H_generator(self):
        h_n = random.uniform(0, 1)

        for i in range(self.n):

            if h_n < sum(self.p_fix[0:i + 1]):

                h = i

                break
        return h
    
    def Cal_NegEntropy(self,CurrP):
        H = 0
        for iter in range(self.NoProcess):
            q = [CurrP[iter], 1-CurrP[iter]]
            if q[0] == 0:
                q[0] = 10 ** (-12)
                
            if q[1] == 0:
                q[1] = 10 ** (-12)
            H += q[0]* np.log(q[0]) + q[1]*np.log(q[1])
        return H


    def H_reset(self):
        h = 0
        return h


    def Observe(self, u, h):
        Obs = []
        for ui in self.Actions[u]:
            Prob = self.Q[h][ui][0]
            ob = random.uniform(0, 1)
            if ob < Prob:
                y = 0
            else:
                y = 1
            Obs.append(y)    
        return Obs

    def ComputePDF(self, Sigma):
        PDF = copy.copy(self.p_fix)
        for ui in range(self.NoHypo):
            BinRep = [int(b) for b in list(bin(ui)[2:].zfill(self.NoProcess))]
            for indx in range(self.NoProcess):
                PDF[ui] = PDF[ui]*Sigma[indx + BinRep[indx]*self.NoProcess]
        PDF = PDF/sum(PDF)
        return PDF
    
    def UpdateSigma(self, PrevSigma, CurrentAction, Observation):
        P = PrevSigma.copy()
        
        Obs = Observation[0]
        py0 = self.Pf**(1-Obs)*(1-self.Pf)**Obs
        py1 = self.Pf**(Obs)*(1-self.Pf)**(1-Obs)
        
        for iter in range(self.NoProcess):
            p_matrix = self.p_update[iter,CurrentAction]
            update0 = py0*p_matrix[0,0] + py1*p_matrix[0,1]
            update1 = py0*p_matrix[1,0] + py1*p_matrix[1,1]
            
            P[iter] = (P[iter]*update0)/(P[iter]*update0 + (1-P[iter])*update1) 
       
        return P





    # #########  End_Environment ##################




