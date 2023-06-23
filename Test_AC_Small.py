import numpy as np
from Actor import DRLactor
from Critic import DRLcritic
# from environment_oldVersion import Hypothesis
from environment_Small import Hypothesis
import tensorflow as tf
import copy
import scipy.io as sio
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(None)
tf.reset_default_graph()

if __name__ == "__main__":

    lbd_list = [0]#, 0.0005, 0.005, 0.05, 0.1, 0.2, 0.5]
    rho_list =  [0, 0.2, 0.4,0.6, 0.8, 1]

    UpperProb =  [0.8, 0.83, 0.86, 0.89, 0.92, 0.95, 0.98, 0.99]
    LowerProb = 0.5
    LtUpperProb = len(UpperProb)

    for l_id in range(len(lbd_list)):
        lbd = lbd_list[l_id]

        for r_id in range(len(rho_list)):
            rho = rho_list[r_id]
            TotalEpisodes = 1000
            MaxTimeSteps = 200

            LR_A = 0.0005
            LR_C = 0.005
            delta_epsilon = 0.9
            
            N_process = 5
            N_actions = N_process
            ActionPerTime = 1;
            N_features = 2*N_process

            ActorNet = DRLactor(n_features=N_process, n_actions=N_process, lr=LR_A, num=ActionPerTime, lbd=lbd, rho=rho)
            CriticNet = DRLcritic(n_features=N_process, lr=LR_C, num=ActionPerTime, lbd=lbd, rho=rho)
            ActorNet.reload_model()
            CriticNet.reload_model()
            
            Env = Hypothesis(N_process, [ActionPerTime], rho)
            Miss = [0]*LtUpperProb
            Loss = [0]*LtUpperProb
            StoppingTime = [0]*LtUpperProb
            EstCount = [0]*LtUpperProb
            MesCount = [0]*LtUpperProb
            TimeCount = [0]*LtUpperProb
        
            for EpisodeNo in range(TotalEpisodes):
                
                EstListNew = list(range(LtUpperProb))
                Mes = 0
                CountActions = [0]*N_actions
                CountTime = 0

                CurrentSigma = [Env.Pf]*N_process
                CurrentHypothesis = Env.H_generator()

                for N in range(MaxTimeSteps):
                    t0 = time.perf_counter()
                    PrevSigma = copy.deepcopy(CurrentSigma)
                    CurrentAction = ActorNet.choose_action([PrevSigma], delta_epsilon**(EpisodeNo*MaxTimeSteps + N))
                    CountActions[CurrentAction] += 1
                    Observation = Env.Observe(CurrentAction, CurrentHypothesis)
                    CurrentSigma = Env.UpdateSigma(PrevSigma, CurrentAction, Observation)
                    
                    CountTime += time.perf_counter() - t0
                    EstimateList = [0]*N_process
                    PDFList = [0]*N_process
                    for hypo in range(N_process):
                        SigmaHypo = [CurrentSigma[hypo], 1-CurrentSigma[hypo]];
                        EstimateList[hypo] = np.argmax(SigmaHypo)
                        PDFList[hypo] = np.max(SigmaHypo)

                    HypothesisEstimate = int("".join(str(x) for x in EstimateList), 2)
                        
                    EstList = EstListNew.copy()   
                    Mes = Mes + ActionPerTime
                    
                    for UP in EstList:
                        if np.min(PDFList) > UpperProb[UP]:
                            MesCount[UP] = MesCount[UP] + Mes
                            StoppingTime[UP] = StoppingTime[UP]+N+1
                            EstCount[UP] = EstCount[UP]+1
                            TimeCount[UP] += CountTime
                            EstListNew.remove(UP)
                            
                            if HypothesisEstimate != CurrentHypothesis:
                                Loss[UP] = Loss[UP] + 1
                    if len(EstListNew) == 0:
                       break
                        
                for M in EstListNew: 
                    Miss[M] = Miss[M]+1
                #print('Episode',EpisodeNo)

            AvgMiss = [round(M/TotalEpisodes,2) for M in Miss]
            AvgLoss = [round(M/max(D,1),4) for M,D in zip(Loss,EstCount)]
            Accuracy   = [1-(L+M)/TotalEpisodes for L,M in zip(Loss,Miss)]
            AvgRunTime = [round(1000*M/max(O,1),4) for M,D,O in zip(TimeCount,EstCount,MesCount)]

                
            AvgStoppingTime = [round(M/max(D,1),2) for M,D in zip(StoppingTime,EstCount)]
            AvgCost = [round(M/max(D,1),2) for M,D in zip(MesCount,EstCount)]
            #    print('Wrong Conclusion:',AvgMissd,'\n','No Change', AvgMissd)
            print('Miss:', AvgMiss)
            print('Accuracy:', Accuracy,'\n RunTime:', AvgRunTime,'\n Measurement:', AvgCost)
            
            sio.savemat('Test_AC_'+ '_' + str(lbd) + '_' + str(rho) +'.mat', {'UpperThreshold': UpperProb, 'Accuracy': Accuracy, 'StoppingTime': AvgStoppingTime, 'Cost': AvgCost, 'lbd':lbd, 'rho':rho, 'RunTime':AvgRunTime})

            ActorNet.kill_graph()
            CriticNet.kill_graph()
            print('Work done.')

