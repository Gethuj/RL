import numpy as np
from Actor import DRLactor
from Critic import DRLcritic
# from Environment_oldVersion import Hypothesis
from environment_Small import Hypothesis
import tensorflow as tf
import copy

np.random.seed(None)
tf.reset_default_graph()


if __name__ == "__main__":
    lbd_list = [0]# [0, 0.0005, 0.005, 0.05, 0.1, 0.2, 0.5]
    rho_list = [0, 0.2, 0.4, 0.6, 0.8,1]

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
            ActionPerTime = 1
            N_features = 2*N_process

            ActorNet = DRLactor(n_features=N_process, n_actions=N_process, lr=LR_A, num=ActionPerTime, lbd=lbd, rho=rho)
            CriticNet = DRLcritic(n_features=N_process, lr=LR_C, num=ActionPerTime, lbd=lbd, rho=rho)

            Env = Hypothesis(N_process, [ActionPerTime], rho)
            WrongCount = 0
            for EpisodeNo in range(TotalEpisodes):
                CurrentSigma = [Env.Pf]*N_process
                
                CurrentEst = Env.Cal_NegEntropy(CurrentSigma)
                
                CurrentHypothesis = Env.H_generator()
                CountActions = [0]*N_actions
                for N in range(MaxTimeSteps):
#                    PrevPDF = copy.deepcopy(CurrentPDF)
                    PrevSigma = copy.deepcopy(CurrentSigma)
                    PrevEst = copy.deepcopy(CurrentEst)
                    
                    CurrentAction = ActorNet.choose_action([PrevSigma], delta_epsilon**(EpisodeNo*MaxTimeSteps + N))
                    CountActions[CurrentAction] += 1
                    Observation = Env.Observe(CurrentAction, CurrentHypothesis)
                    
                    CurrentSigma = Env.UpdateSigma(PrevSigma, CurrentAction, Observation)
                    CurrentEst = Env.Cal_NegEntropy(CurrentSigma)
                    Reward = CurrentEst - PrevEst
                    
                    TDError = CriticNet.learn([PrevSigma], Reward, [CurrentSigma], EpisodeNo)
                    ActorNet.learn([PrevSigma], CurrentAction, TDError, EpisodeNo)
                
                EstimateList = [0]*N_process
                for hypo in range(N_process):
                    EstimateList[hypo] = np.argmax([CurrentSigma[hypo], 1-CurrentSigma[hypo]])
                HypothesisEstimate = int("".join(str(x) for x in EstimateList), 2)    
                print(EpisodeNo, HypothesisEstimate-CurrentHypothesis, CurrentHypothesis, CountActions)
                if HypothesisEstimate-CurrentHypothesis != 0:
                    WrongCount += 1
            print(WrongCount)
            CriticNet.save_model()
            ActorNet.save_model()
            ActorNet.kill_graph()
            CriticNet.kill_graph()
            print('Work done.')

