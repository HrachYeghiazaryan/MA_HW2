"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import ttest_ind
from abc import ABC, abstractmethod
from logs import *
logger = logging.getLogger("MAB Application")
logger.setLevel(logging.DEBUG) # this on you need for you tests.

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.p = p

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def plot_performance(self): # you name it
        # Visualize the performance of each bandit
        pass


    @abstractmethod
    def report(self):
        # store the data in csv
        # print the average reward: using logging package
        # print average regret: using logging package
        pass

#--------------------------------------#
# I will leave epsilon as simple as possible being 1/t where t is the number of current trial.
# Also, in Thompson Sampling I will leave precision again as simple as possible, 1.
class EpsilonGreedy(Bandit):
    def __init__(self, p):
        super().__init__(p)
        self.p_estimate = 0
        self.N = 0

    def __repr__(self):
        """
        A description for the class.     
        """
        return 'A complete class to run Epsilon Greedy algorithm'

    def pull(self):
        """
        Performs a trial for the chosen bandit.
        """
        return np.random.randn() + self.p

    def update(self, res):
        """
        Based on the result passed (supposedly from the trial) updates the belief about the mean
        of the bandit.
        Inputs:
        res := the result of the trial based on which we want to update the belief about the mean
        """
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + res) / self.N
    
    def experiment(self, bandit_reward=[1,2,3,4], num_trials=20000, seed=42, verbose=True):
        """
        Here we perform the experiment and observe some results about the performance of 
        the algorithm.
        Inputs:
        bandit_reward := The actual rewards that bandits have
        num_trials := The number of trials to be performed
        seed := random seed
        verbose := set true to print the results
        """
        bandits = [EpsilonGreedy(reward) for reward in bandit_reward]
        true_best = np.argmax(bandit_reward)
        count_suboptimal = 0
        lr_ls = []
        reward_ls = []
        b_ls = []
        np.random.seed(seed=seed)

        for i in range(1,num_trials+1):
            eps = 1/i 
            p = np.random.random()           
            if p<eps:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            band_max = np.argmax([b.p_estimate for b in bandits])
            exp_reward = (1-eps)*bandits[band_max].p_estimate + eps*((sum([bandit.p_estimate for bandit in bandits])-bandits[band_max].p_estimate)/(len(bandits)-1))
            lr_ls.append(exp_reward) 
            b_ls.append(j)

            res = bandits[j].pull()
            bandits[j].update(res)

            if j!= true_best:
                count_suboptimal += 1
            
            reward_ls.append(res)
        cumulative_average = np.cumsum(reward_ls)/(np.arange(num_trials)+1)
        estimated_avg_rewards = [bandit.p_estimate for bandit in bandits]
        if verbose:
            logger.info(f'Estimated average rewards: {estimated_avg_rewards}')
            logger.info(f'Percent suboptimal : {float(count_suboptimal)/ num_trials}')
            logger.info("--------------------------------------------------")
        self.cumulative_average = cumulative_average
        self.cumulative_reward = np.cumsum(reward_ls)
        self.learning = lr_ls
        self.bandit_choices = b_ls
        self.rewards = reward_ls
        self.regret = count_suboptimal
        self.best_bandit_index = np.argmax([b.p_estimate for b in bandits])
    
    def plot_performance(self):
        """
        Plots the cumulative average.
        """
        plt.plot(self.cumulative_average, label=f'harmonically decaying epsilon')
        plt.title('Win rate convergence for Epsilon Greedy')
        plt.xlabel('Number of trials')
        plt.ylabel('Estimated reward')
        plt.show()

    def report(self, verbose=True, df_save_path=None):
        """
        Plots the statistics and outputs a dataframe with columns [Bandit, Reward, Algorithm]
        Inputs:
        verbose := set False to only return the dataframe.
        """
        if verbose:
            fig, (ax1, ax2) = plt.subplots(2,1)
            fig.set_size_inches(10,10)
            ax1.plot(self.learning)
            ax1.set_title('The learning rate of Epsilon Greedy with epsilon 1/t')
            ax1.set_xlabel('trial')
            ax1.set_ylabel('expected reward')
            ax2.plot(self.cumulative_reward)
            ax2.set_title('The cumulative reward of Epsilon Greedy with epsilon 1/t')
            ax2.set_xlabel('number of trials')
            ax2.set_ylabel('cumulative reward')
            logger.info(f'Cumulative reward: {np.sum(self.rewards)}')
            logger.info(f'Cumulative regret: {self.regret}')
        df_dict = {'Bandit': self.bandit_choices, 'Reward': self.rewards, 'Algorithm': 'Epsilon-Greedy'}
        df = pd.DataFrame(df_dict)
        if df_save_path!=None:
            df.to_csv(df_save_path)
        return df


#--------------------------------------#
class ThompsonSampling(Bandit):
    def __init__(self, true_mean):
        super().__init__(true_mean)
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0
        self.cum_reward = 0

    def __repr__(self):
        """
        A description for the class.     
        """
        return 'A complete class to run Thompson Sampling algorithm'

    def pull(self):
        """
        Performs a trial for the chosen bandit.
        """
        return (np.random.randn()/np.sqrt(self.tau)) + self.p

    def sample(self):
        """
        Draws a sample from the posterior distribution of the mean for the chosen bandit.
        """
        return (np.random.randn()/np.sqrt(self.lambda_)) + self.m

    def update(self, res):
        """
        Based on the result passed (supposedly from the trial) updates the belief about the mean
        of the bandit.
        Inputs:
        res := the result of the trial based on which we want to update the belief about the mean
        """
        self.lambda_ += self.tau
        self.cum_reward += res
        self.m = (self.tau*self.cum_reward)/self.lambda_
        self.N += 1
    
    def __plot_distributions(self, bandits, num_trials):
        """
        Plots the posterior distributions of rewards for each bandit when a particular number
        of trials is done.
        A method not to be called outside the class, for drawing the plot for particular number
        of trials please pass the number in the list for num_samples_plot in method experiment.
        Inputs:
        bandits := the list of bandits
        num_trials := the posterior distributions to plot based on this number of trials
        """
        x = np.linspace(-3, 8, 400)
        for bandit in bandits:
            y = norm.pdf(x, bandit.m, np.sqrt(1/bandit.lambda_))
            plt.plot(x, y, label=f"True Mean: {bandit.p:.2f}, Num of Trials: {bandit.N}")
            plt.title(f"Beliefs of bandit reward distributions after {num_trials} trials")
        plt.legend()
        plt.show()

    def experiment(self, bandit_means=[1,2,3,4], num_trials=20000, 
                         num_samples_plot =[10,50,100,200,500,1000,5000,10000,20000], seed=42, verbose=True):
        """
        Here we perform the experiment and observe some results about the performance of 
        the algorithm.
        Inputs:
        bandit_means := The actual means of reward distributions that bandits have
        num_trials := The number of trials to be performed
        num_samples_plot := will plot the posterior distributions of rewards for each bandit after each number of trials from the list
        seed := random seed
        verbose := set true to print the results
        """
        bandits = [ThompsonSampling(mean) for mean in bandit_means]
        true_best = np.argmax(bandit_means)
        rewards = []
        lr_ls = []
        b_ls = []
        count_suboptimal = 0
        np.random.seed(seed=seed)
        for i in range(1,num_trials+1):
            j = np.argmax([bandit.sample() for bandit in bandits])
            lr_ls.append(bandits[j].m)
            b_ls.append(j)
            if j!=true_best:
                count_suboptimal += 1

            res = bandits[j].pull()
            bandits[j].update(res)
            rewards.append(res)

            if (i in num_samples_plot) & verbose:
                self.__plot_distributions(bandits, i)
        
        self.cumulative_average = np.cumsum(rewards)/(np.arange(num_trials)+1)
        self.learning = lr_ls
        self.cumulative_reward = np.cumsum(rewards)
        self.bandit_choices = b_ls
        self.rewards = rewards
        self.regret = count_suboptimal
        self.best_bandit_index = np.argmax([bandit.sample() for bandit in bandits])

    def plot_performance(self):
        """
        Plots the cumulative average.
        """
        plt.plot(self.cumulative_average, label=f'harmonically decaying epsilon')
        plt.title('Win rate convergence for Thompson Sampling')
        plt.xlabel('Number of trials')
        plt.ylabel('Estimated reward')
        plt.show()

    def report(self, verbose=True, df_save_path=None):
        """
        Plots the statistics and outputs a dataframe with columns [Bandit, Reward, Algorithm]
        Inputs:
        verbose := set False to only return the dataframe.
        """
        if verbose:
            fig, (ax1, ax2) = plt.subplots(2,1)
            fig.set_size_inches(10,10)
            ax1.plot(self.learning)
            ax1.set_title('The learning rate of Thompson sampling')
            ax1.set_xlabel('trial')
            ax1.set_ylabel('expected reward')
            ax2.plot(self.cumulative_reward)
            ax2.set_title('The cumulative reward of Thompson sampling')
            ax2.set_xlabel('number of trials')
            ax2.set_ylabel('cumulative reward')
            logger.info(f'Cumulative reward: {np.sum(self.rewards)}')
            logger.info(f'Cumulative regret: {self.regret}')
        df_dict = {'Bandit': self.bandit_choices, 'Reward': self.rewards, 'Algorithm': 'Thompson-Sampling'}
        df = pd.DataFrame(df_dict)
        if df_save_path!=None:
            df.to_csv(df_save_path)
        return df

# Since the algorithms have the purpose to first of all find the better arm, therefore,
# the function first checks whose best arm has higher reward.
# If the two algorithms return arms that have equal true reward as optimal,the next most 
# important thing from business perspective is how much money did the business 
# lose due to the exploration part. It isn't a good idea to look on the number of regrets, since in 
# business the money is the most important and therefore we will pay attention to the cumulative gains 
# on each step. Then, to be formal we will conduct a t-test whether the mean of rewards from epsilon
#greedy is not equal to mean of rewards from Thompson sampling.

def comparison(bandit_rewards = [1,2,3,4], num_trials=20000, seed=42): 
    """
    A function that compares the Epsilon Greedy and Thompson Sampling algorithms.
    Inputs:
    bandit_reward := The actual rewards that bandits have
    num_trials := The number of trials to be performed
    seed := random seed
    """
    epsilon_greedy = EpsilonGreedy(1)
    epsilon_greedy.experiment(bandit_reward=bandit_rewards, num_trials=num_trials, seed=seed, verbose=False)
    df_epsilon = epsilon_greedy.report(verbose=False)

    thompson_sampling = ThompsonSampling(1)
    thompson_sampling.experiment(bandit_means=bandit_rewards, num_trials=num_trials, seed=seed, verbose=False)
    df_thompson = thompson_sampling.report(verbose=False)
    epsilon_best_reward = bandit_rewards[epsilon_greedy.best_bandit_index]
    thompson_best_reward = bandit_rewards[thompson_sampling.best_bandit_index]
    if thompson_best_reward>epsilon_best_reward:
        logger.info('Thompson sampling has won, since it found a better arm.')
    elif epsilon_best_reward>thompson_best_reward:
        logger.info('Epsilon greedy has won, since it found a better arm.')
    else:
        logger.info('The best arms for both algorithms are equal. A further investigation is needed')
        epsilon_rewards = df_epsilon.loc[:,'Reward'].tolist()
        thompson_rewards = df_thompson.loc[:,'Reward'].tolist()
        epsilon_cumsum = np.cumsum(epsilon_rewards)
        thompson_cumsum = np.cumsum(thompson_rewards)
        diff_ls = epsilon_cumsum-thompson_cumsum
        plt.plot(diff_ls)
        plt.title('Cumsum(Epsilon Greedy rewards)-Cumsum(Thompson Sampling rewards)')
        plt.xlabel('Number of trials')
        plt.ylabel('Difference of cumulative sums')
        plt.show()
        logger.info('T-test results whether the mean of rewards from Epsilon Greedy is the same as of the Thompson Sampling (alternative hypothesis mean(Epsilon Greedy)!=mean(Thompson Sampling))')
        logger.info(ttest_ind(epsilon_rewards, thompson_rewards, alternative='two-sided'))



# if __name__=='__main__':
   
#     logger.debug("debug message")
#     logger.info("info message")
#     logger.warning("warning message")
#     logger.error("error message")
#     logger.critical("critical message")
