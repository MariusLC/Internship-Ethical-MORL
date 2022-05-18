import numpy as np
import math
import scipy.stats as st


class PreferenceLearner:
    def __init__(self, n_iter, warmup, d):
        self.n_iter = n_iter
        self.warmup = warmup
        self.d = d
        self.accept_rates = None
        self.deltas = []
        self.prefs = []

    def log_preference(self, delta, pref):
        self.deltas.append(delta)
        self.prefs.append(pref)

    def w_prior(self, w):
        if np.linalg.norm(w) <=1 and np.all(np.array(w) >= 0):
            return (2**self.d)/(math.pi**(self.d/2)/math.gamma(self.d/2 + 1))
        else:
            return 0

    def sample_w_prior(self, n):
        sample = np.random.rand(n, self.d)
        w_out = []
        for w in sample:
            w_out.append(list(w/np.linalg.norm(w)))
        return np.array(w_out)

    @staticmethod
    def f_loglik(w, delta, pref):
        return np.log(np.minimum(1, np.exp(pref*np.dot(w, delta)) + 1e-5))

    @staticmethod
    def vanilla_loglik(w, delta, pref):
        return np.log(1/(1+np.exp(-pref*np.dot(w, delta))))

    @staticmethod
    def propose_w_prob(w1, w2):
        q = st.multivariate_normal(mean=w1, cov=1).pdf(w2)
        return q

    @staticmethod
    def propose_w(w_curr):
        w_new = st.multivariate_normal(mean=w_curr, cov=1).rvs()
        return w_new

    def posterior_log_prob(self, deltas, prefs, w):
        f_logliks = []
        for i in range(len(prefs)):
            f_logliks.append(self.f_loglik(w, deltas[i], prefs[i]))
        loglik = np.sum(f_logliks)
        log_prior = np.log(self.w_prior(w) + 1e-5)

        return loglik + log_prior

    def mcmc_vanilla(self, w_init='mode'):
        if w_init == 'mode':
            w_init = [0 for i in range(self.d)]

        w_arr = []
        w_curr = w_init
        accept_rates = []
        accept_cum = 0

        for i in range(1, self.warmup + self.n_iter + 1):
            w_new = self.propose_w(w_curr)

            prob_curr = self.posterior_log_prob(self.deltas, self.prefs, w_curr)
            prob_new = self.posterior_log_prob(self.deltas, self.prefs, w_new)

            if prob_new > prob_curr:
                acceptance_ratio = 1
            else:
                qr = self.propose_w_prob(w_curr, w_new) / self.propose_w_prob(w_new, w_curr)
                acceptance_ratio = np.exp(prob_new - prob_curr) * qr
            acceptance_prob = min(1, acceptance_ratio)

            if acceptance_prob > st.uniform(0, 1).rvs():
                w_curr = w_new
                accept_cum = accept_cum + 1
                w_arr.append(w_new)
            else:
                w_arr.append(w_curr)

            accept_rates.append(accept_cum / i)

        self.accept_rates = np.array(accept_rates)[self.warmup:]

        return np.array(w_arr)[self.warmup:]


class VolumeBuffer:
    def __init__(self, auto_pref=True):
        self.auto_pref = auto_pref
        self.best_volume = -np.inf
        self.best_delta = None
        self.best_observed_returns = None
        self.best_returns = None
        self.observed_logs = []
        self.objective_logs = []

    def log_statistics(self, statistics):
        # print("objective LOG : ", statistics)
        self.objective_logs.append(statistics)

    def log_rewards(self, rewards):
        # print("observed LOG : ", rewards)
        self.observed_logs.append(rewards)

    @staticmethod
    def volume_removal(w_posterior, delta):
        expected_volume_a = 0
        expected_volume_b = 0
        for w in w_posterior:
            # print("w :",w)
            # print("delta :",delta)
            expected_volume_a += (1 - PreferenceLearner.f_loglik(w, delta, 1))
            expected_volume_b += (1 - PreferenceLearner.f_loglik(w, delta, -1))

        # print("expected_volume_a : ", expected_volume_a)
        # print("expected_volume_b : ", expected_volume_b)
        # print("len(w_posterior) = ", len(w_posterior)) # == batch size des rollout ?
        return min(expected_volume_a / len(w_posterior), expected_volume_b / len(w_posterior))

    def sample_return_pair(self):
        # len(self.observed_logs) = 74 donc tous les états de 1 trajectoire, car on a demandé un batchsize de 12 pour 12 workers ?
        # len(self.observed_logs[0]) = 12 donc tous les rewards des 12 workers à l'état 0 ?
        # len(self.observed_logs[0][0]) = 3 donc les rewards des 3 objectifs de l'état 0 pour le workers 0 ?
        # len(observed_logs_returns) = 12 donc on a fait la somme des rewards pour tous les états des trajectoires de chaque worker ?


        # observed_logs_returns est l'ensemble des vectorized_rewards, pour chaque état des trajectoires mémorisées. 
        # vectorized_rewards étant la liste des rewards pour chaque objectif selon chaque agent 
        # (vectorized_rewards[i] = liste des rewards pour l'objectif i, vectorized_rewards[i][j] = reward pour l'objectif i et l'expert j)
        # On fait ici la somme des vectorized_rewards pour tous les états mémorisé ?
        # On aurait donc une liste pour chaque objectif de sommes des rewards (effectifs, et d'experts) pour chaque état ?
        # print("len(self.observed_logs) = ", len(self.observed_logs))
        # print("len(self.observed_logs[0]) = ", len(self.observed_logs[0]))
        # print("len(self.observed_logs[0][0]) = ", len(self.observed_logs[0][0]))
        observed_logs_returns = np.array(self.observed_logs).sum(axis=0)
        # print("len(observed_logs_returns) = ", len(observed_logs_returns))
        # On choisit deux états parmi tous les états des trajec
        rand_idx = np.random.choice(np.arange(len(observed_logs_returns)), 2, replace=False)
        # print("rand_idx = ", rand_idx)

        # v2-Environment comparison
        #new_returns_a = observed_logs_returns[rand_idx[0]]
        #new_returns_b = observed_logs_returns[rand_idx[1]]

        # v3-Environment comparison (vase agnostic)
        # print("len(observed_logs_returns[rand_idx[0]]) = ", len(observed_logs_returns[rand_idx[0]]))
        new_returns_a = observed_logs_returns[rand_idx[0], 0:3]
        new_returns_b = observed_logs_returns[rand_idx[1], 0:3]
        # print("observed_logs_returns = ", observed_logs_returns)
        # print("observed_logs_returns 0 = ", new_returns_a)
        # print("observed_logs_returns 1 = ", new_returns_b)

        # Reset observed logs
        self.observed_logs = []

        # Also return ground truth logs for automatic preferences
        if self.auto_pref:
            objective_logs_returns = np.array(self.objective_logs).sum(axis=0)

            # v2-Environment comparison
            #logs_a = objective_logs_returns[rand_idx[0]]
            #logs_b = objective_logs_returns[rand_idx[1]]

            # v3-Environment comparison (vase agnostic)
            logs_a = objective_logs_returns[rand_idx[0], 0:3]
            logs_b = objective_logs_returns[rand_idx[1], 0:3]
            # print("objective_logs_returns = ", objective_logs_returns)
            # print("logs_a = ", logs_a)
            # print("logs_b = ", logs_b)

            self.objective_logs = []
            return np.array(new_returns_a), np.array(new_returns_b), logs_a, logs_b
        else:
            return np.array(new_returns_a), np.array(new_returns_b)

    def compare_delta(self, w_posterior, new_returns_a, new_returns_b, logs_a=None, logs_b=None, random=False):
        delta = new_returns_a - new_returns_b
        # print("delta = ", delta)
        volume_delta = self.volume_removal(w_posterior, delta)
        # print("volume_delta = ", volume_delta)
        # print("best_volume = ", self.best_volume)
        if volume_delta > self.best_volume or random:
            self.best_volume = volume_delta
            self.best_delta = delta
            self.best_observed_returns = (new_returns_a, new_returns_b)
            # print("self.best_observed_returns ", self.best_observed_returns)
            self.best_returns = (logs_a, logs_b)
            # print("self.best_returns ", self.best_returns)

    def reset(self):
        self.best_volume = -np.inf
        self.best_delta = None
        self.best_returns = None
        self.best_observed_returns = (None, None)

