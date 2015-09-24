import numpy, scipy;
import math, random;
import scipy.stats;
from monte_carlo import MonteCarlo

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class UncollapsedGibbs(MonteCarlo):
    """
    sample the corpus to train the parameters
    """
    def learning(self):
        self._counter += 1;
        
        assert(self._Z.shape == (self._N, self._K));
        assert(self._A.shape == (self._K, self._D));
        assert(self._X.shape == (self._N, self._D));
        
        # sample every object
        order = numpy.random.permutation(self._N);
        for (object_counter, object_index) in enumerate(order):
            # sample Z_n
            singleton_features = self.sample_Zn(object_index);
            
            if self._metropolis_hastings_k_new:
                # sample K_new using metropolis hasting
                self.metropolis_hastings_K_new(object_index, singleton_features);
            
        # regularize matrices
        self.regularize_matrices();    

        self.sample_A();
        
        if self._alpha_hyper_parameter != None:
            self._alpha = self.sample_alpha();
        
        if self._sigma_x_hyper_parameter != None:
            self._sigma_x = self.sample_sigma_x(self._sigma_x_hyper_parameter);
        
        if self._sigma_a_hyper_parameter != None:
            self._sigma_a = self.sample_sigma_a(self._sigma_a_hyper_parameter);
        
        return self.log_likelihood_model();
          
    """
    @param object_index: an int data type, indicates the object index (row index) of Z we want to sample
    """
    def sample_Zn(self, object_index):
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64);
        
        # calculate initial feature possess counts
        m = self._Z.sum(axis=0);
        
        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(numpy.float);
        
        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(new_m / self._N);
        log_prob_z0 = numpy.log(1.0 - new_m / self._N);
        
        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0];
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]
        
        order = numpy.random.permutation(self._K);
        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                #old_Znk = self._Z[object_index, feature_index];

                # compute the log likelihood when Znk=0
                self._Z[object_index, feature_index] = 0;
                prob_z0 = self.log_likelihood_X(self._X[[object_index], :], self._Z[[object_index], :]);
                prob_z0 += log_prob_z0[feature_index];
                prob_z0 = numpy.exp(prob_z0);
                
                # compute the log likelihood when Znk=1
                self._Z[object_index, feature_index] = 1;
                prob_z1 = self.log_likelihood_X(self._X[[object_index], :], self._Z[[object_index], :]);
                prob_z1 += log_prob_z1[feature_index]
                prob_z1 = numpy.exp(prob_z1);
                
                Znk_is_0 = prob_z0 / (prob_z0 + prob_z1);
                if random.random() < Znk_is_0:
                    self._Z[object_index, feature_index] = 0;
                else:
                    self._Z[object_index, feature_index] = 1;
                    
        return singleton_features;

    """
    sample K_new using metropolis hastings algorithm
    """
    def metropolis_hastings_K_new(self, object_index, singleton_features):
        if type(object_index) != list:
            object_index = [object_index];
    
        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = scipy.stats.poisson.rvs(self._alpha / self._N);
        
        if K_temp <= 0 and len(singleton_features) <= 0:
            return False;

        # generate new features from a normal distribution with mean 0 and variance sigma_a, a K_new-by-D matrix
        A_prior = numpy.tile(self._A_prior, (K_temp, 1));
        A_temp = numpy.random.normal(0, self._sigma_a, (K_temp, self._D)) + A_prior;
        A_new = numpy.vstack((self._A[[k for k in xrange(self._K) if k not in singleton_features], :], A_temp));
        # generate new z matrix row
        #print K_temp, object_index, [k for k in xrange(self._K) if k not in singleton_features], self._Z[[object_index], [k for k in xrange(self._K) if k not in singleton_features]].shape, numpy.ones((len(object_index), K_temp)).shape
        Z_new = numpy.hstack((self._Z[[object_index], [k for k in xrange(self._K) if k not in singleton_features]], numpy.ones((len(object_index), K_temp))));
        
        K_new = self._K + K_temp - len(singleton_features);
        
        # compute the probability of generating new features
        prob_new = numpy.exp(self.log_likelihood_X(self._X[object_index, :], Z_new, A_new));
        
        # construct the A_old and Z_old
        A_old = self._A;
        Z_old = self._Z[object_index, :];
        K_old = self._K;

        assert(A_old.shape == (K_old, self._D));
        assert(A_new.shape == (K_new, self._D));
        assert(Z_old.shape == (len(object_index), K_old));
        assert(Z_new.shape == (len(object_index), K_new));
        
        # compute the probability of using old features
        prob_old = numpy.exp(self.log_likelihood_X(self._X[object_index, :], Z_old, A_old));
        
        # compute the probability of generating new features
        prob_new = prob_new / (prob_old + prob_new);
        
        # if we accept the proposal, we will replace old A and Z matrices
        if random.random() < prob_new:
            # construct A_new and Z_new
            self._A = A_new;
            self._Z = numpy.hstack((self._Z[:, [k for k in xrange(self._K) if k not in singleton_features]], numpy.zeros((self._N, K_temp))));
            self._Z[object_index, :] = Z_new;
            self._K = K_new;
            return True;

        return False;

    """
    """
    def sample_A(self):
        # sample every feature
        order = numpy.random.permutation(self._D);
        for (observation_counter, observation_index) in enumerate(order):
            # sample A_d
            (mean, std_dev) = self.sufficient_statistics_A([observation_index]);
            assert(std_dev.shape == (self._K, self._K));
            assert(mean.shape == (self._K, len([observation_index])));
            self._A[:, [observation_index]] = numpy.dot(std_dev, numpy.random.normal(0, 1, (self._K, len([observation_index])))) + mean;
        
        return
    
    """
    compute the mean and co-variance, i.e., sufficient statistics, of A
    @param observation_index: a list data type, recorded down the observation indices (column numbers) of A we want to compute
    """
    def sufficient_statistics_A(self, observation_index=None):
        if observation_index == None:
            X = self._X;
            observation_index = range(self._D);
        else:
            X = self._X[:, observation_index]
        
        assert(type(observation_index) == list);
        
        D = X.shape[1];
        #mean_a = numpy.zeros((self._K, D));
        #for k in range(self._K):
        #    mean_a[k, :] = self._mean_a[0, observation_index];
        A_prior = numpy.tile(self._A_prior[0, observation_index], (self._K, 1));

        assert(X.shape == (self._N, D));
        assert(self._Z.shape == (self._N, self._K));
        assert(A_prior.shape == (self._K, D))
        
        # compute M = (Z' * Z - (sigma_x^2) / (sigma_a^2) * I)^-1
        M = self.compute_M();
        # compute the mean of the matrix A
        mean_A = numpy.dot(M, numpy.dot(self._Z.transpose(), X) + (self._sigma_x / self._sigma_a) ** 2 * A_prior);
        # compute the co-variance of the matrix A
        std_dev_A = numpy.linalg.cholesky(self._sigma_x ** 2 * M).transpose();
        
        return (mean_A, std_dev_A)
    
    """
    remove the empty column in matrix Z and the corresponding feature in A
    """
    def regularize_matrices(self):
        assert(self._Z.shape == (self._N, self._K));
        Z_sum = numpy.sum(self._Z, axis=0);
        assert(len(Z_sum) == self._K);
        indices = numpy.nonzero(Z_sum == 0);
        #assert(numpy.min(indices)>=0 and numpy.max(indices)<self._K);
        
        #print self._K, indices, [k for k in range(self._K) if k not in indices]
        self._Z = self._Z[:, [k for k in range(self._K) if k not in indices]];
        self._A = self._A[[k for k in range(self._K) if k not in indices], :];
        
        self._K = self._Z.shape[1];
        assert(self._Z.shape == (self._N, self._K));
        assert(self._A.shape == (self._K, self._D));

    """
    compute the log-likelihood of the data X
    @param X: a 2-D numpy array
    @param Z: a 2-D numpy boolean array
    @param A: a 2-D numpy array, integrate A out if it is set to None
    """
    def log_likelihood_X(self, X=None, Z=None, A=None):
        if A == None:
            A = self._A;
        if Z == None:
            Z = self._Z;
        if X == None:
            X = self._X;
            
        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape;
        assert(A.shape == (K, D));
        
        log_likelihood = X - numpy.dot(Z, A);
        
        (row, column) = log_likelihood.shape;
        if row > column:
            log_likelihood = numpy.trace(numpy.dot(log_likelihood.transpose(), log_likelihood));
        else:
            log_likelihood = numpy.trace(numpy.dot(log_likelihood, log_likelihood.transpose()));
        
        log_likelihood = -0.5 * log_likelihood / numpy.power(self._sigma_x, 2);
        log_likelihood -= N * D * 0.5 * numpy.log(2 * numpy.pi * numpy.power(self._sigma_x, 2));
                       
        return log_likelihood
    
    """
    compute the log-likelihood of A
    """
    def log_likelihood_A(self):
        log_likelihood = -0.5 * self._K * self._D * numpy.log(2 * numpy.pi * self._sigma_a * self._sigma_a);
        #for k in range(self._K):
        #    A_prior[k, :] = self._mean_a[0, :];
        A_prior = numpy.tile(self._A_prior, (self._K, 1))
        log_likelihood -= numpy.trace(numpy.dot((self._A - A_prior).transpose(), (self._A - A_prior))) * 0.5 / (self._sigma_a ** 2);
        
        return log_likelihood;
    
    """
    compute the log-likelihood of the model
    """
    def log_likelihood_model(self):
        #print self.log_likelihood_X(self._X, self._Z, self._A), self.log_likelihood_A(), self.log_likelihood_Z();
        return self.log_likelihood_X() + self.log_likelihood_A() + self.log_likelihood_Z();
