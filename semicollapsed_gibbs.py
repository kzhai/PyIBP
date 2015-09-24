import numpy, scipy;
import math, random;
import scipy.stats;
from monte_carlo import MonteCarlo;
from uncollapsed_gibbs import UncollapsedGibbs;

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class SemiCollapsedGibbs(UncollapsedGibbs):
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

        self.sample_A();
        
        if self._alpha_hyper_parameter != None:
            self._alpha = self.sample_alpha();
        
        if self._sigma_x_hyper_parameter != None:
            self._sigma_x = self.sample_sigma_x(self._sigma_x_hyper_parameter);
        
        if self._sigma_a_hyper_parameter != None:
            self._sigma_a = self.sample_sigma_a(self._sigma_a_hyper_parameter);
            
        return self.log_likelihood_model();

    """
    sample K_new using metropolis hastings algorithm
    """
    def metropolis_hastings_K_new(self, object_index, singleton_features):
        if type(object_index) != list:
            object_index = [object_index];
    
        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_new = scipy.stats.poisson.rvs(self._alpha / self._N);
        K_old = len(singleton_features);
        
        if K_new <= 0 and K_old <= 0:
            return False;
        
        Z_i_tmp = self._Z[object_index, :];
        Z_i_tmp[:, singleton_features] = 0;
        X_residue = self._X[object_index, :] - numpy.dot(Z_i_tmp, self._A);

        log_new_old = 0;
        for d in xrange(self._D):
            log_new_old -= 0.5 * numpy.log((self._sigma_x ** 2 + K_new * self._sigma_a ** 2) / (self._sigma_x ** 2 + K_old * self._sigma_a ** 2));
            log_new_old -= 0.5 * X_residue[0, d] ** 2 * (1 / (self._sigma_x ** 2 + K_new * self._sigma_a ** 2) - 1 / (self._sigma_x ** 2 + K_old * self._sigma_a ** 2));
            
        accept_new = 1.0 / (1.0 + 1.0 / numpy.exp(log_new_old));

        self._A = self._A[[k for k in xrange(self._K) if k not in singleton_features], :];
        self._Z = self._Z[:, [k for k in xrange(self._K) if k not in singleton_features]];
        self._K -= K_old
        
        # if we accept the proposal, we will replace old A and Z matrices
        if random.random() > accept_new:
            K_new = K_old;
        
        if K_new > 0:
            # construct A_new and Z_new
            Z_new = numpy.zeros((self._N, K_new));
            Z_i_new = numpy.ones((1, K_new));
            Z_new[object_index, :] = Z_i_new;
            M_new = self.compute_M(Z_i_new);
            A_new = numpy.dot(M_new, numpy.dot(Z_new.transpose(), self._X - numpy.dot(self._Z, self._A)));
            self._A = numpy.vstack((self._A, A_new));
            self._Z = numpy.hstack((self._Z, Z_new));
            #self._Z[object_index, :] = Z_i_new;
            self._K += K_new
            return True;

        return False;
    
        '''
        # compute the log likelihood if we use old features
        Z_i_old = numpy.ones((1, K_old));
        M_old = self.compute_M(Z_i_old);
        assert(M_old.shape==(K_old, K_old));
        log_likelihood_old = 1-numpy.dot(numpy.dot(Z_i_old, M_old), Z_i_old.transpose());
        log_likelihood_old = -numpy.trace(numpy.dot(numpy.dot(X_residue.transpose(), log_likelihood_old), X_residue));
        log_likelihood_old /= (2 * self._sigma_x**2);
        log_likelihood_old += self._D / 2 * numpy.linalg.det(M_old);
        log_likelihood_old -= (1-K_old)*self._D * numpy.log(self._sigma_x) + (K_old*self._D) * numpy.log(self._sigma_a);
        
        # compute the log likelihood if we use new features
        Z_i_new = numpy.ones((1, K_new));
        M_new = self.compute_M(Z_i_new);
        assert(M_new.shape==(K_new, K_new));
        log_likelihood_new = 1-numpy.dot(numpy.dot(Z_i_new, M_new), Z_i_new.transpose());
        log_likelihood_new = -numpy.trace(numpy.dot(numpy.dot(X_residue.transpose(), log_likelihood_new), X_residue));
        log_likelihood_new /= (2 * self._sigma_x**2);
        log_likelihood_new += self._D / 2 * numpy.linalg.det(M_new);
        log_likelihood_new -= (1-K_new)*self._D * numpy.log(self._sigma_x) + (K_new*self._D) * numpy.log(self._sigma_a);
        
        # compute the probability of accepting new features                
        accept_new = 1.0/(1.0 + numpy.exp(log_likelihood_old-log_likelihood_new));
        '''
