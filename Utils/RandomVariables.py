import enum
from abc import ABC
import numpy as np

class Activations(enum.Enum):
    SIGMOID = 1
    LINEAR = 2
    TANH = 3
    
class RandomVariable(ABC):
    def __init__(self, onlySampleOnce = True):
        self.MostRecentSample = None
        self.SampleOnce = onlySampleOnce
    
    def ValidateRV(rv):
        if(isinstance(rv, RandomVariable)):
            return rv

        if(type(rv) in [float,complex,int] or np.issubdtype(rv, np.number)):
            return ConstantRV(rv)
    
        raise TypeError()
    
    def __add__(self, other_rv):
        other_rv = RandomVariable.ValidateRV(other_rv)
        
        if(type(other_rv) == AdditionCombinationRV):
            return other_rv + self
        
        return AdditionCombinationRV([self, other_rv])
    
    def __radd__(self, other_rv):
        return self.__add__(other_rv)
    
    def __sub__(self, other_rv):
        other_rv = RandomVariable.ValidateRV(other_rv)
        return AdditionCombinationRV([self, other_rv * -1])
    
    def __rsub__(self, other_rv):
        other_rv = RandomVariable.ValidateRV(other_rv)
        return AdditionCombinationRV([-1 * self, other_rv])
    
    def __mul__(self, other_rv):
        other_rv = RandomVariable.ValidateRV(other_rv)
        return MultipliedRV(self, other_rv)

    def __rmul__(self, other_rv):
        return self.__mul__(other_rv)

    def __truediv__(self, other_rv):
        return self * GeomInverseRV(other_rv)
        
    def __rtruediv__(self, other_rv):
        return other_rv * GeomInverseRV(self)

    def Sample(self):
        if(not self.SampleOnce or self.MostRecentSample is None):
            self.MostRecentSample = self._sample_logic()
        return self.MostRecentSample
        
    def _sample_logic(self):
        raise NotImplementedError('Subclass must implement')

class GeomInverseRV(RandomVariable):
    def __init__(self, rv):
        self.RV = RandomVariable.ValidateRV(rv)
        super().__init__()
        
    def _sample_logic(self):
        return 1/self.RV.Sample()

class MultipliedRV(RandomVariable):
    def __init__(self, rv_left: RandomVariable, rv_right: RandomVariable):
        self.Left = RandomVariable.ValidateRV(rv_left)
        self.Right = RandomVariable.ValidateRV(rv_right)
        super().__init__()

    def _sample_logic(self):
        return self.Left.Sample() * self.Right.Sample()

class AdditionCombinationRV(RandomVariable):
    def __init__(self, rvs):
        self.RVs = rvs
        super().__init__()
        
    def __add__(self, other_rv):
        other_rv = RandomVariable.ValidateRV(other_rv)
        self.RVs.append(other_rv)
        return AdditionCombinationRV(self.RVs)
    
    def _sample_logic(self):
        return np.sum([rv.Sample() for rv in self.RVs])

class ConstantRV(RandomVariable):
    def __init__(self, const):
        self.Constant = const
        super().__init__()
    
    def _sample_logic(self):
        if(isinstance(self.Constant,RandomVariable)):
            return self.Constant.Sample()
        return self.Constant
    
class NormalRV(RandomVariable):
    def __init__(self, mu = 0, var = 1):
        self.Mu = RandomVariable.ValidateRV(mu)
        self.Variance = RandomVariable.ValidateRV(var)
        super().__init__()
        
    def _sample_logic(self):
        return np.random.normal(self.Mu.Sample(), self.Variance.Sample())
    
class UniformRV(RandomVariable):
    def __init__(self, lower = 0, upper = 1):
        self.Lower = RandomVariable.ValidateRV(lower)
        self.Upper = RandomVariable.ValidateRV(upper)
        super().__init__()
        
    def _sample_logic(self):
        return np.random.uniform(self.Lower.Sample(), self.Upper.Sample())

class BernoulliRV(RandomVariable):
    def __init__(self, z = .5, activation = Activations.SIGMOID):
        if(activation not in [Activations.SIGMOID, Activations.LINEAR]):
            raise NotImplementedError('Have not implemented anything outside of sigmoid and linear activation for Bernoulli')
        self.Zval = RandomVariable.ValidateRV(z)
        self.Activation = activation
        super().__init__()
    
    def _sample_logic(self):
        pVal = None
        if(self.Activation == Activations.LINEAR):
            pVal = max(min(self.Zval.Sample(), 1),0)
        elif(self.Activation == Activations.SIGMOID):
            pVal = 1/(1 + np.exp(self.Zval.Sample() * -1))
        return np.random.binomial(p=pVal,n=1)
    
class PoissonRV(RandomVariable):
    def __init__(self, lambda_param):
        self.Lambda = RandomVariable.ValidateRV(lambda_param)
        super().__init__()
    
    def _sample_logic(self):
        return np.random.poisson(np.exp(self.Lambda.Sample()))
    
class NegativeBinomialRV(RandomVariable):
    def __init__(self, mu, shape, activation=Activations.SIGMOID):
        if(activation not in [Activations.SIGMOID, Activations.LINEAR]):
            raise NotImplementedError('Have not implemented anything outside of sigmoid and linear activation for Bernoulli')
        self.Mu = RandomVariable.ValidateRV(mu)
        self.Shape = RandomVariable.ValidateRV(shape)
        super().__init__()
    
    def _sample_logic(self):
        mu = np.exp(self.Mu.Sample())
        n = self.Shape.Sample()
    
        if(n <= 0):
            return 0
        
        pVal = n / (mu + n)
        return int(np.floor(np.random.negative_binomial(n = n, p = pVal) + n))