import pandas as pd
import numpy as np
from Utils.RandomVariables import *

def SampleDriver(SampleHandler, sampleSize = 100, seed = 0):
    np.random.seed(seed)
    return pd.DataFrame([SampleHandler() for _ in range(sampleSize)])

def GenerateSimpleDataSet_OLS(sample_size = 5, seed = 0):
    def GenerateSample():
        x1 = UniformRV(ConstantRV(-5), ConstantRV(5))
        y = NormalRV(.75 * x1, ConstantRV(1))
        y.Sample();
        
        return {
            'X1': x1.MostRecentSample,
            'Y': y.MostRecentSample 
        }

    return SampleDriver(GenerateSample, sampleSize=sample_size, seed = seed)

def GenerateSimpleDataSet_NegativeBinomial(sample_size, seed = 0):
    def GenerateSample():
        x1 = UniformRV(-5, 5)
        x2 = UniformRV(20, 30)
        x3 = NormalRV(1, 6)
        x4 = NormalRV(10, 2)
        x5 = BernoulliRV(.7, Activations.LINEAR)
        x6 = BernoulliRV(.1, Activations.LINEAR)
        y = NegativeBinomialRV((x1 / 5 + (x2-20) / -30 + x3 / -12 + -1 * (x4-10) + x5 + x6 * 3), .5)
        y.Sample();
        
        return {
            'X1': x1.MostRecentSample,
            'X2': x2.MostRecentSample,
            'X3': x3.MostRecentSample,
            'X4': x4.MostRecentSample,
            'X5': x5.MostRecentSample,
            'X6': x6.MostRecentSample,
            'Z': (x1 / 5 + (x2-20) / -30 + x3 / -12 + -1 * (x4-10) + x5 + x6 * 3).Sample(),
            'Y': y.MostRecentSample 
        }

    return SampleDriver(GenerateSample, sampleSize=sample_size, seed = seed)

def GenerateSimpleDataSet_Bernoulli(sample_size, seed = 0):
    def GenerateSample():
        x0 = UniformRV(-1, 1)
        x1 = UniformRV(50, 100)
        x2 = UniformRV(0, 500)
        x3 = NormalRV(50, 20)
        x4 = NormalRV(30, 2)
        x5 = BernoulliRV(.7, Activations.LINEAR)
        x6 = BernoulliRV(.1, Activations.LINEAR)
        x7 = BernoulliRV(.4, Activations.LINEAR)
        x8 = BernoulliRV(.5, Activations.LINEAR)
        x9 = BernoulliRV(.2, Activations.LINEAR)
        y = BernoulliRV(x0 * .01 + (x1 - 75)/50 * -1 + (x2-250)/250 * -.5 + (x3-50)/np.sqrt(20) * 1.24 + (x4-30)/np.sqrt(2) + x5 * 1.5 + x6 * -1.2 + x7 * 2 + x8 * -3 + x9 * 1.8)
        y.Sample();
        
        return {
            'X1': x1.MostRecentSample,
            'X2': x2.MostRecentSample,
            'X3': x3.MostRecentSample,
            'X4': x4.MostRecentSample,
            'X5': x5.MostRecentSample,
            'X6': x6.MostRecentSample,
            'X7': x6.MostRecentSample,
            'X8': x6.MostRecentSample,
            'X9': x6.MostRecentSample,
            'Y': y.MostRecentSample 
        }

    return SampleDriver(GenerateSample, sampleSize=sample_size, seed = seed)

def GenerateSimpleDataSet_Regularization(sample_size, seed = 0):
    def GenerateSample():
        x1 = NormalRV(15, 3)
        x2 = NormalRV(30, 10)
        x3 = NormalRV(4, 1)
        x4 = NormalRV(8, 20)
        x5 = PoissonRV(np.log(6))
        y = NormalRV(x1 * 2 + x2 * -.7 + x2 * 3 + x3 * 1.5 + x4 * -1.3 + x5 * 1,3)
        y.Sample();
        ret = {
            'X1': x1.MostRecentSample,
            'X2': x2.MostRecentSample,
            'X3': x3.MostRecentSample,
            'X4': x4.MostRecentSample,
            'X5': x5.MostRecentSample
        }
        
        for i in (np.arange(20) + 6):
            ret[f'X{i}'] = BernoulliRV(.5,Activations.LINEAR).Sample()
        ret['Y'] =y.MostRecentSample 
        return ret

    return SampleDriver(GenerateSample, sampleSize=sample_size, seed = seed)


def GenerateSimpleDataSet_Confounder(sample_size, seed = 0):
    def GenerateSample():
        c = NormalRV(15, 2)
        a = NormalRV((c - 15)/np.sqrt(5) * 4, 2)
        b = NormalRV(8, 3)
        y = NormalRV((c - 15)/np.sqrt(2) * 10 + (b - 8)/np.sqrt(3) * 8 + 20, 1)
        
        ret = {
            'A': a.Sample(),
            'B': b.Sample(),
            'Confounder': c.Sample(),
            'Y': y.Sample()
        }
        
        return ret

    return SampleDriver(GenerateSample, sampleSize=sample_size, seed = seed)

def GenerateSimpleDataSet_Collider(sample_size, seed = 0):
    def GenerateSample():
        a = NormalRV(20, 4)
        b = NormalRV(13, 5)
        y = NormalRV((b - 13)/np.sqrt(5) * 8 + 20, 1)
        c = NormalRV(a * .5 + y * 1.5 + 4, 5)
        ret = {
            'A': a.Sample(),
            'B': b.Sample(),
            'Collider': c.Sample(),
            'Y': y.Sample()
        }
        
        return ret

    return SampleDriver(GenerateSample, sampleSize=sample_size, seed = seed)

def GenerateSimpleDataSet_Mediator(sample_size, seed = 0):
    def GenerateSample():
        a = NormalRV(0, 1)
        b = NormalRV(0, 5)
        m = NormalRV(a * 10 + 6, 1)
        y = NormalRV(m * 3 + b/np.sqrt(5) *2 - 4, .4)
        y.Sample();
        ret = {
            'A': a.Sample(),
            'B': b.Sample(),
            'Mediator': m.Sample(),
            'Y': y.Sample()
        }
        
        return ret

    return SampleDriver(GenerateSample, sampleSize=sample_size, seed = seed)


def GenerateComplexDataset_BrendansBikeRides(sample_size, seed = 0):
    def GenerateSample():
        temp = UniformRV(40,100)
        isRainy = BernoulliRV(.2,Activations.LINEAR)
        distanceMilesPlanned = UniformRV(10,100)
        hoursUntilNextPlans = PoissonRV(3)
        hoursSleptNightBefore = UniformRV(5,11)
        isCovidPandemic = BernoulliRV(.3,Activations.LINEAR)
        isWeekend = BernoulliRV(.7,Activations.LINEAR)
        
        mosquitoIndex = NormalRV((temp-30)/10,2)
        tirePressurePSI = NormalRV(70,10)
        isBadHairDay = BernoulliRV(.02,Activations.LINEAR)
        unfinishedGamesDownloaded = PoissonRV(np.log(20))
        hasNetflixSubscription = BernoulliRV(.33, Activations.LINEAR)
        
        wentOnBikeRide = BernoulliRV(.15 * (temp-40)/60 
                                    + .8 * isRainy * ((temp-70)/30) * ((temp-70)/30)
                                    + -1 * (distanceMilesPlanned - 40)/90
                                    + .6 * (3 - hoursUntilNextPlans)/3
                                    + .25* (hoursSleptNightBefore-8)/6 
                                    + .3 * isCovidPandemic 
                                    + 1.2* isWeekend, Activations.SIGMOID)
        return {
            'X1_MosquitoIndex' : mosquitoIndex.Sample(),
            'X2_UnfinishedGamesDownloaded' : unfinishedGamesDownloaded.Sample(),
            'X3_IsBadHairDay' : isBadHairDay.Sample(),
            'X4_HoursSleptNightBefore' :hoursSleptNightBefore.Sample(),
            'X5_IsCovidPandemic' : isCovidPandemic.Sample(),
            'X6_HoursUntilNextPlans' : hoursUntilNextPlans.Sample(),
            'X7_IsWeekend' : isWeekend.Sample(),
            'X8_NetflixActive' : hasNetflixSubscription.Sample(),
            'X9_TirePressurePSI' : tirePressurePSI.Sample(),
            'X10_TempF' : temp.Sample(),
            'X11_IsRainy' : isRainy.Sample(),
            'X12_DistanceMilesPlanned' : distanceMilesPlanned.Sample(),
            'Y_WentOnBikeRide' : wentOnBikeRide.Sample()
        }

    return SampleDriver(GenerateSample, sampleSize=sample_size, seed = seed)