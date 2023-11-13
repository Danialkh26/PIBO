import numpy as np
from kawin.Thermodynamics import BinaryThermodynamics
from kawin.KWNEuler import PrecipitateModel
from kawin.ElasticFactors import StrainEnergy

def precipitator(x): #temp in K, time in s

    def ar3(r):
        a = 18.4012307
        b = 51.3197329
        R = r*1e9
        return a*R/(b-R)

    comp=x[0,0]
    temp=x[0,1]
    time=x[0,2]
	
    
    #inputs
    xInit = comp
    T = temp
    
    t0 = 1
    tf = time
    steps = 1e4
    
    #physical properties
    gamma = 0.053
    Dni = lambda x, T: 1.8e-8 * np.exp(-155000/(8.314*T))
    Vaalpha = 0.02681144066*1e-27
    nalpha = 2
    Vabeta = 0.184614835*1e-27
    nbeta = 14
    
    eigenstrain = [-0.00417, -0.00417, -0.0257]
    
    B2e = np.asarray([175,45,35]) * 1e9
    
    rotate = [[-4/np.sqrt(42), 5/np.sqrt(42), -1/np.sqrt(42)],
              [-2/np.sqrt(14), -1/np.sqrt(14), 3/np.sqrt(14)],
              [1/np.sqrt(3),1/np.sqrt(3), 1/np.sqrt(3)]]
    
    #initializing thermodynamics model
    therm = BinaryThermodynamics('NiTi_SMA.tdb', ['TI', 'NI'], ['BCC_B2', 'TI3NI4'])
    therm.setGuessComposition(0.56)
    
    #initializing precipitate model
    model = PrecipitateModel(t0, tf, steps, linearTimeSpacing = False)
    model.setInitialComposition(xInit)
    model.setTemperature(T)
    
    model.setInterfacialEnergy(gamma)
    model.setDiffusivity(Dni)
    model.setVaAlpha(Vaalpha, nalpha)
    model.setVaBeta(Vabeta, nbeta)
    
    #initializing strain energy model
    se = StrainEnergy()
    se.setEllipsoidal()
    se.setEigenstrain(eigenstrain)
    se.setElasticConstants(B2e[0], B2e[1], B2e[2])
    se.setRotationMatrix(rotate)
    
    model.setThermodynamics(therm, addDiffusivity=False)
    model.setStrainEnergy(se, calculateAspectRatio=False)
    model.setAspectRatioPlate(ar3)
    model.setNucleationDensity()
    
    #solving model
    model.solve(verbose=False)
    
    mComp = model.xComp[-1] #final matrix composition
    
    #fitting parameters for transformation temperature equation (these are for Ms only)
    A = 4511.2373
    B = -83.32325
    C = -0.04753
    D = 204.86781
    
    Ms_temp = A + (B*mComp)*100 + C*D**((mComp*100)-50) #martensitic start temperature in K
    bFrac = model.betaFrac[-1][-1]
    nDens = model.precipitateDensity[0][-1] #number of precipitates/m^3

    if nDens > 0:
        mipd = np.cbrt(3/(4*np.pi*nDens)) #mean inter-particle distance in meters
    else:
        mipd = 1000 # artificially high number to indicate no precipitates form, may need to be changed
    
    return np.array(Ms_temp), np.array(mipd), np.array(bFrac)