 
from matplotlib.pylab import seed
import numpy as np
 
 
def simulate_income_distribution(N, p_e, s_e, h_e0, delta_e, delta, sigma_psi,
                                  lam, sigma_sep, y_su, rho, y_floor,
                                  age_min, age_max, seed):

    #random number generator
    rng = np.random.default_rng(seed)
 
    #ages array
    ages = np.arange(age_min, age_max + 1)
    T = len(ages)  # number of periods (years) simulated
 
    #draw education types for individuals with prob p_e
    # e is 0 = short, 1 = medium, 2 = long education, drawn with probabilities p_e
    e = rng.choice(3, size=N, p=p_e)
 
    #np.array(s_e)[e] looks up the value of s_e (connects individuals to their characteristics/variables)
    Se = np.array(s_e)[e]          # years of schooling
    h0 = np.array(h_e0)[e]         # initial human capital entering labor market
    de = np.array(delta_e)[e]      # employed human capital growth rate
 
    #variables
    h = np.full(N, np.nan)             #human capital; NaN while still in school
    employed = np.zeros(N, dtype=bool) #current employment status
    ever_employed = np.zeros(N, dtype=bool)  #held a job, ever?
    last_wage = np.zeros(N)            #most recent income
 
    #simulate and store history in arrays
    hist_income = np.empty((N, T))
    hist_h = np.full((N, T), np.nan)
    hist_employed = np.zeros((N, T), dtype=bool)
 
    #Simulation loop ##should we vectorize?
    for t, age in enumerate(ages):
 
        #in school means age is less than the years of schooling plus the minimum age
        in_school = (age - age_min) < Se
 
        #the age of individuals entering the labor market is = to Se + age_min
        entering = (age - age_min) == Se
 
        #Individuals entering the labor market start with their education human capital and begin as unemployed
        h[entering] = h0[entering]
        employed[entering] = False
 
        #Indviduals are either in school or labor market
        labor = ~in_school
 
        #this years income
        income_t = np.empty(N)
 
        # Those in school receive the student grant
        income_t[in_school] = y_su
 
        #split labor market by employment status
        emp_mask = labor & employed
        unemp_mask = labor & ~employed
 
        #human capital = wages for employed
        income_t[emp_mask] = h[emp_mask]
 
        #Unemployedment benefits are a fraction rho of last wage
        had_job_before = unemp_mask & ever_employed
        income_t[had_job_before] = rho * last_wage[had_job_before]
 
        #Unemployed individuals who have never held a job get the income floor
        never_had_job = unemp_mask & ~ever_employed
        income_t[never_had_job] = y_floor
 
        #Create an array of results ot be returned
        hist_income[:, t] = income_t
        hist_h[:, t] = h
        hist_employed[:, t] = emp_mask
 
        #Base benefits on last known income
        last_wage[emp_mask] = h[emp_mask]
        ever_employed[emp_mask] = True
 
        #human capital and employment status for next period
        if t < T - 1:
            #shock this year
            psi = rng.lognormal(-0.5 * sigma_psi**2, sigma_psi, size=N)
 
            #Modelling human capital employed = growth, unemployed = depreciates, incl shock in model
            h_grow = h * (1 + de) * psi
            h_shrink = h * (1 - delta) * psi
            h_next = np.where(employed, h_grow, h_shrink)
 
            #keep student capital NaN until they enter the labor market
            h[labor] = h_next[labor]
 
            #random draws for employment
            finds_job = (~employed) & (rng.random(N) < lam)
            loses_job = employed & (rng.random(N) < sigma_sep)
 
            #next period employment status
            employed = np.where(finds_job, True, employed)
            employed = np.where(loses_job, False, employed)

    #Check that the lognormal shock has mean 1
    rng_check = np.random.default_rng(seed)
    psi_check = rng_check.lognormal(-0.5 * sigma_psi**2, sigma_psi, size=1_000_000)
    print(f'Mean of psi: {psi_check.mean():.4f}')  # should be very close to 1.00
 
    return {
        'ages': ages,
        'income': hist_income,
        'employed': hist_employed,
        'education': e,
    }
