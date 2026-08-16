 
import numpy as np
 
 
def simulate_income_distribution(N, p_e, s_e, h_e0, delta_e, delta, sigma_psi,
                                  lam, sigma_sep, y_su, rho, y_floor,
                                  age_min, age_max, seed):

    # random number generator
    rng = np.random.default_rng(seed)
 
    #ages array
    ages = np.arange(age_min, age_max + 1)
    T = len(ages)  # number of periods (years) simulated
 
    #draw education types for individuals with prob p_e
    # e is 0 = short, 1 = medium, 2 = long education, drawn with probabilities p_e
    e = rng.choice(3, size=N, p=p_e)
 
    # np.array(s_e)[e] looks up the value of s_e (connects individuals to their characteristics/variables)
    Se = np.array(s_e)[e]          # years of schooling
    h0 = np.array(h_e0)[e]         # initial human capital entering labor market
    de = np.array(delta_e)[e]      # employed human capital growth rate
 
    # --- State variables, updated as we step through time ---
    h = np.full(N, np.nan)             # human capital; NaN while still in school
    employed = np.zeros(N, dtype=bool) # current employment status
    ever_employed = np.zeros(N, dtype=bool)  # has this person ever held a job?
    last_wage = np.zeros(N)            # income earned in their most recent job
 
    # --- Arrays to store the full simulated history for later analysis ---
    hist_income = np.empty((N, T))
    hist_h = np.full((N, T), np.nan)
    hist_employed = np.zeros((N, T), dtype=bool)
 
    # --- Main simulation loop over time ---
    for t, age in enumerate(ages):
 
        # True for individuals still in school this year: their years since
        # age_min have not yet reached their required schooling length Se
        in_school = (age - age_min) < Se
 
        # True for individuals entering the labor market for the first time
        # this year, i.e. this is exactly the year school ends
        entering = (age - age_min) == Se
 
        # Individuals entering the labor market start with their
        # education-specific human capital and begin as unemployed
        h[entering] = h0[entering]
        employed[entering] = False
 
        # Everyone not in school is in the labor market this period
        labor = ~in_school
 
        # --- Compute this period's income for everyone ---
        income_t = np.empty(N)
 
        # Those in school receive the student grant
        income_t[in_school] = y_su
 
        # Among labor market participants, split by employment status
        emp_mask = labor & employed
        unemp_mask = labor & ~employed
 
        # Employed individuals earn income equal to their human capital
        income_t[emp_mask] = h[emp_mask]
 
        # Unemployed individuals who have held a job before earn a fraction
        # (the replacement rate) of what they earned in their last job
        had_job_before = unemp_mask & ever_employed
        income_t[had_job_before] = rho * last_wage[had_job_before]
 
        # Unemployed individuals who have never held a job get the income floor
        never_had_job = unemp_mask & ~ever_employed
        income_t[never_had_job] = y_floor
 
        # Store this period's results before moving on
        hist_income[:, t] = income_t
        hist_h[:, t] = h
        hist_employed[:, t] = emp_mask
 
        # Update "last job income" and "ever employed" for anyone employed now,
        # so that if they lose their job later, benefits are based on this wage
        last_wage[emp_mask] = h[emp_mask]
        ever_employed[emp_mask] = True
 
        # --- Evolve human capital and employment status for next period ---
        if t < T - 1:
            # Draw this year's shock: mean-one lognormal, i.e.
            # log(psi) ~ Normal(-0.5*sigma_psi^2, sigma_psi^2), so E[psi] = 1
            psi = rng.lognormal(-0.5 * sigma_psi**2, sigma_psi, size=N)
 
            #Modelling human capital employed = growth, unemployed = depreciates, incl shock in model
            h_grow = h * (1 + de) * psi
            h_shrink = h * (1 - delta) * psi
            h_next = np.where(employed, h_grow, h_shrink)
 
            #keep student capital NaN until they enter the labor market
            h[labor] = h_next[labor]
 
            # independent random draws for employment
            finds_job = (~employed) & (rng.random(N) < lam)
            loses_job = employed & (rng.random(N) < sigma_sep)
 
            #next period employment status
            employed = np.where(finds_job, True, employed)
            employed = np.where(loses_job, False, employed)
 
    return {
        'ages': ages,
        'income': hist_income,
        'employed': hist_employed,
        'education': e,
    }
#Check that the lognormal shock has mean 1
rng_check = np.random.default_rng(seed)
psi_check = rng_check.lognormal(-0.5 * sigma_psi**2, sigma_psi, size=1_000_000)

print(f'Mean of psi: {psi_check.mean():.4f}')  # should be very close to 1.00