"""
income_simulation.py
 
Simulates a life-cycle model of education, employment, and income for a
cohort of individuals followed from age 18 to retirement at 65.
 
Model summary
-------------
1. Education: each individual draws an education type (short/medium/long)
   at age 18, spends S_e years in school receiving a student grant, then
   enters the labor market as unemployed.
2. Employment: a two-state persistent Markov chain (unemployed <-> employed)
   governs labor market status each year.
3. Human capital: grows while employed, depreciates while unemployed,
   subject to a mean-one lognormal shock each year.
4. Income: student grant while in school; human capital while employed;
   a replacement rate times last job's income while unemployed (or a
   floor for those who have never held a job).
"""
 
import numpy as np
 
 
def simulate_income_distribution(N, p_e, s_e, h_e0, delta_e, delta, sigma_psi,
                                  lam, sigma_sep, y_su, rho, y_floor,
                                  age_min, age_max, seed):
    """
    Simulate the life-cycle income model.
 
    Parameters
    ----------
    N : int
        Number of individuals to simulate.
    p_e : tuple of float
        Probabilities of drawing (short, medium, long) education.
    s_e : tuple of float
        Years of schooling for (short, medium, long) education.
    h_e0 : tuple of float
        Initial human capital when entering the labor market, by education.
    delta_e : tuple of float
        Human capital growth rate while employed, by education.
    delta : float
        Human capital depreciation rate while unemployed.
    sigma_psi : float
        Standard deviation of the underlying normal for the lognormal shock.
    lam : float
        Probability an unemployed person finds a job (job-finding rate).
    sigma_sep : float
        Probability an employed person loses their job (separation rate).
    y_su : float
        Student grant received while in education.
    rho : float
        Replacement rate: unemployment income as a share of last job's income.
    y_floor : float
        Minimum income floor for those who have never held a job.
    age_min, age_max : int
        First and last age simulated (inclusive), e.g. 18 and 65.
    seed : int
        Seed for the random number generator, for reproducibility.
 
    Returns
    -------
    dict with keys:
        'ages'      : array of ages simulated, shape (T,)
        'income'    : income for each individual at each age, shape (N, T)
        'employed'  : employment status (bool) at each age, shape (N, T)
        'education' : education type (0/1/2) for each individual, shape (N,)
    """
    # Set up a reproducible random number generator (per the assignment hint)
    rng = np.random.default_rng(seed)
 
    # Build the array of ages simulated, e.g. 18, 19, ..., 65
    ages = np.arange(age_min, age_max + 1)
    T = len(ages)  # number of periods (years) simulated
 
    # --- Draw each individual's education type at age 18 ---
    # e is 0 = short, 1 = medium, 2 = long education, drawn with probabilities p_e
    e = rng.choice(3, size=N, p=p_e)
 
    # Map each individual's education type to their personal parameters.
    # np.array(s_e)[e] looks up, for every individual, the value of s_e
    # corresponding to their own education draw.
    Se = np.array(s_e)[e]          # years of schooling for this individual
    h0 = np.array(h_e0)[e]         # initial human capital once they start working
    de = np.array(delta_e)[e]      # human capital growth rate while employed
 
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
 
            # Human capital grows if employed, depreciates if unemployed,
            # both cases scaled by this year's random shock
            h_grow = h * (1 + de) * psi
            h_shrink = h * (1 - delta) * psi
            h_next = np.where(employed, h_grow, h_shrink)
 
            # Only update human capital for those already in the labor market
            # (students' human capital stays undefined/NaN until they start working)
            h[labor] = h_next[labor]
 
            # Draw whether unemployed people find a job, and whether
            # employed people lose theirs, using independent random draws
            finds_job = (~employed) & (rng.random(N) < lam)
            loses_job = employed & (rng.random(N) < sigma_sep)
 
            # Apply the transitions to get next period's employment status
            employed = np.where(finds_job, True, employed)
            employed = np.where(loses_job, False, employed)
 
    return {
        'ages': ages,
        'income': hist_income,
        'employed': hist_employed,
        'education': e,
    }
 
 
def gini(x):
    """
    Compute the Gini coefficient of a 1-D array of incomes x.
 
    Uses the mean absolute difference formula:
        G = (sum over all pairs |x_i - x_j|) / (2 * N^2 * mean(x))
    which is equivalent to the standard Lorenz-curve definition but
    simpler to implement and verify directly.
    """
    x = np.sort(np.asarray(x, dtype=float))  # sort incomes ascending
    N = len(x)
    index = np.arange(1, N + 1)              # ranks 1, 2, ..., N
    # Standard rank-based formula for the Gini coefficient of sorted data
    return (2 * np.sum(index * x) - (N + 1) * np.sum(x)) / (N * np.sum(x))