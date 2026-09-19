import numpy as np

class HUGSSimulation:
    def __init__(self, n_sims=10000, years=40) -> None:
        self.n_sims = n_sims
        self.years = years
        
        # Initial Portfolio and Expenses
        self.initial_corpus = 145_000_000  # 17 Crores (1CR is 10^7)
        self.start_essential = 2_000_000  # 20 Lakhs
        self.start_discretionary = 1_000_000 # 10 Lakhs
        self.start_travel = 800_000      # 8 Lakhs
        
        # Market & Inflation Assumptions (Mean, Volatility)
        self.mu_eq, self.vol_eq = 0.12, 0.20
        self.mu_debt, self.vol_debt = 0.07, 0.02
        self.mu_inf_ess, self.vol_inf_ess = 0.06, 0.025
        self.mu_inf_disc, self.vol_inf_disc = 0.06, 0.025
        self.mu_inf_trv, self.vol_inf_trv = 0.05, 0.01
        
        # Rebalancing Bands for 50/50 Target
        self.target_eq = 0.80
        self.lower_eq_band = 0.70
        self.upper_eq_band = 0.90

    def run(self) -> np.ndarray:
        np.random.seed(42)
        ret_eq = np.random.normal(self.mu_eq, self.vol_eq, (self.years, self.n_sims))
        ret_debt = np.random.normal(self.mu_debt, self.vol_debt, (self.years, self.n_sims))
        inf_ess = np.random.normal(self.mu_inf_ess, self.vol_inf_ess, (self.years, self.n_sims))
        inf_disc = np.random.normal(self.mu_inf_disc, self.vol_inf_disc, (self.years, self.n_sims))
        inf_trv = np.random.normal(self.mu_inf_trv, self.vol_inf_trv, (self.years, self.n_sims))

        corpus = np.full(self.n_sims, float(self.initial_corpus))
        eq_weights = np.full(self.n_sims, self.target_eq)
        
        base_ess = np.full(self.n_sims, float(self.start_essential))
        base_disc = np.full(self.n_sims, float(self.start_discretionary))
        base_trv = np.full(self.n_sims, float(self.start_travel))
        
        survival = np.ones(self.n_sims, dtype=bool)
        years_in_danger = np.zeros(self.n_sims)
        total_travel_15y = np.zeros(self.n_sims)
        rebalance_counts = np.zeros(self.n_sims)
        
        # Real Inflation Tracker
        cumulative_inflation = np.ones(self.n_sims)

        prev_withdrawal_rate = np.full(self.n_sims, (self.start_essential + self.start_discretionary + self.start_travel) / self.initial_corpus)

        for y in range(self.years):
            # Compound actual macroeconomic inflation for this year
            cumulative_inflation *= (1 + inf_ess[y])
            
            actual_spend = np.zeros(self.n_sims)
            multiple = corpus / base_ess
            
            severe_danger = (multiple <= 10)
            danger = (multiple > 10) & (multiple <= 15)
            stability = (multiple > 15) & (prev_withdrawal_rate > 0.06)
            normal = (multiple > 15) & (prev_withdrawal_rate >= 0.04) & (prev_withdrawal_rate <= 0.06)
            prosperity = (multiple > 15) & (prev_withdrawal_rate < 0.04)
            
            years_in_danger += (multiple <= 15)
            
            base_ess[severe_danger] *= 0.90
            actual_spend[severe_danger] = base_ess[severe_danger]
            
            actual_spend[danger] = base_ess[danger]
            
            base_ess[stability] *= (1 + inf_ess[y, stability])
            base_disc[stability] *= (1 + inf_disc[y, stability]) 
            actual_spend[stability] = base_ess[stability] + (0.5 * base_disc[stability])
            
            mask_norm_prosp = normal | prosperity
            base_ess[mask_norm_prosp] *= (1 + inf_ess[y, mask_norm_prosp])
            base_disc[mask_norm_prosp] *= (1 + inf_disc[y, mask_norm_prosp])
            base_trv[mask_norm_prosp] *= (1 + inf_trv[y, mask_norm_prosp])
            
            spend_np = base_ess[mask_norm_prosp] + base_disc[mask_norm_prosp] + base_trv[mask_norm_prosp]
            
            bonus = np.zeros(self.n_sims)
            bonus[prosperity] = 0.25 * base_disc[prosperity]
            
            actual_spend[mask_norm_prosp] = spend_np + bonus[mask_norm_prosp]
            
            if y < 15:
                travel_spend = np.zeros(self.n_sims)
                travel_spend[mask_norm_prosp] = base_trv[mask_norm_prosp]
                total_travel_15y += travel_spend

            corpus -= actual_spend
            
            failed_this_year = corpus <= 0
            survival[failed_this_year] = False
            corpus[failed_this_year] = 0
            
            safe_corpus = np.where(corpus > 0, corpus, 1)
            prev_withdrawal_rate = actual_spend / safe_corpus
            prev_withdrawal_rate[failed_this_year] = 1.0 

            port_ret = (eq_weights * ret_eq[y]) + ((1 - eq_weights) * ret_debt[y])
            corpus *= (1 + port_ret)
            
            eq_value = corpus * eq_weights * (1 + ret_eq[y])
            total_value = corpus 
            eq_weights = np.where(total_value > 0, eq_value / total_value, self.target_eq)
            
            needs_rebalance = (eq_weights < self.lower_eq_band) | (eq_weights > self.upper_eq_band)
            rebalance_counts += needs_rebalance
            eq_weights[needs_rebalance] = self.target_eq

        survival_rate = np.mean(survival)
        
        # Calculate Real Terminal Corpus
        real_corpus = corpus / cumulative_inflation
        med_terminal_corpus_real = np.median(real_corpus[survival])
        percentiles = np.percentile(real_corpus[survival], [5,10,25, 50, 75, 90, 95])
        
        avg_danger_years = np.mean(years_in_danger)
        avg_travel_15y = np.mean(total_travel_15y)
        avg_rebalances = np.mean(rebalance_counts)
        
        print("=== Monte Carlo Simulation Results ===")
        print(f"Total Simulations Run: {self.n_sims}")
        print(f"Portfolio Survival Rate: {survival_rate * 100:.2f}%")
        print(f"Percentiles Corpus (Real): 5th: ₹{percentiles[0]/10**7:,.2f}, 10th: ₹{percentiles[1]/10**7:,.2f}, 25th: ₹{percentiles[2]/10**7:,.2f}, 50th: ₹{percentiles[3]/10**7:,.2f}, 75th: ₹{percentiles[4]/10**7:,.2f}, 90th: ₹{percentiles[5]/10**7:,.2f}, 95th: ₹{percentiles[6]/10**7:,.2f}")
        print(f"Average Years in Danger Zone (<15x): {avg_danger_years:.1f} years")
        print(f"Average Travel Spend (First 15 Yrs): ₹{avg_travel_15y:,.0f}")
        print(f"Average Portfolio Rebalances: {avg_rebalances:.1f} times")

        return corpus

if __name__ == "__main__":
    sim = HUGSSimulation(n_sims=100000)
    final_corpus = sim.run()