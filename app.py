import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Set wide layout for better dashboard viewing
st.set_page_config(page_title="HUGS Monte Carlo Simulator", layout="wide")

class DynamicHUGSSimulation:
    def __init__(self, df_data, n_sims=1000, years=40, equity_shift_pct=0.01):
        self.n_sims = n_sims
        self.years = years
        self.shift_pct = equity_shift_pct
        
        # Load directly from Pandas DataFrame
        self.asset_names = df_data['Investment Type'].astype(str).tolist()
        self.categories = df_data['Category'].astype(str).str.title().values
        
        # Convert Crores input to absolute values
        amount_col = 'Amount (Crores)' if 'Amount (Crores)' in df_data.columns else 'Amount'
        self.amounts = df_data[amount_col].astype(float).values * 10_000_000
        
        self.mu_assets = df_data['Mean Return'].astype(float).values
        self.vol_assets = df_data['Volatility'].astype(float).values
            
        self.n_assets = len(self.asset_names)
        
        # Map Category Indices
        self.eq_idx = np.where(self.categories == 'Equity')[0]
        self.debt_idx = np.where(self.categories == 'Debt')[0]
        self.cash_idx = np.where(self.categories == 'Cash')[0]
        
        self.initial_corpus = np.sum(self.amounts)
        
        # Starting Expenses (Absolute values)
        self.start_essential = 1_800_000
        self.start_discretionary = 600_000
        self.start_travel = 600_000
        
        # Macroeconomic Inflation Assumptions
        self.mu_inf_ess, self.vol_inf_ess = 0.06, 0.025
        self.mu_inf_disc, self.vol_inf_disc = 0.06, 0.025
        self.mu_inf_trv, self.vol_inf_trv = 0.05, 0.01

        # Track history for plotting
        self.real_corpus_history = np.zeros((self.years, self.n_sims))

    def run(self):
        np.random.seed(42)
        
        # Pre-generate Market Returns & Inflation
        ret_assets = np.zeros((self.years, self.n_sims, self.n_assets))
        for i in range(self.n_assets):
            ret_assets[:, :, i] = np.random.normal(self.mu_assets[i], self.vol_assets[i], (self.years, self.n_sims))
            
        inf_ess = np.random.normal(self.mu_inf_ess, self.vol_inf_ess, (self.years, self.n_sims))
        inf_disc = np.random.normal(self.mu_inf_disc, self.vol_inf_disc, (self.years, self.n_sims))
        inf_trv = np.random.normal(self.mu_inf_trv, self.vol_inf_trv, (self.years, self.n_sims))

        asset_values = np.tile(self.amounts, (self.n_sims, 1))
        
        base_ess = np.full(self.n_sims, float(self.start_essential))
        base_disc = np.full(self.n_sims, float(self.start_discretionary))
        base_trv = np.full(self.n_sims, float(self.start_travel))
        
        survival = np.ones(self.n_sims, dtype=bool)
        years_in_danger = np.zeros(self.n_sims)
        cumulative_inflation = np.ones(self.n_sims)
        
        prev_withdrawal_rate = np.full(self.n_sims, (self.start_essential + self.start_discretionary + self.start_travel) / self.initial_corpus)

        for y in range(self.years):
            corpus = np.sum(asset_values, axis=1)
            cumulative_inflation *= (1 + inf_ess[y])
            actual_spend = np.zeros(self.n_sims)
            
            safe_corpus_eval = np.where(corpus > 0, corpus, 1)
            multiple = safe_corpus_eval / base_ess
            
            severe_danger = (multiple <= 10) & survival
            danger = (multiple > 10) & (multiple <= 15) & survival
            stability = (multiple > 15) & (prev_withdrawal_rate > 0.06) & survival
            normal = (multiple > 15) & (prev_withdrawal_rate >= 0.04) & (prev_withdrawal_rate <= 0.06) & survival
            prosperity = (multiple > 15) & (prev_withdrawal_rate < 0.04) & survival
            
            years_in_danger += (multiple <= 15) & survival
            
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

            remaining_spend = actual_spend.copy()
            
            for cat_idx in [self.cash_idx, self.debt_idx, self.eq_idx]:
                if len(cat_idx) == 0: continue
                cat_values = asset_values[:, cat_idx]
                cat_totals = np.sum(cat_values, axis=1)
                withdraw_from_cat = np.minimum(remaining_spend, cat_totals)
                safe_totals = np.where(cat_totals > 0, cat_totals, 1)
                weights = cat_values / safe_totals[:, None]
                asset_values[:, cat_idx] -= withdraw_from_cat[:, None] * weights
                remaining_spend -= withdraw_from_cat
            
            corpus = np.sum(asset_values, axis=1)
            failed_this_year = corpus <= 1.0
            survival[failed_this_year] = False
            asset_values[failed_this_year] = 0
            
            safe_corpus = np.where(corpus > 0, corpus, 1)
            prev_withdrawal_rate = actual_spend / safe_corpus
            prev_withdrawal_rate[failed_this_year] = 1.0 

            asset_values *= (1 + ret_assets[y])
            
            if len(self.debt_idx) > 0 and len(self.eq_idx) > 0:
                current_totals = np.sum(asset_values, axis=1)
                desired_shift = current_totals * self.shift_pct
                debt_values = asset_values[:, self.debt_idx]
                debt_totals = np.sum(debt_values, axis=1)
                actual_shift = np.minimum(desired_shift, debt_totals)
                
                safe_debt = np.where(debt_totals > 0, debt_totals, 1)
                debt_weights = debt_values / safe_debt[:, None]
                asset_values[:, self.debt_idx] -= actual_shift[:, None] * debt_weights
                eq_add = actual_shift / len(self.eq_idx)
                asset_values[:, self.eq_idx] += eq_add[:, None]

            # Record real corpus history
            self.real_corpus_history[y, :] = np.sum(asset_values, axis=1) / cumulative_inflation

        # Summary Metrics
        results = {
            "survival_rate": np.mean(survival),
            "med_terminal_corpus_real": np.median(self.real_corpus_history[-1, survival]),
            "avg_danger_years": np.mean(years_in_danger),
            "history": self.real_corpus_history
        }
        return results

# --- UI Setup ---

st.title("HUGS Retirement Simulator")
st.markdown("Dynamic Guardrails • Cascading Withdrawals (Cash → Debt → Equity) • Glidepaths")

# Sidebar Configuration
st.sidebar.header("Simulation Settings")
n_sims = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=20000, value=5000, step=500)
years = st.sidebar.slider("Retirement Horizon (Years)", 10, 60, 40)
equity_shift_pct = st.sidebar.slider("Annual Debt-to-Equity Shift (%)", 0.0, 5.0, 1.0, 0.1) / 100.0

# Main Area Inputs
st.subheader("Portfolio Configuration")

# Initialize default data in Crores
default_data = pd.DataFrame({
    "Investment Type": ["Domestic Equity", "International Equity", "Long Term Bonds", "Liquid Funds"],
    "Category": ["Equity", "Equity", "Debt", "Cash"],
    "Amount (Crores)": [3.5, 1.5, 1.5, 0.5],
    "Mean Return": [0.12, 0.14, 0.07, 0.04],
    "Volatility": [0.20, 0.22, 0.02, 0.01]
})

edited_df = st.data_editor(
    default_data, 
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True
)

if st.button("Run Monte Carlo Simulation", type="primary"):
    with st.spinner('Running quantitative paths...'):
        sim = DynamicHUGSSimulation(edited_df, n_sims=n_sims, years=years, equity_shift_pct=equity_shift_pct)
        results = sim.run()
        
        st.divider()
        st.subheader("Simulation Outcomes")
        
        # Convert terminal corpus back to Crores for clean display
        terminal_cr = results['med_terminal_corpus_real'] / 10_000_000
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Portfolio Survival Rate", f"{results['survival_rate'] * 100:.1f}%")
        col2.metric("Median Terminal Corpus (Real)", f"₹{terminal_cr:,.2f} Cr")
        col3.metric("Avg Years in Danger Zone", f"{results['avg_danger_years']:.1f}")
        
        st.subheader("Real Portfolio Trajectories (Inflation-Adjusted)")
        
        # Scale the history arrays down to Crores for the Y-axis
        history_cr = results['history'] / 10_000_000
        x_years = np.arange(1, years + 1)
        
        median_path = np.median(history_cr, axis=1)
        p10_path = np.percentile(history_cr, 10, axis=1)
        p90_path = np.percentile(history_cr, 90, axis=1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x_years, y=p90_path, mode='lines', 
                                 line=dict(color='rgba(46, 204, 113, 0.5)', width=1, dash='dash'),
                                 name='90th Percentile (Prosperity)'))
        fig.add_trace(go.Scatter(x=x_years, y=median_path, mode='lines', 
                                 line=dict(color='rgba(52, 152, 219, 1)', width=3),
                                 name='Median Path'))
        fig.add_trace(go.Scatter(x=x_years, y=p10_path, mode='lines', 
                                 line=dict(color='rgba(231, 76, 60, 0.5)', width=1, dash='dash'),
                                 name='10th Percentile (Stress Test)'))
        
        fig.update_layout(
            xaxis_title="Years in Retirement",
            yaxis_title="Real Portfolio Value (Crores ₹)",
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Note: Real Portfolio Value is discounted for macroeconomic inflation. Paths hitting zero represent portfolio depletion.")