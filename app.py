import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import base64
import io

st.set_page_config(page_title="HUGS Monte Carlo Simulator", layout="wide")

class DynamicHUGSSimulation:
    def __init__(self, df_data, n_sims=1000, years=40, equity_shift_pct=0.01,
                 start_essential=1_800_000, start_discretionary=600_000, start_travel=600_000,
                 mu_inf_ess=0.06, vol_inf_ess=0.025, 
                 mu_inf_disc=0.06, vol_inf_disc=0.025, 
                 mu_inf_trv=0.05, vol_inf_trv=0.01):
        
        self.n_sims = int(n_sims)
        self.years = int(years)
        self.shift_pct = float(equity_shift_pct)
        
        self.asset_names = df_data['Investment Type'].astype(str).tolist()
        self.categories = df_data['Category'].astype(str).str.title().values
        
        amount_col = 'Amount (Crores)' if 'Amount (Crores)' in df_data.columns else 'Amount'
        self.amounts = df_data[amount_col].astype(float).values * 10_000_000
        
        self.mu_assets = df_data['Mean Return'].astype(float).values
        self.vol_assets = df_data['Volatility'].astype(float).values
            
        self.n_assets = len(self.asset_names)
        
        self.eq_idx = np.where(self.categories == 'Equity')[0]
        self.debt_idx = np.where(self.categories == 'Debt')[0]
        self.cash_idx = np.where(self.categories == 'Cash')[0]
        
        self.initial_corpus = np.sum(self.amounts)
        
        self.start_essential = float(start_essential)
        self.start_discretionary = float(start_discretionary)
        self.start_travel = float(start_travel)
        
        self.mu_inf_ess, self.vol_inf_ess = float(mu_inf_ess), float(vol_inf_ess)
        self.mu_inf_disc, self.vol_inf_disc = float(mu_inf_disc), float(vol_inf_disc)
        self.mu_inf_trv, self.vol_inf_trv = float(mu_inf_trv), float(vol_inf_trv)

        self.real_corpus_history = np.zeros((self.years, self.n_sims))
        self.eq_history = np.zeros((self.years, self.n_sims))
        self.debt_history = np.zeros((self.years, self.n_sims))
        self.cash_history = np.zeros((self.years, self.n_sims))

    def run(self):
        np.random.seed(42)
        
        # Pre-generate Market Returns & Inflation arrays
        ret_assets = np.zeros((self.years, self.n_sims, self.n_assets))
        for i in range(self.n_assets):
            ret_assets[:, :, i] = np.random.normal(self.mu_assets[i], self.vol_assets[i], (self.years, self.n_sims))
            
        inf_ess = np.random.normal(self.mu_inf_ess, self.vol_inf_ess, (self.years, self.n_sims))
        inf_disc = np.random.normal(self.mu_inf_disc, self.vol_inf_disc, (self.years, self.n_sims))
        inf_trv = np.random.normal(self.mu_inf_trv, self.vol_inf_trv, (self.years, self.n_sims))

        asset_values = np.tile(self.amounts, (self.n_sims, 1))
        
        base_ess = np.full(self.n_sims, self.start_essential)
        base_disc = np.full(self.n_sims, self.start_discretionary)
        base_trv = np.full(self.n_sims, self.start_travel)
        
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
            
            # Guardrails Logic
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

            # Cascading Withdrawals: Cash -> Debt -> Equity
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

            # Market Returns
            asset_values *= (1 + ret_assets[y])
            
            # Debt-to-Equity Glidepath Shift
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

            # Record inflation-adjusted corpus for this year
            self.real_corpus_history[y, :] = np.sum(asset_values, axis=1) / cumulative_inflation
            self.eq_history[y, :] = np.sum(asset_values[:, self.eq_idx], axis=1) / cumulative_inflation
            self.debt_history[y, :] = np.sum(asset_values[:, self.debt_idx], axis=1) / cumulative_inflation
            self.cash_history[y, :] = np.sum(asset_values[:, self.cash_idx], axis=1) / cumulative_inflation

        results = {
            "survival_rate": np.mean(survival),
            "med_terminal_corpus_real": np.median(self.real_corpus_history[-1, survival]),
            "avg_danger_years": np.mean(years_in_danger),
            "history": self.real_corpus_history,
            "eq_history": self.eq_history,
            "debt_history": self.debt_history,
            "cash_history": self.cash_history
        }
        return results

def update_query_params():
    """Saves session state into the URL query parameters so it persists on page refresh."""
    for key in default_scalars.keys():
        st.query_params[key] = str(st.session_state[key])

def update_portfolio_param(df):
    """Encodes the portfolio dataframe into a base64 string for URL persistence."""
    try:
        json_str = df.to_json(orient="records")
        encoded = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        st.query_params["portfolio"] = encoded
    except Exception as e:
        pass

# Default scalar parameters
default_scalars = {
    "n_sims": 5000,
    "years": 40,
    "equity_shift_pct": 1.0,
    "ess_lakhs": 18.0,
    "disc_lakhs": 6.0,
    "trv_lakhs": 6.0,
    "mu_inf_ess": 6.0,
    "vol_inf_ess": 2.5,
    "mu_inf_disc": 6.0,
    "vol_inf_disc": 2.5,
    "mu_inf_trv": 5.0,
    "vol_inf_trv": 1.0
}

# Initialize session state from URL parameters if available
for key, val in default_scalars.items():
    if key not in st.session_state:
        if key in st.query_params:
            try:
                st.session_state[key] = type(val)(st.query_params[key])
            except ValueError:
                st.session_state[key] = val
        else:
            st.session_state[key] = val

default_df = pd.DataFrame({
    "Investment Type": ["Domestic Equity", "International Equity", "Long Term Bonds", "Liquid Funds"],
    "Category": ["Equity", "Equity", "Debt", "Cash"],
    "Amount (Crores)": [3.82, 3.47, 6.98, 3.0],
    "Mean Return": [0.12, 0.14, 0.07, 0.04],
    "Volatility": [0.20, 0.22, 0.02, 0.01]
})

# Load dataframe from URL if present
if "portfolio" in st.query_params:
    try:
        decoded = base64.b64decode(st.query_params["portfolio"]).decode('utf-8')
        loaded_df = pd.read_json(io.StringIO(decoded), orient="records")
        if not loaded_df.empty:
            st.session_state["portfolio_df"] = loaded_df
    except Exception:
        st.session_state["portfolio_df"] = default_df
else:
    if "portfolio_df" not in st.session_state:
        st.session_state["portfolio_df"] = default_df

st.title("HUGS Retirement Simulator")
st.markdown("Dynamic Guardrails • Cascading Withdrawals (Cash → Debt → Equity) • Glidepaths • **Autosaved**")

st.sidebar.header("Simulation Settings")
n_sims = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=20000, key="n_sims", step=500, on_change=update_query_params)
years = st.sidebar.slider("Retirement Horizon (Years)", 10, 60, key="years", on_change=update_query_params)
equity_shift_pct = st.sidebar.slider("Annual Debt-to-Equity Shift (%)", 0.0, 5.0, key="equity_shift_pct", step=0.1, on_change=update_query_params) / 100.0

with st.sidebar.expander("Initial Expenses (Lakhs ₹)", expanded=True):
    ess_lakhs = st.number_input("Essential Lifestyle", key="ess_lakhs", step=0.5, on_change=update_query_params)
    disc_lakhs = st.number_input("Discretionary Lifestyle", key="disc_lakhs", step=0.5, on_change=update_query_params)
    trv_lakhs = st.number_input("Travel", key="trv_lakhs", step=0.5, on_change=update_query_params)

with st.sidebar.expander("Inflation Assumptions (%)", expanded=False):
    st.markdown("**Essential**")
    mu_inf_ess = st.number_input("Mean (Ess)", key="mu_inf_ess", step=0.5, on_change=update_query_params) / 100.0
    vol_inf_ess = st.number_input("Volatility (Ess)", key="vol_inf_ess", step=0.1, on_change=update_query_params) / 100.0
    
    st.markdown("**Discretionary**")
    mu_inf_disc = st.number_input("Mean (Disc)", key="mu_inf_disc", step=0.5, on_change=update_query_params) / 100.0
    vol_inf_disc = st.number_input("Volatility (Disc)", key="vol_inf_disc", step=0.1, on_change=update_query_params) / 100.0
    
    st.markdown("**Travel**")
    mu_inf_trv = st.number_input("Mean (Trv)", key="mu_inf_trv", step=0.5, on_change=update_query_params) / 100.0
    vol_inf_trv = st.number_input("Volatility (Trv)", key="vol_inf_trv", step=0.1, on_change=update_query_params) / 100.0

st.subheader("Portfolio Configuration")

edited_df = st.data_editor(
    st.session_state["portfolio_df"], 
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True
)

# Persist the dataframe silently on change to query params
update_portfolio_param(edited_df)

if st.button("Run Monte Carlo Simulation", type="primary"):
    with st.spinner('Running quantitative paths...'):
        
        sim = DynamicHUGSSimulation(
            df_data=edited_df, 
            n_sims=n_sims, 
            years=years, 
            equity_shift_pct=equity_shift_pct,
            start_essential=ess_lakhs * 100_000,
            start_discretionary=disc_lakhs * 100_000,
            start_travel=trv_lakhs * 100_000,
            mu_inf_ess=mu_inf_ess, vol_inf_ess=vol_inf_ess,
            mu_inf_disc=mu_inf_disc, vol_inf_disc=vol_inf_disc,
            mu_inf_trv=mu_inf_trv, vol_inf_trv=vol_inf_trv
        )
        
        results = sim.run()
        
        st.divider()
        st.subheader("Simulation Outcomes")
        
        terminal_cr = results['med_terminal_corpus_real'] / 10_000_000
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Portfolio Survival Rate", f"{results['survival_rate'] * 100:.1f}%")
        col2.metric("Median Terminal Corpus (Real)", f"₹{terminal_cr:,.2f} Cr")
        col3.metric("Avg Years in Danger Zone", f"{results['avg_danger_years']:.1f}")
        
        st.subheader("Real Portfolio Trajectories (Inflation-Adjusted)")
        
        history_cr = results['history'] / 10_000_000
        x_years = np.arange(1, years + 1)
        
        # Calculate percentiles across ALL simulated paths
        median_path = np.median(history_cr, axis=1)
        p10_path = np.percentile(history_cr, 10, axis=1)
        p75_path = np.percentile(history_cr, 75, axis=1)
        p90_path = np.percentile(history_cr, 90, axis=1)
        
        fig = go.Figure()

        # Identify all paths that finish in the bottom 10% 
        terminal_values = history_cr[-1, :]
        p10_terminal = np.percentile(terminal_values, 10)
        bottom_10_idx = np.where(terminal_values <= p10_terminal)[0]

        for idx in bottom_10_idx:
            fig.add_trace(go.Scattergl(
                x=x_years, 
                y=history_cr[:, idx], 
                mode='lines',
                line=dict(color='rgba(231, 76, 60, 0.05)', width=1), 
                showlegend=False,
                hoverinfo='skip'
            ))
            
        fig.add_trace(go.Scattergl(x=x_years, y=p90_path, mode='lines', 
                                 line=dict(color='rgba(46, 204, 113, 1)', width=2, dash='dash'),
                                 name='90th Percentile (Prosperity)'))
        
        fig.add_trace(go.Scattergl(x=x_years, y=p75_path, mode='lines', 
                                 line=dict(color='rgba(241, 196, 15, 1)', width=2, dash='dashdot'),
                                 name='75th Percentile (Comfort)'))

        fig.add_trace(go.Scattergl(x=x_years, y=median_path, mode='lines', 
                                 line=dict(color='rgba(52, 152, 219, 1)', width=4),
                                 name='Median Path'))
        
        fig.add_trace(go.Scattergl(x=x_years, y=p10_path, mode='lines', 
                                 line=dict(color='rgba(231, 76, 60, 1)', width=3, dash='solid'),
                                 name='10th Percentile (Stress Boundary)'))
        
        fig.update_layout(
            xaxis_title="Years in Retirement",
            yaxis_title="Real Portfolio Value (Crores ₹)",
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- NEW PORTFOLIO SPLIT VISUALIZATION ---
        st.divider()
        st.subheader("Portfolio Split Over Time (Cash vs Debt vs Equity)")
        st.markdown("Comparing the asset composition shift for a representative **Median path** vs a **10th Percentile (Stress) path**.")
        
        # Find representative paths
        terminal_values = history_cr[-1, :]
        idx_median = np.argmin(np.abs(terminal_values - np.percentile(terminal_values, 50)))
        idx_p10 = np.argmin(np.abs(terminal_values - np.percentile(terminal_values, 10)))
        
        eq_cr = results['eq_history'] / 10_000_000
        debt_cr = results['debt_history'] / 10_000_000
        cash_cr = results['cash_history'] / 10_000_000
        
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            fig_med = go.Figure()
            fig_med.add_trace(go.Scatter(x=x_years, y=cash_cr[:, idx_median], mode='lines', stackgroup='one', name='Cash', line=dict(color='rgba(241, 196, 15, 0.8)')))
            fig_med.add_trace(go.Scatter(x=x_years, y=debt_cr[:, idx_median], mode='lines', stackgroup='one', name='Debt', line=dict(color='rgba(149, 165, 166, 0.8)')))
            fig_med.add_trace(go.Scatter(x=x_years, y=eq_cr[:, idx_median], mode='lines', stackgroup='one', name='Equity', line=dict(color='rgba(52, 152, 219, 0.8)')))
            fig_med.update_layout(title="Median Path Composition", xaxis_title="Years in Retirement", yaxis_title="Real Value (Crores ₹)", template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_med, use_container_width=True)
            
        with col_plot2:
            fig_p10 = go.Figure()
            fig_p10.add_trace(go.Scatter(x=x_years, y=cash_cr[:, idx_p10], mode='lines', stackgroup='one', name='Cash', line=dict(color='rgba(241, 196, 15, 0.8)')))
            fig_p10.add_trace(go.Scatter(x=x_years, y=debt_cr[:, idx_p10], mode='lines', stackgroup='one', name='Debt', line=dict(color='rgba(149, 165, 166, 0.8)')))
            fig_p10.add_trace(go.Scatter(x=x_years, y=eq_cr[:, idx_p10], mode='lines', stackgroup='one', name='Equity', line=dict(color='rgba(52, 152, 219, 0.8)')))
            fig_p10.update_layout(title="10th Percentile Path Composition", xaxis_title="Years in Retirement", yaxis_title="Real Value (Crores ₹)", template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_p10, use_container_width=True)

        st.caption("Settings are currently autosaving to your URL. Bookmark or copy the link to return to this exact setup.")