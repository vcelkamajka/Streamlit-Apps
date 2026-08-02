import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import isprime
import io

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from scipy.optimize import differential_evolution
from skopt import gp_minimize

np.set_printoptions(legacy='1.25') # removes the np.float in output
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# START OF OPTIMISATION CODEBASE

def factor_pair(num):
    div_res_list = []
    factor_list = []
    diff_list = []

    if isprime(num) is True:
        return 1, num
    else:
        for n in range(1, num+1):
            if num % n == 0:
                div_res_list.append(n)

        for pair1, pair2 in enumerate(div_res_list):
            for n in range(1, len(div_res_list)):
                val = pair2 * div_res_list[n]
                if val == num:
                    factor_list.append([div_res_list[n], pair2])

        for n in range(len(factor_list)):
            diff = factor_list[n][0] - factor_list[n][1]
            diff_list.append([factor_list[n][0], factor_list[n][1], diff])

        for n in range(len(factor_list)):
            if diff_list[n][2] <= 0:
                return factor_list[n][0], factor_list[n][1]


class BayesianOptimiser:
    def __init__(self, df, x_features,x=None, y=None,scaler=None,x_scaled=None):
        self.df = df
        self.features = x_features
        self.y = y

        if self.y is None:
            self.y = self.df.iloc[:,-1].values
        else:
            self.y = self.df[y].values

        self.x = self.df[x_features].values

        self.scaler = StandardScaler()
        self.x_scaled = self.scaler.fit_transform(self.x)


    def gpr(self, gp=None):
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(self.x.shape[1])) + WhiteKernel(noise_level=1.0)


        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
        self.gp.fit(self.x_scaled, self.y)


        cv_scores = cross_val_score(self.gp, self.x_scaled, self.y, cv=5, scoring="r2")
        st.write(f"Surrogate model 5-fold CV R²: {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")

        if cv_scores.mean() < 0.7:
            st.write('R² value is below 0.7, model may be poorly fitting.')


    def predict(self, minmax = 'max', optimise_method='Bayesian', de_res=None, bo_res=None, bounds=None):

        self.minmax = minmax

        def predict_yield(params):
            params_scaled = self.scaler.transform([params])
            pred = self.gp.predict(params_scaled)[0]
            return pred


        if self.minmax == 'Maximise':
            def yield_func(params):
                return -predict_yield(params)


        if self.minmax == 'Minimise':
            def yield_func(params):
                return predict_yield(params)

        feat_df = pd.DataFrame()
        features_list = []
        optimise_list = []

        if optimise_method == 'Differential Evolution':

            self.bounds = []
            for col in self.features:
                low, high = self.df[col].min(), self.df[col].max()
                self.bounds.append((low, high))

            self.de_res = differential_evolution(yield_func, self.bounds, seed=42, maxiter=300, tol=1e-8)
            st.write("\n--- Differential Evolution Result ---")

            for n in range(len(self.features)):
                features_list.append(self.features[n])
                optimise_list.append(self.de_res.x[n])

            data = {'Feature': features_list, 'Optimal Values': optimise_list}
            feat_df = pd.DataFrame.from_dict(data)
            st.dataframe(feat_df, hide_index=True, width='stretch')

        if optimise_method == 'Bayesian':

            self.bounds = []
            count = 0
            for col in self.features:
                low, high = self.df[col].min(), self.df[col].max()
                count += 1
                self.bounds.append((low, high))

            self.bo_res = gp_minimize(yield_func, self.bounds, n_calls=40, random_state=1, verbose=False)
            st.write("\n--- Bayesian Optimisation Result ---")

            for n in range(len(self.features)):
                features_list.append(self.features[n])
                optimise_list.append(self.bo_res.x[n])

            data = {'Feature': features_list, 'Optimal Values': optimise_list}
            feat_df = pd.DataFrame.from_dict(data)
            st.dataframe(feat_df, hide_index=True, width='stretch')

    def plots(self, chosen_model='Bayesian'):

        def predict_yield(params):
            params_scaled = self.scaler.transform([params])
            pred = self.gp.predict(params_scaled)[0]
            return pred

        if self.minmax == 'Maximise':
            def yield_func(params):
                return -predict_yield(params)

        if self.minmax == 'Minimise':
            def yield_func(params):
                return predict_yield(params)

        if chosen_model == 'Bayesian':
            best = self.bo_res.x
        if chosen_model == 'Differential Evolution':
            best = self.de_res.x

        labels = self.features

        row, col = factor_pair(len(self.features))

        fig, axes = plt.subplots(row, col, figsize=(11, 8))
        axes = axes.ravel()

        for i in range(len(self.features)):
            ax = axes[i]
            label = labels[i]
            lo = self.bounds[i][0]
            hi = self.bounds[i][1]

            sweep = np.linspace(lo, hi, 60)
            preds = []
            for val in sweep:
                p = best.copy()
                p[i] = val
                preds.append(predict_yield(p))

            ax.plot(sweep, preds, color="#2563eb", linewidth=2)
            ax.axvline(best[i], color="#dc2626", linestyle="--", linewidth=1.5, label=f"optimum = {best[i]:.2f}")
            ax.set_xlabel(label)
            ax.set_ylabel(f"{self.df.columns[-1]}")
            ax.set_title(f"Sensitivity: {label}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        #st.pyplot(fig)
        #fig.clear()

        return fig, buf


# END OF OPTIMISATION CODEBASE


st.title('Optimiser Tool Using Bayesian Optimisation')


uploaded_file = st.file_uploader('Choose a file (in .csv format):', type=['csv'], accept_multiple_files=False)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Filter Data")
    st.write('*Optional, used to look through specific columns.')
    columns = df.columns.tolist()
    selected_column = st.multiselect("Select column to filter by:", columns)
    st.write(df[selected_column])

    st.subheader("Optimise Data")
    x_cols = st.multiselect("Select x features to use:", columns)
    y_cols = st.multiselect("Select y features to use - ensure to not include any x features:", columns)

    res = list(set(x_cols).intersection(set(y_cols)))
    if len(res) > 0:
        st.error('Warning! There are similar x and/or y variables selected in both feature boxes!',icon="🚨")


    col1, col2 = st.columns(2)
    with col1:
        optimisation_method = st.radio('Choose optimisation method:',
                 key='optimisation_method',
                 options=['Bayesian', 'Differential Evolution'])

    with col2:
        max_or_min = st.radio('Maximise or minimise?:',
                                  ('Maximise', 'Minimise'))

    if optimisation_method == 'Bayesian':
        st.warning('Running Bayesian optimisation may take a few minutes depending on the size of your data.', icon='⚠️')


    if st.button('Run Optimisation'):
        try:
            optimisation = BayesianOptimiser(df, x_cols)
            optimisation.gpr()
            optimisation.predict(max_or_min, optimisation_method)
            fig, buf = optimisation.plots(optimisation_method)

            # stash everything needed to survive future reruns
            st.session_state['plot_fig'] = fig
            st.session_state['plot_buf'] = buf
            st.session_state['has_run'] = True

            # Rendered every rerun as long as we've run at least once —
            # reading from session_state, not tied to the button's one-shot True
        except ValueError:
            st.error('Something went wrong. Please ensure that you have chosen x and y features.',icon="🚨")
    if st.session_state.get('has_run'):
        #st.pyplot(st.session_state['plot_fig'])
        st.image(st.session_state['plot_buf'], use_container_width=True)

        on = st.toggle('Note on sensitivity plot')
        if on:
            st.info(
                'Sensitivity plots show each factor as a single variable with the rest of the factors kept at '
                'their best values. The wider the slope, the more resistant the variable is to change and vice versa. '
                'Wide slopes are ideal for minimising the factor whilst maintaining a high response.'
            )

        st.download_button(
            label='Download plot as png',
            data=st.session_state['plot_buf'],
            file_name='sensitivity_plots.png',
            mime='image/png',
        )

        if st.button('Reset plot'):
            st.session_state.clear()
            st.rerun()


