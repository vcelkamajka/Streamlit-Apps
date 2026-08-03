import itertools

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import isprime
import io

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score

from scipy.optimize import differential_evolution
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

st.html("""
    <style>
        section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
            max-width: 1000px;
        }
    </style>
""")

np.set_printoptions(legacy='1.25')  # removes the np.float in output
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

danger_icon = ':material/error:'
success_icon = ':material/check:'
warning_icon = ':material/warning:'
info_icon = ':material/info:'
star_icon = ':material/star_border:'

main_plot_colour = '#942553' # the colour of the line
secondary_plot_colour = '#255394' # the colour of the secondary plot (bar chart)
optimal_line_colour = '#ED5557' # the colour of the optimal line

# START OF OPTIMISATION CODEBASE

def factor_pair(num):
    div_res_list = []
    factor_list = []
    diff_list = []

    if isprime(num) is True:
        return 1, num
    else:
        for n in range(1, num + 1):
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


def grid_shape(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def build_mixed_space(df, x_cols):
    """Inspects each column's dtype and returns a skopt space plus metadata.
    Used exclusively by the 'Bayesian Space' path (MixedBayesianOptimiser),
    where NO manual encoding has been applied to df."""
    space = []
    col_kinds = {}  # col -> "categorical" | "real" | "integer"

    for col in x_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            lo, hi = float(df[col].min()), float(df[col].max())
            if pd.api.types.is_integer_dtype(df[col]):
                space.append(Integer(int(lo), int(hi), name=col))
                col_kinds[col] = "integer"
            else:
                space.append(Real(lo, hi, name=col))
                col_kinds[col] = "real"
        else:
            choices = sorted(df[col].dropna().unique().tolist())
            space.append(Categorical(choices, name=col))
            col_kinds[col] = "categorical"

    return space, col_kinds


def label_encoding(df):

    df = df.copy()

    label_encoders = {}

    categorical = df.select_dtypes(include=['object']).columns

    if len(categorical) == 0:
        st.write('No categorical columns found', icon=warning_icon)

    df_encoded = pd.DataFrame()

    encoded_col_list = []
    decoded_col_list = []

    for col in categorical:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
        df_encoded[col] = df[col]

        col = col + '_encoded'
        label_encoders[col] = label_encoder
        encoded_col_list.append(col)

    for col in categorical:
        df_encoded = df_encoded.rename(columns={col: col + '_encoded'})

    decoded = pd.DataFrame()
    categorical = df_encoded.columns

    for col in categorical:
        decoded[col] = label_encoders[col].inverse_transform(df_encoded[col])
        base_name = col[:-len('_encoded')] if col.endswith('_encoded') else col
        decoded = decoded.rename(columns={col: base_name + '_decoded'})
        col = base_name + '_decoded'
        decoded_col_list.append(col)

    res = pd.concat([df_encoded, decoded], axis=1)
    res = res.drop_duplicates()

    st.badge('Categorical columns have been transformed into their encoded and decoded values.', icon=success_icon,
             color="green")
    st.info(
        'The table below shows the encoded and decoded values of each column, simply match the encoded values to decoded for each column and value. This is essential for understanding optimised results as they are encoded.',
        icon=info_icon)
    res = (
        res.style
        .format(precision=2)
        .set_properties(**{"color": "#31333f"})  # data cell text color
        .set_table_styles([
            {
                "selector": "th.col_heading",
                "props": [
                    ("background-color", "#ADBDD9"),  # header fill color
                    ("color", "#2F3D52"),  # header text color (optional)
                ]
            }
        ])
    )

    st.table(res, hide_index=True)

    return df


def hot_encoding(df):
    hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    categorical = df.select_dtypes(include=['object']).columns.tolist()

    encoded_parts = []
    decoded_parts = []
    categorical_groups = {}  # maps original column name -> list of its one-hot column names

    for col in categorical:
        encoded = hot_encoder.fit_transform(df[[col]]).astype(int)
        encoded_parts.append(encoded)
        categorical_groups[col] = encoded.columns.tolist()

        decoded = df[[col]].rename(columns={col: col + '_decoded'})
        decoded_parts.append(decoded)

    display_df = pd.concat(encoded_parts + decoded_parts, axis=1)
    display_df = display_df.drop_duplicates().reset_index(drop=True)

    styled_display = (
        display_df.style
        .format(precision=2)
        .set_properties(**{"color": "#31333f"})
        .set_table_styles([
            {"selector": "th.col_heading",
             "props": [("background-color", "#ADBDD9"), ("color", "#2F3D52")]}
        ])
    )

    st.badge(f'One Hot Encoding has created {sum(e.shape[1] for e in encoded_parts)} new columns.',
             icon=success_icon, color="green")
    st.info(
        'The table below shows the encoded and decoded values of each column, simply match the encoded values to decoded for each column and value. This is essential for understanding optimised results as they are encoded.',
        icon=info_icon)
    st.table(styled_display, hide_index=True)

    st.session_state['categorical_groups'] = categorical_groups

    df_rest = df.drop(columns=categorical)
    full_df = pd.concat([df_rest] + encoded_parts, axis=1)

    return full_df


def decode_optimised_results(feat_df, categorical_groups):
    """
    Collapses one-hot encoded rows in feat_df back into a single row per
    original categorical column, using argmax over the optimised values.
    Only relevant to the One Hot Encoder path -- the 'Bayesian Space' path
    never produces one-hot rows in the first place, so this is skipped there.
    """
    feat_df = feat_df.copy()
    onehot_cols_flat = [c for cols in categorical_groups.values() for c in cols]

    rows_to_keep = feat_df[~feat_df["Feature"].isin(onehot_cols_flat)]

    decoded_rows = []
    for group_name, cols in categorical_groups.items():
        group_rows = feat_df[feat_df["Feature"].isin(cols)]
        if group_rows.empty:
            continue
        winner_row = group_rows.loc[group_rows["Optimal Values"].idxmax()]
        winner_col = winner_row["Feature"]
        category_label = winner_col[len(group_name) + 1:]
        decoded_rows.append({"Feature": group_name, "Optimal Values": category_label})

    if not decoded_rows:
        return feat_df

    decoded_df = pd.DataFrame(decoded_rows)
    result = pd.concat([rows_to_keep, decoded_df], axis=0, ignore_index=True)
    return result


class BayesianOptimiser:
    """Used for the None / One Hot Encoder / Label Encoder paths, where df
    is already fully numeric by the time x_cols is selected."""

    def __init__(self, df, x_features, x=None, y=None, y_col=None, scaler=None, x_scaled=None):
        self.df = df
        self.features = x_features
        self.y = y
        self.y_col = y_col if y_col is not None else df.columns[-1]

        if self.y is None:
            self.y = self.df[self.y_col].values
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
            st.badge('R² value is below 0.7, model may be poorly fitting.', icon=danger_icon, color="red")

        if cv_scores.mean() > 0.98:
            st.badge("R² this high is unusual for noisy real data. Data generated may be unreliable.", icon=danger_icon,
                     color="red")

    def predict(self, minmax='max', optimise_method='Bayesian', feat_df=None, de_res=None, bo_res=None, bounds=None):
        self.feat_df = feat_df
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

        self.feat_df = pd.DataFrame()
        features_list = []
        optimise_list = []

        if optimise_method == 'Differential Evolution':

            self.bounds = []
            for col in self.features:
                low, high = self.df[col].min(), self.df[col].max()
                self.bounds.append((low, high))

            self.de_res = differential_evolution(yield_func, self.bounds, seed=42, maxiter=300, tol=1e-8)

            for n in range(len(self.features)):
                features_list.append(self.features[n])
                optimise_list.append(self.de_res.x[n])

            data = {'Feature': features_list, 'Optimal Values': optimise_list}
            self.feat_df = pd.DataFrame.from_dict(data)

        if optimise_method == 'Bayesian':

            self.bounds = []
            for col in self.features:
                low, high = self.df[col].min(), self.df[col].max()
                self.bounds.append((low, high))

            self.bo_res = gp_minimize(yield_func, self.bounds, n_calls=40, random_state=1, verbose=False)

            for n in range(len(self.features)):
                features_list.append(self.features[n])
                optimise_list.append(self.bo_res.x[n])

            data = {'Feature': features_list, 'Optimal Values': optimise_list}
            self.feat_df = pd.DataFrame.from_dict(data)

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
        saved_ax_imgs = []

        for i in range(len(self.features)):
            label = labels[i]
            lo = self.bounds[i][0]
            hi = self.bounds[i][1]

            sweep = np.linspace(lo, hi, 60)
            preds = []
            for val in sweep:
                p = best.copy()
                p[i] = val
                preds.append(predict_yield(p))

            single_fig, single_ax = plt.subplots(figsize=(6, 4.5))
            single_ax.plot(sweep, preds, color=main_plot_colour, linewidth=2)
            single_ax.axvline(best[i], color=optimal_line_colour, linestyle="--", linewidth=1.5,
                              label=f"optimum = {best[i]:.2f}")
            single_ax.axhline(y=0, color='k',linewidth=1.5, linestyle="--")

            if min(preds) < 0:
                min_y = min(preds) * 1.1
            if min(preds) > 0:
                min_y = min(preds) * 0.9

            if max(preds) < 0:
                max_y = max(preds) * 0.9
            if max(preds) > 0:
                max_y = max(preds) * 1.1

            single_ax.set_ylim(min_y, max_y)
            single_ax.set_xlabel(label)
            single_ax.set_ylabel(f"{self.y_col}")
            single_ax.set_title(f"Sensitivity: {label}")
            single_ax.legend(fontsize=8)
            single_ax.grid(alpha=0.3)
            single_ax.set_facecolor("white")

            buffer = io.BytesIO()
            single_fig.savefig(buffer, format="png", dpi=800, bbox_inches="tight")
            saved_ax_imgs.append(buffer.getvalue())
            plt.close(single_fig)

        row, col = grid_shape(len(self.features))
        fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 4))
        axes = np.array(axes).ravel()

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

            ax.plot(sweep, preds, color=main_plot_colour, linewidth=2)
            ax.axvline(best[i], color=optimal_line_colour, linestyle="--", linewidth=1.5, label=f"optimum = {best[i]:.2f}")
            ax.axhline(y=0, color='k',linewidth=1.5, linestyle="--")

            if min(preds) < 0:
                min_y = min(preds) * 1.1
            if min(preds) > 0:
                min_y = min(preds) * 0.9

            if max(preds) < 0:
                max_y = max(preds) * 0.9
            if max(preds) > 0:
                max_y = max(preds) * 1.1

            ax.set_ylim(min_y, max_y)
            ax.set_xlabel(label)
            ax.set_ylabel(f"{self.y_col}")
            ax.set_title(f"Sensitivity: {label}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_facecolor("white")

        for j in range(len(self.features), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=800, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return buf, saved_ax_imgs


class MixedBayesianOptimiser:
    """
    Used exclusively for the 'Bayesian Space' encoder choice, where NO
    manual encoding is applied to df -- categorical columns stay as their
    original string labels. Builds a genuine skopt mixed space
    (Categorical/Real/Integer) via build_mixed_space, one-hot encodes
    internally purely for surrogate-model fitting, and reports results in
    terms of real category labels (no fractional relaxation, no decode
    step needed afterward).
    """

    def __init__(self, df, x_features, y_col=None):
        self.df = df
        self.features = x_features
        self.y = df.iloc[:, -1].values if y_col is None else df[y_col].values
        self.y_col = y_col if y_col is not None else df.columns[-1]  # <-- add this line

        self.space, self.col_kinds = build_mixed_space(df, x_features)
        self.categorical_cols = [c for c, k in self.col_kinds.items() if k == "categorical"]
        self.numeric_cols = [c for c, k in self.col_kinds.items() if k != "categorical"]

        if self.categorical_cols:
            encoded_df = pd.get_dummies(df[self.categorical_cols], columns=self.categorical_cols, dtype=int)
        else:
            encoded_df = pd.DataFrame(index=df.index)

        self.X_df = pd.concat(
            [df[self.numeric_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)],
            axis=1,
        )
        self.feature_cols = self.X_df.columns.tolist()

        self.scaler = StandardScaler()
        self.x_scaled = self.scaler.fit_transform(self.X_df.values)

    def gpr(self):
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(self.x_scaled.shape[1])) + WhiteKernel(noise_level=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
        self.gp.fit(self.x_scaled, self.y)

        cv_scores = cross_val_score(self.gp, self.x_scaled, self.y, cv=5, scoring="r2")
        st.write(f"Surrogate model 5-fold CV R²: {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")

        if cv_scores.mean() < 0.7:
            st.badge('R² value is below 0.7, model may be poorly fitting.', icon=danger_icon, color="red")
        if cv_scores.mean() > 0.98:
            st.badge("R² this high is unusual for noisy real data. Data generated may be unreliable.",
                     icon=danger_icon, color="red")

    def _build_encoded_row(self, raw_params: dict):
        row = {col: raw_params[col] for col in self.numeric_cols}
        for col in self.categorical_cols:
            chosen = raw_params[col]
            for choice in sorted(self.df[col].dropna().unique()):
                row[f"{col}_{choice}"] = 1 if choice == chosen else 0
        row_df = pd.DataFrame([row])[self.feature_cols]
        return self.scaler.transform(row_df.values)

    def _predict_yield(self, raw_params: dict):
        row_scaled = self._build_encoded_row(raw_params)
        return self.gp.predict(row_scaled)[0]

    def predict(self, minmax="Maximise", optimise_method="Bayesian"):
        self.minmax = minmax
        sign = -1 if minmax == "Maximise" else 1

        def yield_func(raw_params):
            return sign * self._predict_yield(raw_params)

        if optimise_method == "Bayesian":

            @use_named_args(self.space)
            def objective(**params):
                return yield_func(params)

            self.bo_res = gp_minimize(objective, self.space, n_calls=40, random_state=1, verbose=False)
            self.best_values = dict(zip([d.name for d in self.space], self.bo_res.x))

        elif optimise_method == "Differential Evolution":
            numeric_bounds = [(float(self.df[c].min()), float(self.df[c].max())) for c in self.numeric_cols]

            if self.categorical_cols:
                # DE has no native concept of categories -- run one DE pass
                # per real category combination, keep whichever wins.
                combos = list(itertools.product(
                    *[sorted(self.df[c].dropna().unique()) for c in self.categorical_cols]
                ))
                best_score, best_combo, best_numeric = np.inf, None, None

                for combo in combos:
                    combo_dict = dict(zip(self.categorical_cols, combo))

                    def obj(numeric_values, combo_dict=combo_dict):
                        params = {**combo_dict, **dict(zip(self.numeric_cols, numeric_values))}
                        return yield_func(params)

                    res = differential_evolution(obj, numeric_bounds, seed=42, maxiter=300, tol=1e-8)
                    if res.fun < best_score:
                        best_score = res.fun
                        best_combo = combo_dict
                        best_numeric = res.x

                self.best_values = {**best_combo, **dict(zip(self.numeric_cols, best_numeric))}
            else:
                res = differential_evolution(
                    lambda vals: yield_func(dict(zip(self.numeric_cols, vals))),
                    numeric_bounds, seed=42, maxiter=300, tol=1e-8,
                )
                self.best_values = dict(zip(self.numeric_cols, res.x))

        self.feat_df = pd.DataFrame({
            "Feature": list(self.best_values.keys()),
            "Optimal Values": list(self.best_values.values()),
        })
        return self.feat_df

    def plots(self):
        saved_ax_imgs = []

        for feat in self.features:
            single_fig, single_ax = plt.subplots(figsize=(6, 4.5))

            if self.col_kinds[feat] == "categorical":
                choices = sorted(self.df[feat].dropna().unique().tolist())
                preds = []
                for choice in choices:
                    p = self.best_values.copy()
                    p[feat] = choice
                    preds.append(self._predict_yield(p))

                colors = [main_plot_colour if c == self.best_values[feat] else secondary_plot_colour for c in choices]
                single_ax.bar(choices, preds, color=colors)
                single_ax.set_xlabel(feat)

            else:
                lo = float(self.df[feat].min())
                hi = float(self.df[feat].max())
                sweep = np.linspace(lo, hi, 60)
                preds = []
                for val in sweep:
                    p = self.best_values.copy()
                    p[feat] = val
                    preds.append(self._predict_yield(p))

                single_ax.plot(sweep, preds, color=main_plot_colour, linewidth=2)
                single_ax.axvline(self.best_values[feat], color=optimal_line_colour, linestyle="--",
                                  linewidth=1.5, label=f"optimum = {self.best_values[feat]:.2f}")
                single_ax.axhline(y=0, color='k', linewidth=1.5, linestyle="--")

                if min(preds) < 0:
                    min_y = min(preds) * 1.1
                if min(preds) > 0:
                    min_y = min(preds) * 0.9

                if max(preds) < 0:
                    max_y = max(preds) * 0.9
                if max(preds) > 0:
                    max_y = max(preds) * 1.1

                single_ax.set_ylim(min_y, max_y)
                single_ax.legend(fontsize=8)
                single_ax.set_xlabel(feat)

            single_ax.set_ylabel(f"{self.y_col}")
            single_ax.set_title(f"Sensitivity: {feat}")
            single_ax.grid(alpha=0.3)
            single_ax.set_facecolor("white")

            buffer = io.BytesIO()
            single_fig.savefig(buffer, format="png", dpi=800, bbox_inches="tight")
            saved_ax_imgs.append(buffer.getvalue())
            plt.close(single_fig)

        row, col = grid_shape(len(self.features))
        fig, axes = plt.subplots(row, col, figsize=(col * 5, row * 4))
        axes = np.array(axes).ravel()

        for i, feat in enumerate(self.features):
            ax = axes[i]
            if self.col_kinds[feat] == "categorical":
                choices = sorted(self.df[feat].dropna().unique().tolist())
                preds = []
                for choice in choices:
                    p = self.best_values.copy()
                    p[feat] = choice
                    preds.append(self._predict_yield(p))
                colors = [main_plot_colour if c == self.best_values[feat] else secondary_plot_colour for c in choices]
                ax.bar(choices, preds, color=colors)
            else:
                lo, hi = float(self.df[feat].min()), float(self.df[feat].max())
                sweep = np.linspace(lo, hi, 60)
                preds = []
                for val in sweep:
                    p = self.best_values.copy()
                    p[feat] = val
                    preds.append(self._predict_yield(p))
                ax.plot(sweep, preds, color=main_plot_colour, linewidth=2)
                ax.axhline(y=0, color='k',linewidth=1.5, linestyle="--")

                if min(preds) < 0:
                    min_y = min(preds) * 1.1
                if min(preds) > 0:
                    min_y = min(preds) * 0.9

                if max(preds) < 0:
                    max_y = max(preds) * 0.9
                if max(preds) > 0:
                    max_y = max(preds) * 1.1

                ax.set_ylim(min_y , max_y)
                ax.axvline(self.best_values[feat], color=optimal_line_colour, linestyle="--", linewidth=1.5)

            ax.set_xlabel(feat)
            ax.set_ylabel(f"{self.y_col}")
            ax.set_title(f"Sensitivity: {feat}")
            ax.grid(alpha=0.3)
            ax.set_facecolor("white")

        for j in range(len(self.features), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=800, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return buf, saved_ax_imgs

# END OF OPTIMISATION CODEBASE

# DEMO DATA:

@st.cache_data
def load_demo():
    demo_df = pd.read_csv('coupling_reaction_data.csv')
    return demo_df

st.sidebar.title('Navigation')
with st.sidebar:
    st.subheader('Important Sections',divider='gray')
    st.markdown('[:material/star_border: Choose Dataset](#optimiser-tool-using-bayesian-optimisation)')
    st.markdown('[:material/star_border: Data Cleaner](#data-cleaner)')
    st.markdown('[:material/star_border: Data Optimiser](#optimise-data)')

    st.subheader('Optional Sections',divider='gray')

    st.markdown('[Colour Picker](#colour-picker)')
    st.markdown('[Data Preview](#data-preview)')
    st.markdown('[Data Filter](#filter-data)')

    if st.button('Reset'):
        st.session_state.clear()
        st.rerun()


st.title('Optimiser Tool Using Bayesian Optimisation', text_alignment='center')

st.write(
    'Bayesian optimisation is suitable for low parameter spaces (< 15), particularly for costly experiments where exploration is limited.')
st.markdown(f'The typical work flow for this app is:\n' 
         '1) Either use the demo dataset or upload a .csv file with your data;\n'
         '2) Go through the *Data Preview* to ensure the correct data is there, similarly, look at *Data Summary*;\n'
            '3) Clean your data with *Data Cleaner* if it contains any columns you want to remove;\n'
            '4) Go to *Optimise Data* and choose encode method (Bayesian Space recommended), input all x and y factors you want to run;\n'
            '5) Choose the optimisation method (Bayesian recommended) followed by if you want to maximise or minimuise your data;\n'
            '6) Click *Run Optimisation* and wait a couple minutes;\n'
            '7) The optimised data will be generated alongside any plots. These can now be downloaded;\n'
            '8) Click *Reset* at the bottom of the page or in the navigation pane to start over.',text_alignment='justify')

st.divider()
demo_clicked = st.button('Press for a demo dataset')
st.write('or...')
uploaded_file = st.file_uploader('**Choose your own file (in .csv format):**', type=['csv'])

if demo_clicked:
    st.session_state['df'] = load_demo()
    st.session_state['data_source'] = 'demo'

if uploaded_file is not None:
    st.session_state['df'] = pd.read_csv(uploaded_file)
    st.session_state['data_source'] = 'upload'
    if demo_clicked:
        st.warning('If you wish to run the demo, remove your current file.',icon=warning_icon)

if 'df' in st.session_state:
    df = st.session_state['df']

    if st.session_state['data_source'] == 'demo':
        st.info('Demo dataset loaded. Uploading a file will replace it.', icon=info_icon)

    # everything below here is now OUTSIDE the demo-only check,
    # and runs regardless of whether df came from demo or upload:
    st.divider()
    st.subheader('Colour Picker')
    st.markdown('Select your custom colours based on the image below, if you wish to retain the default colour scheme, leave this unchanged.', text_alignment='justify')
    st.markdown('You can adjust each colour and see its result on the demo plot below, if you wish to return to the default scheme, press *Colour Reset*.')
    st.warning(f'You must change the colour **before** running *Optimise Data*. If you wish to re-colour already generated plots, rerun *Optimise Data* after picking your colours.', icon=warning_icon)

    st.image('example_plot.png',caption='Example plot with default colouring', width=960)

    main_plot_colour_c = '#942553'  # the colour of the line
    secondary_plot_colour_c = '#255394'  # the colour of the secondary plot (bar chart)
    optimal_line_colour_c = '#ED5557'  # the colour of the optimal line

    def reset_colour():
        st.session_state.colour_picker_main = main_plot_colour_c
        st.session_state.colour_picker_sec = secondary_plot_colour_c
        st.session_state.colour_picker_opt = optimal_line_colour_c

    col1, col2, col3 = st.columns(3)
    with col1:
        main_plot_colour = st.color_picker('Main Plot Colour', main_plot_colour, key='colour_picker_main')
    with col2:
        secondary_plot_colour = st.color_picker('Secondary Plot Colour', secondary_plot_colour,key='colour_picker_sec' )
    with col3:
        optimal_line_colour = st.color_picker('Optimal Line Colour', optimal_line_colour,key='colour_picker_opt' )

    x = np.linspace(0, 10, 20)
    y = np.sin(x)/2 + x**2/100 + 0.5

    fig, ax = plt.subplots()
    ax.set_ylim(0,2.2)
    ax.set_xlim(0,9)
    ax.plot(x,y, color=main_plot_colour,label='Main Colour Plot')
    ax.bar([1.2,1.5,2],[1,1,1], width=0.5, color= secondary_plot_colour,label='Secondary Plot Colour Plot')
    ax.bar([4.4,4.5,4.6], [0.25,0.25,0.25], color= secondary_plot_colour)
    ax.bar([8.1,8.2],[1.6,1.6], color=main_plot_colour)
    ax.axvline(x=8.2,color=optimal_line_colour, ls='--',label='Optimal Line Colour')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()
    st.pyplot(fig)

    st.button('Reset Colour', on_click=reset_colour)

    st.divider()

    st.subheader("Data Preview")

    decimal_points = st.slider('How many decimal points?:', 0, 8, 1)

    def style_df(df, decimals=decimal_points):
        return (
            df.style
            .format(precision=decimals)
            .set_properties(**{"color": "#31333f"})  # data cell text color
            .set_table_styles([
                {"selector": "th.col_heading", "props": [("background-color", "#ADBDD9"), ("color", "#31333f")]},
                {"selector": "th.blank", "props": [("background-color", "#ADBDD9")]},
                {"selector": "th.row_heading", "props": [("background-color", "#ADBDD9"), ("color", "#31333f")]}
            ])
        )


    st.table(style_df(df.head()))

    st.divider()

    st.subheader("Data Summary")
    st.table(style_df(df.describe()))

    st.divider()

    st.subheader("Filter Data")
    st.info('*Optional, used to look through specific columns.', icon=info_icon)

    columns = df.columns.tolist()
    selected_column = st.multiselect("Select column to filter by:", columns)

    if selected_column:
        st.table(style_df(df[selected_column]), height=450)

    st.divider()

    st.subheader("Data Cleaner")
    st.info('''Please remove any irrelevant columns such as: batch/I.D. numbers.  
                    **Keeping such data will produce irregular and incorrect data.**''',
            icon=info_icon)

    cols_to_remove = st.multiselect("Select columns to remove:", columns)
    columns = list(set(columns) - set(cols_to_remove))
    df = df[columns]

    st.divider()

    st.subheader("Optimise Data")

    st.info( 'Bayesian Space encode method will produce the most accurate results.\n\n'
        '**Note:** You **do not** require categorical data for Bayesian Space, and no separate '
        'encoding step is needed, categorical columns are handled automatically.',
        icon=info_icon)
    encoder_choice = st.radio(
        f'Select encoder method if you have **categorical data**, else press **None** or **Bayesian Space** if you intend to Bayesian optimise:',
        ['Bayesian Space', 'One Hot Encoder', 'Label Encoder', 'None'], horizontal=True)

    if encoder_choice != 'One Hot Encoder' and 'categorical_groups' in st.session_state:
        del st.session_state['categorical_groups']

    if encoder_choice == 'Label Encoder':
        try:
            encoder = label_encoding(df)
            df = encoder
            if df.empty is True:
                st.warning('No data was found, please try again later.')
            columns = df.columns.tolist()
        except:
            st.error('Dataframe was found to have no categorical data!', icon=danger_icon)

    if encoder_choice == 'One Hot Encoder':
        try:
            encoder = hot_encoding(df)
            df = encoder
            columns = df.columns.tolist()
        except:
            st.error('Dataframe was found to have no categorical data!', icon=danger_icon)

    if encoder_choice == 'None':
        categorical_cols = df.select_dtypes(include=["object", "string", "category"])
        if not categorical_cols.empty:
            st.error(
                f"Categorical data detected, ensure you have chosen the correct encoder method! Error with **{categorical_cols.columns}**",
                icon=danger_icon,
            )
        else:
            st.badge('No categorical data detected.', icon=success_icon, color="green")

    if encoder_choice == 'Bayesian Space':
        st.badge('No encoding required, categorical columns will be handled automatically as part of '
                 'Bayesian optimisation.', icon=success_icon, color="green")

    x_cols = st.multiselect("Select x features to use:", columns)

    if len(x_cols) < 2:
        st.warning("Please select at least two x features to use.", icon=warning_icon)

    y_cols = st.multiselect("Select y features to use - ensure to not include any x features:", columns)

    if len(y_cols) < 1:
        st.warning("Please select one y feature to use.", icon=warning_icon)
    if len(y_cols) > 1:
        st.warning("Too many y features selected, outputs will be incorrect.", icon=warning_icon)

    res = list(set(x_cols).intersection(set(y_cols)))
    if len(res) > 0:
        st.error('Warning! There are similar x and/or y variables selected in both feature boxes!', icon=danger_icon)

    if encoder_choice == 'Bayesian Space' and x_cols:
        n_cat = sum(1 for c in x_cols if not pd.api.types.is_numeric_dtype(df[c]))
        n_num = len(x_cols) - n_cat
        st.caption(f"Detected **{n_num}** numeric and **{n_cat}** categorical feature(s) among your selection.")

        if n_cat > 0:
            n_combos = 1
            for c in x_cols:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    n_combos *= df[c].nunique()
            st.caption(f"Categorical combinations: **{n_combos}** "
                       f"(relevant if you choose Differential Evolution, which runs once per combination).")
            if n_combos > 100:
                st.warning(f"{n_combos} combinations may make Differential Evolution slow, "
                           f"Bayesian optimisation is recommended instead.", icon=warning_icon)

    col1, col2 = st.columns(2)
    with col1:
        optimisation_method = st.radio(f'Choose optimisation method **(Bayesian is recommended)**:',
                                       key='optimisation_method',
                                       options=['Bayesian', 'Differential Evolution'])

    with col2:
        max_or_min = st.radio('Maximise or minimise?:',
                              ('Maximise', 'Minimise'))

    if optimisation_method == 'Bayesian':
        st.warning('Running Bayesian optimisation may take a few minutes to run depending on the size of your data.',
                   icon=warning_icon)

    if st.button('Run Optimisation'):
        try:
            if encoder_choice == 'Bayesian Space':

                temp = st.badge('Running calculations...', icon=info_icon, color="blue")
                optimisation = MixedBayesianOptimiser(df, x_cols, y_col=y_cols[0])
                optimisation.gpr()
                temp.empty()
                temp = st.badge('Drawing plots...', icon=info_icon, color="blue")
                optimisation.predict(max_or_min, optimisation_method)
                temp.empty()
                buf, saved_ax_imgs = optimisation.plots()
            else:
                temp = st.badge('Running calculations...', icon=info_icon, color="blue")
                optimisation = BayesianOptimiser(df, x_cols, y_col=y_cols[0])
                optimisation.gpr()
                temp.empty()
                temp = st.badge('Drawing plots...', icon=info_icon, color="blue")
                optimisation.predict(max_or_min, optimisation_method)
                temp.empty()
                buf, saved_ax_imgs = optimisation.plots(optimisation_method)

            st.session_state['plot_buf'] = buf
            st.session_state['plot_individual_imgs'] = saved_ax_imgs
            st.session_state['plot_feature_labels'] = optimisation.features
            st.session_state['results_table'] = optimisation.feat_df
            st.session_state['has_run'] = True

            # Rendered every rerun as long as we've run at least once —
            # reading from session_state, not tied to the button's one-shot True
        except Exception as e:
            categorical_cols = df.select_dtypes(include=["object", "string", "category"])
            if not categorical_cols.empty and encoder_choice != 'Bayesian Space':
                st.error(f'Something went wrong. Please ensure that you have removed or encoded any categorical data.',
                         icon=danger_icon)
            else:
                st.error(f'Something went wrong: {e}', icon=danger_icon)
    if st.session_state.get('has_run'):
        st.subheader(f"--- {optimisation_method} Result ---")
        result_df = st.session_state['results_table']

        # Collapse one-hot encoded rows back into readable category labels,
        # if one-hot encoding was used earlier in the pipeline. The
        # 'Bayesian Space' path never produces one-hot rows, so this is
        # naturally skipped there (no 'categorical_groups' key is set).
        if 'categorical_groups' in st.session_state:
            result_df = decode_optimised_results(result_df, st.session_state['categorical_groups'])

        styled_result = (
            result_df.style
            .set_properties(**{"color": "#31333f"})
            .set_table_styles([
                {"selector": "th.col_heading", "props": [("background-color", "#ADBDD9"), ("color", "#2F3D52")]}
            ])
        )
        st.table(styled_result)

        feature_labels = st.session_state['plot_feature_labels']
        individual_imgs = st.session_state['plot_individual_imgs']

        st.subheader('View all plots')
        with st.expander('Note on sensitivity plots'):
            st.info(
                'Sensitivity plots show each factor as a single variable with the rest of the factors kept at '
                'their optimal values. The wider the slope, the more resistant the variable is to change and vice versa. '
                'Wide slopes are ideal for minimising the factor whilst maintaining a high response.\n\n In the case of categorical data, bar plots are used where the tallest bar is the optimal option. In the case of minimisation, the lowest bar is the optimal option.', icon=info_icon
            )

        #show_full_grid = st.checkbox("--- Show full grid of all sensitivity plots (may be overcrowded) ---",
        #                             width='stretch')
        #if show_full_grid:
        st.image(st.session_state['plot_buf'], use_container_width=True)
        st.download_button(
            label='Download full grid as PNG',
            data=st.session_state['plot_buf'],
            file_name='all_sensitivity_plots.png',
            mime='image/png',
        )
        st.subheader("View an individual sensitivity plot")
        chosen_label = st.selectbox("Choose a feature:", options=feature_labels, key="chosen_feature")
        chosen_index = feature_labels.index(chosen_label)

        plot_placeholder = st.empty()
        plot_placeholder.image(individual_imgs[chosen_index], use_container_width=True)

        st.download_button(
            label='Download current plot as PNG',
            data=individual_imgs[chosen_index],
            file_name=f'{chosen_label}_sensitivity_plot.png',
            mime='image/png',
        )


        if st.button('Reset'):
            st.session_state.clear()
            st.rerun()
else:
    st.info('Upload a CSV or click the demo button to get started.', icon=info_icon)
st.divider()
st.subheader('FAQ')

st.write(f'''Q. What data is supported?\n
 A. Data can be both represented quantitatively or qualitatively. If using qualitative data, ensure that there aren't too many unique values as it'll dramatically increase run time and lead to inconclusive results. For example, if one column contains [Nickel_cat, Ni_cat, Ni_catalyst] rather than just a single [Ni_catalyst] it'll try fit all three.\n
Q. What file formats are supported?\n
A. Only .csv files are supported. If you have data in an excel file (file_name.xlsx), save it as file_name.csv to convert.\n
Q. Do the 'optional' sections matter?\n
A. No, they are used for your viewing and tastes.\n
Q. Is any data altered once uploaded?\n
A. No data is altered, the app makes a copy of the data provided, leaving your file unchanged.\n
Q. What do the encoding methods in *Optimise Data* do?\n
A. Label encoder converts any qualitative values into numbers, e.g., [Ni_cat, Fe_cat, Zn_cat] will transform into [0, 1, 2]. One hot encoding performs a similar action but instead only transforms data into 0 (not present) or 1 (present), this method hence creates new columns for each qualitative value.
Bayesian space categorises your data into real numbers, integers and categorical values. ''',)
