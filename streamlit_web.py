import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from altair import Categorical
from sympy import isprime
import io

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score

from scipy.optimize import differential_evolution
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

st.html("""
    <style>
        section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
            max-width: 1000px;
        }
    </style>
""")

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

def grid_shape(n):

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def label_encoding(df):
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
        decoded = decoded.rename(columns={col: col.rstrip('_encoded') + '_decoded'})
        col = col.rstrip('_encoded') + '_decoded'
        decoded_col_list.append(col)

    res = pd.concat([df_encoded, decoded], axis=1)
    res = res.drop_duplicates()

    st.badge('Categorical columns have been transformed into their encoded and decoded values.', icon=success_icon,
             color="green")
    st.info(
        'The table below shows the encoded and decoded values of each column, simply match the encoded values to decoded for each column and value. This is not overly important as results are decoded.',
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
        encoded = hot_encoder.fit_transform(df[[col]]).astype(
            int)
        encoded_parts.append(encoded)
        categorical_groups[col] = encoded.columns.tolist()

        decoded = df[[col]].rename(columns={col: col + '_decoded'})  # original labels, renamed for clarity
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
        'The table below shows the encoded and decoded values of each column, simply match the encoded values to decoded for each column and value. This is not overly important as results are decoded.',
        icon=info_icon)
    st.table(styled_display, hide_index=True)

    # Persist the grouping so the results table can be decoded back to
    # readable category labels later, after optimisation runs.
    st.session_state['categorical_groups'] = categorical_groups

    df_rest = df.drop(columns=categorical)
    full_df = pd.concat([df_rest] + encoded_parts, axis=1)

    return full_df

def decode_optimised_results(feat_df, categorical_groups):

    feat_df = feat_df.copy()
    onehot_cols_flat = [c for cols in categorical_groups.values() for c in cols]

    rows_to_keep = feat_df[~feat_df["Feature"].isin(onehot_cols_flat)]

    decoded_rows = []
    for group_name, cols in categorical_groups.items():
        group_rows = feat_df[feat_df["Feature"].isin(cols)]
        if group_rows.empty:
            continue  # this categorical column wasn't part of the chosen x features
        winner_row = group_rows.loc[group_rows["Optimal Values"].idxmax()]
        winner_col = winner_row["Feature"]  # e.g. "catalyst_Pd"
        category_label = winner_col[len(group_name) + 1:]  # strip "catalyst_" -> "Pd"
        decoded_rows.append({"Feature": group_name, "Optimal Values": category_label})

    if not decoded_rows:
        return feat_df  # nothing to decode -- return unchanged

    decoded_df = pd.DataFrame(decoded_rows)
    result = pd.concat([rows_to_keep, decoded_df], axis=0, ignore_index=True)
    return result


class BayesianOptimiser:
    def __init__(self, df, x_features, x=None, y=None, scaler=None, x_scaled=None):
        self.df = df
        self.features = x_features
        self.y = y

        if self.y is None:
            self.y = self.df.iloc[:, -1].values
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
            self.feat_df = pd.DataFrame.from_dict(data)  # keep this as the RAW dataframe, nothing more

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
            single_ax.plot(sweep, preds, color="#2563eb", linewidth=2)
            single_ax.axvline(best[i], color="#dc2626", linestyle="--", linewidth=1.5,
                              label=f"optimum = {best[i]:.2f}")
            single_ax.set_xlabel(label)
            single_ax.set_ylabel(f"{self.df.columns[-1]}")
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

            ax.plot(sweep, preds, color="#2563eb", linewidth=2)
            ax.axvline(best[i], color="#dc2626", linestyle="--", linewidth=1.5, label=f"optimum = {best[i]:.2f}")
            ax.set_xlabel(label)
            ax.set_ylabel(f"{self.df.columns[-1]}")
            ax.set_title(f"Sensitivity: {label}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_facecolor("white")

        # hide any unused subplot slots (e.g. the 4th panel with only 3 features)
        for j in range(len(self.features), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=800, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return fig, buf, saved_ax_imgs


# END OF OPTIMISATION CODEBASE


st.title('Optimiser Tool Using Bayesian Optimisation')

st.write(
    'Bayesian optimisation is suitable for low parameter spaces (< 15), particularly for costly experiments where exploration is limited.')
st.write('For more detail: https://doi.org/10.1039/d3dd00234a')

danger_icon = ':material/error:'
success_icon = ':material/check:'
warning_icon = ':material/warning:'
info_icon = ':material/info:'

st.divider()
uploaded_file = st.file_uploader(f'**Choose a file (in .csv format):**', type=['csv'], accept_multiple_files=False)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown(f'**Chosen file:** {uploaded_file.name}')

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
                    Keeping such data will produce irregular and incorrect data, **this is essential if encoding.**''',
            icon=info_icon)

    cols_to_remove = st.multiselect("Select columns to remove:", columns)
    columns = list(set(columns) - set(cols_to_remove))
    df = df[columns]

    st.divider()

    st.subheader("Optimise Data")

    st.info('OneHotEncoder is encouraged to prevent numerical advantages.', icon=info_icon)
    encoder_choice = st.radio('Select encoder method if you have categorical data, else press None:',
                              ['None', 'One Hot Encoder', 'Label Encoder'], horizontal=True)

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

    col1, col2 = st.columns(2)
    with col1:
        optimisation_method = st.radio('Choose optimisation method:',
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
            optimisation = BayesianOptimiser(df, x_cols)
            optimisation.gpr()
            optimisation.predict(max_or_min, optimisation_method)
            fig, buf, saved_ax_imgs = optimisation.plots(optimisation_method)

            st.session_state['plot_fig'] = fig
            st.session_state['plot_buf'] = buf
            st.session_state['plot_individual_imgs'] = saved_ax_imgs
            st.session_state['plot_feature_labels'] = optimisation.features
            st.session_state['results_table'] = optimisation.feat_df
            st.session_state['has_run'] = True

            # Rendered every rerun as long as we've run at least once —
            # reading from session_state, not tied to the button's one-shot True
        except:
            categorical_cols = df.select_dtypes(include=["object", "string", "category"])
            if not categorical_cols.empty:
                st.error(f'Something went wrong. Please ensure that you have removed or encoded any categorical data.',
                         icon=danger_icon)
            else:
                st.error('Something went wrong. Ensure that you have chosen x and y features.', icon=danger_icon)
    if st.session_state.get('has_run'):

        st.subheader(f"--- {optimisation_method} Result ---")
        result_df = st.session_state['results_table']

        # Collapse one-hot encoded rows back into readable category labels,
        # if one-hot encoding was used earlier in the pipeline.
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

        show_full_grid = st.checkbox("Show full grid of all sensitivity plots (may be overcrowded)")
        if show_full_grid:
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

        with st.expander('Note on sensitivity plots'):
            st.info(
                'Sensitivity plots show each factor as a single variable with the rest of the factors kept at '
                'their optimal values. The wider the slope, the more resistant the variable is to change and vice versa. '
                'Wide slopes are ideal for minimising the factor whilst maintaining a high response.', icon=info_icon
            )

        st.download_button(
            label='Download current plot as PNG',
            data=individual_imgs[chosen_index],
            file_name=f'{chosen_label}_sensitivity_plot.png',
            mime='image/png',
        )

        if st.button('Reset plot'):
            st.session_state.clear()
            st.rerun()


