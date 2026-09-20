import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Professional Data Analysis Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid")

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)

        # Default file expected beside app.py
        return pd.read_csv("Cars.csv")
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Unable to read the dataset: {e}")
        return None


# ============================================================
# DATA CLEANING
# Reproduces the important cleaning steps from the supplied
# CAR DATA CLEANING notebook, while remaining safe for other
# datasets.
# ============================================================
@st.cache_data
def clean_data(df):
    data = df.copy()

    # Remove completely empty rows/columns
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Dataset-specific raw columns found in the supplied notebook.
    # They are checked before use so the application does not
    # crash when another dataset is uploaded.
    if "New_Price" in data.columns:
        data = data.drop(columns=["New_Price"])

    # Split combined vehicle specification columns if present.
    for source_col, value_col, unit_col in [
        ("Mileage", "Mileage_value", "Mileage_unit"),
        ("Engine", "Engine_value", "Engine_unit"),
        ("Power", "Power_value", "Power_unit"),
    ]:
        if source_col in data.columns and value_col not in data.columns:
            parts = data[source_col].astype("string").str.split(" ", n=1, expand=True)
            data[value_col] = pd.to_numeric(parts[0], errors="coerce")
            if unit_col not in data.columns:
                data[unit_col] = parts[1] if parts.shape[1] > 1 else np.nan

    # Split Name into Company_name and Model_name where possible.
    if "Name" in data.columns:
        if "Company_name" not in data.columns or "Model_name" not in data.columns:
            parts = data["Name"].astype("string").str.rsplit(" ", n=1, expand=True)
            if "Company_name" not in data.columns:
                data["Company_name"] = parts[0]
            if "Model_name" not in data.columns:
                data["Model_name"] = parts[1] if parts.shape[1] > 1 else data["Name"]
        data = data.drop(columns=["Name"])

    # Convert numeric-looking columns where appropriate.
    for col in data.columns:
        if data[col].dtype == "object":
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().mean() >= 0.90:
                data[col] = converted

    # Convert obvious date columns.
    for col in list(data.columns):
        if data[col].dtype == "object":
            name = col.lower()
            if "date" in name or "time" in name:
                converted = pd.to_datetime(data[col], errors="coerce")
                if converted.notna().mean() >= 0.70:
                    data[col] = converted

    # Remove exact duplicates.
    data = data.drop_duplicates().reset_index(drop=True)

    # Fill missing categorical values with mode.
    categorical = data.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in categorical:
        if data[col].isna().any():
            mode = data[col].mode(dropna=True)
            if not mode.empty:
                data[col] = data[col].fillna(mode.iloc[0])

    # Fill numeric missing values with median.
    numeric = data.select_dtypes(include=np.number).columns.tolist()
    for col in numeric:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    return data


# ============================================================
# COLUMN TYPE DETECTION
# ============================================================
def get_column_types(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    boolean = df.select_dtypes(include=["bool"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [
        c for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if c not in boolean
    ]

    return numeric, categorical, datetime, boolean


def find_price_column(df):
    candidates = [c for c in df.columns if c.lower() in ["price", "selling_price", "sale_price"]]
    if candidates:
        return candidates[0]

    for c in df.columns:
        if "price" in c.lower():
            return c

    return None


# ============================================================
# GENERAL HELPERS
# ============================================================
def safe_fig():
    return plt.subplots(figsize=(10, 5))


def format_number(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return str(value)


def insight(text):
    st.info(text)


def outlier_report(df, numeric_cols):
    rows = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty or series.nunique() < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((series < lower) | (series > upper)).sum())

        rows.append({
            "Column": col,
            "Potential Outliers": count,
            "Outlier %": round((count / len(series)) * 100, 2)
        })

    return pd.DataFrame(rows)


# ============================================================
# INTRODUCTION
# ============================================================
def show_introduction(df, raw_df, numeric, categorical, datetime, boolean):
    st.header("01 | Introduction")

    st.write(
        "This dashboard presents a professional exploratory analysis of the supplied "
        "dataset. The application detects feature types automatically, evaluates data "
        "quality, explores distributions and relationships, and generates conclusions "
        "from calculated values rather than hardcoded statistics."
    )

    st.subheader("Dataset Overview")

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Numerical Features", f"{len(numeric):,}")
    c4.metric("Categorical Features", f"{len(categorical):,}")
    c5.metric("Missing Values", f"{missing:,}")
    c6.metric("Duplicate Rows", f"{duplicates:,}")

    st.subheader("Project Objective")
    st.write(
        "The objective is to understand the structure, quality, distributions, "
        "relationships and potentially useful predictive features in the dataset. "
        "The analysis does not remove statistical outliers automatically."
    )

    st.subheader("Feature Information")
    feature_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Unique Values": [df[c].nunique(dropna=True) for c in df.columns],
        "Missing Values": [df[c].isna().sum() for c in df.columns]
    })
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    with st.expander("Numerical Features"):
        if numeric:
            st.write(", ".join(numeric))
        else:
            st.write("No numerical features detected.")

    with st.expander("Categorical Features"):
        if categorical:
            st.write(", ".join(categorical))
        else:
            st.write("No categorical features detected.")

    with st.expander("Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    with st.expander("Original Dataset vs Processed Dataset"):
        comparison = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
            "Original": [
                len(raw_df),
                len(raw_df.columns),
                int(raw_df.isna().sum().sum()),
                int(raw_df.duplicated().sum())
            ],
            "Processed": [
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum())
            ]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)


# ============================================================
# DATA QUALITY
# ============================================================
def show_data_quality(df, numeric, categorical, datetime, boolean):
    st.subheader("Data Quality Analysis")

    missing_report = pd.DataFrame({
        "Missing Values": df.isna().sum(),
        "Missing %": (df.isna().sum() / len(df) * 100).round(2)
    })
    missing_report = missing_report[missing_report["Missing Values"] > 0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Missing Values", int(df.isna().sum().sum()))
    c2.metric("Duplicate Rows", int(df.duplicated().sum()))
    c3.metric("Constant Columns", sum(df[c].nunique(dropna=False) <= 1 for c in df.columns))

    if missing_report.empty:
        st.success("No missing values are present in the processed dataset.")
    else:
        st.warning("Some columns still contain missing values.")
        st.dataframe(
            missing_report.reset_index().rename(columns={"index": "Column"}),
            use_container_width=True,
            hide_index=True
        )

    dtype_counts = pd.Series({
        "Numerical": len(numeric),
        "Categorical": len(categorical),
        "Date/Time": len(datetime),
        "Boolean": len(boolean)
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    dtype_counts.plot(kind="bar", ax=ax)
    ax.set_title("Feature Type Distribution")
    ax.set_xlabel("Feature Type")
    ax.set_ylabel("Number of Columns")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        f"The processed dataset contains {len(numeric)} numerical, "
        f"{len(categorical)} categorical, {len(datetime)} date/time and "
        f"{len(boolean)} boolean features."
    )


# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================
def show_univariate(df, numeric, categorical):
    st.subheader("Univariate Analysis")

    if numeric:
        st.markdown("#### Numerical Distributions")

        selected_num = st.selectbox(
            "Select a numerical feature",
            numeric,
            key="univariate_numeric"
        )

        series = df[selected_num].dropna()

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(series, kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_num}")
            ax.set_xlabel(selected_num)
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=series, ax=ax)
            ax.set_title(f"Box Plot of {selected_num}")
            ax.set_xlabel(selected_num)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        skewness = series.skew()
        st.write(
            f"For **{selected_num}**, the mean is {series.mean():.2f}, "
            f"the median is {series.median():.2f}, the standard deviation is "
            f"{series.std():.2f}, and the skewness is {skewness:.2f}."
        )

        if abs(skewness) > 1:
            insight("The selected feature is strongly skewed, so the median may describe its typical value better than the mean.")
        elif abs(skewness) > 0.5:
            insight("The selected feature shows moderate skewness and should be interpreted with both mean and median.")
        else:
            insight("The selected feature is relatively balanced around its central tendency.")

    if categorical:
        st.markdown("#### Categorical Distributions")

        selected_cat = st.selectbox(
            "Select a categorical feature",
            categorical,
            key="univariate_categorical"
        )

        counts = df[selected_cat].value_counts(dropna=False)

        if len(counts) > 15:
            counts = counts.head(15)
            st.info("Only the 15 most frequent categories are shown because the selected feature has high cardinality.")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
        ax.set_title(f"Most Frequent Categories: {selected_cat}")
        ax.set_xlabel(selected_cat)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        top_category = counts.index[0]
        top_count = counts.iloc[0]
        percentage = top_count / len(df) * 100

        insight(
            f"The most frequent category for {selected_cat} is "
            f"'{top_category}', representing approximately {percentage:.2f}% of the processed records."
        )


# ============================================================
# BIVARIATE ANALYSIS
# ============================================================
def show_bivariate(df, numeric, categorical):
    st.subheader("Bivariate Analysis")

    if len(numeric) >= 2:
        st.markdown("#### Numerical vs Numerical")

        x_col, y_col = st.columns(2)
        with x_col:
            x = st.selectbox("X-axis numerical feature", numeric, key="bi_x")
        with y_col:
            y_options = [c for c in numeric if c != x]
            y = st.selectbox("Y-axis numerical feature", y_options, key="bi_y")

        plot_df = df[[x, y]].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=plot_df, x=x, y=y, ax=ax, alpha=0.6)
        ax.set_title(f"{x} vs {y}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        corr = plot_df[x].corr(plot_df[y])
        st.write(f"The Pearson correlation between {x} and {y} is {corr:.3f}.")

        if abs(corr) >= 0.7:
            insight("The selected variables have a strong linear association in this dataset.")
        elif abs(corr) >= 0.4:
            insight("The selected variables have a moderate linear association.")
        else:
            insight("The selected variables have a weak linear association; other factors may explain more of their variation.")

    if categorical and numeric:
        st.markdown("#### Categorical vs Numerical")

        cat = st.selectbox("Categorical feature", categorical, key="cat_num_cat")
        num = st.selectbox("Numerical feature", numeric, key="cat_num_num")

        grouped = (
            df.groupby(cat)[num]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=grouped, x=cat, y=num, ax=ax)
        ax.set_title(f"Average {num} by {cat}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        highest = grouped.iloc[0]
        st.write(
            f"Among the displayed categories, '{highest[cat]}' has the highest "
            f"average {num} at {highest[num]:.2f}."
        )

    if len(categorical) >= 2:
        st.markdown("#### Categorical vs Categorical")

        c1, c2 = st.columns(2)
        with c1:
            cat1 = st.selectbox("First categorical feature", categorical, key="cc1")
        with c2:
            cat2_options = [c for c in categorical if c != cat1]
            cat2 = st.selectbox("Second categorical feature", cat2_options, key="cc2")

        cross = pd.crosstab(df[cat1], df[cat2])

        if cross.shape[0] > 15:
            cross = cross.loc[cross.sum(axis=1).sort_values(ascending=False).head(15).index]

        if cross.shape[1] > 12:
            cross = cross[cross.sum(axis=0).sort_values(ascending=False).head(12).index]

        st.dataframe(cross, use_container_width=True)

        fig, ax = plt.subplots(figsize=(11, 5))
        cross.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"{cat1} and {cat2}")
        ax.set_xlabel(cat1)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# MULTIVARIATE ANALYSIS
# ============================================================
def show_multivariate(df, numeric, categorical):
    st.subheader("Multivariate Analysis")

    if len(numeric) >= 2:
        st.markdown("#### Correlation Heatmap")

        corr = df[numeric].corr()

        fig, ax = plt.subplots(figsize=(11, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Numerical Feature Correlation Matrix")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = corr_pairs.stack().sort_values(key=lambda s: s.abs(), ascending=False)

        if not pairs.empty:
            a, b = pairs.index[0]
            value = pairs.iloc[0]
            st.write(f"The strongest absolute numerical correlation is between {a} and {b}, with correlation {value:.3f}.")

    if categorical and numeric:
        st.markdown("#### Grouped Multivariate Analysis")

        cat_options = [c for c in categorical if df[c].nunique() <= 20]
        if len(cat_options) >= 1:
            group_col = st.selectbox("Grouping feature", cat_options, key="multi_group")
            value_col = st.selectbox("Numerical feature", numeric, key="multi_value")

            grouped = (
                df.groupby(group_col)[value_col]
                .agg(["mean", "median", "count"])
                .sort_values("mean", ascending=False)
                .head(15)
            )

            st.dataframe(grouped.round(2), use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=grouped.reset_index(),
                x=group_col,
                y="mean",
                ax=ax
            )
            ax.set_title(f"Average {value_col} by {group_col}")
            ax.set_ylabel(f"Average {value_col}")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
def show_statistics(df, numeric, categorical):
    st.subheader("Statistical Analysis")

    if numeric:
        st.markdown("#### Numerical Summary")
        summary = df[numeric].describe().T
        summary["median"] = df[numeric].median()
        summary["IQR"] = df[numeric].quantile(0.75) - df[numeric].quantile(0.25)
        summary["skewness"] = df[numeric].skew()
        st.dataframe(summary.round(3), use_container_width=True)

    if categorical:
        st.markdown("#### Categorical Summary")

        rows = []
        for col in categorical:
            counts = df[col].value_counts(dropna=False)
            if not counts.empty:
                rows.append({
                    "Column": col,
                    "Unique Categories": df[col].nunique(dropna=True),
                    "Most Frequent": str(counts.index[0]),
                    "Frequency": int(counts.iloc[0]),
                    "Frequency %": round(counts.iloc[0] / len(df) * 100, 2)
                })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================
def show_outliers(df, numeric):
    st.subheader("Outlier Analysis")

    report = outlier_report(df, numeric)

    if report.empty:
        st.info("There are no numerical features with enough distinct values for an IQR outlier assessment.")
        return

    st.dataframe(report, use_container_width=True, hide_index=True)

    valid = report[report["Potential Outliers"] > 0].sort_values(
        "Potential Outliers", ascending=False
    )

    if valid.empty:
        st.success("No potential outliers were detected using the 1.5 × IQR rule.")
    else:
        selected = st.selectbox(
            "Select a feature for outlier visualization",
            valid["Column"].tolist(),
            key="outlier_feature"
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df[selected], ax=ax)
        ax.set_title(f"Potential Outliers in {selected}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write(
            "Potential outliers are statistical observations outside the "
            "1.5 × IQR boundaries. They are not automatically removed because "
            "an outlier can represent either a valid observation or a data-quality issue."
        )


# ============================================================
# CONCLUSION GENERATION
# ============================================================
def generate_conclusions(df, numeric, categorical, raw_df):
    conclusions = []

    # 1. Dataset scale
    conclusions.append(
        f"The processed dataset contains {len(df):,} records across {len(df.columns)} features, "
        f"providing a substantial base for exploratory analysis."
    )

    # 2. Data quality
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    conclusions.append(
        f"After preprocessing, the dataset contains {missing:,} missing values and "
        f"{duplicates:,} duplicate rows."
    )

    # 3. Numerical structure
    if numeric:
        conclusions.append(
            f"The dataset contains {len(numeric)} numerical features. "
            f"Their distributions should be considered when selecting transformations "
            f"and features for future machine-learning models."
        )
    else:
        conclusions.append("No numerical features were detected, so numerical statistical modelling would require additional feature preparation.")

    # 4. Categorical structure
    if categorical:
        cardinality = {c: df[c].nunique() for c in categorical}
        high_card = max(cardinality, key=cardinality.get)
        conclusions.append(
            f"The dataset contains {len(categorical)} categorical features. "
            f"'{high_card}' has the highest category count at {cardinality[high_card]:,}, "
            f"which may require encoding or grouping in a machine-learning pipeline."
        )
    else:
        conclusions.append("No categorical features were detected in the processed dataset.")

    # 5. Price-specific conclusion when available
    price_col = find_price_column(df)
    if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
        p = df[price_col].dropna()
        conclusions.append(
            f"For {price_col}, the average is {p.mean():.2f}, the median is {p.median():.2f}, "
            f"and the observed range is {p.min():.2f} to {p.max():.2f}."
        )

    # 6. Strongest correlation
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if not upper.empty:
            pair = upper.abs().idxmax()
            value = corr.loc[pair[0], pair[1]]
            conclusions.append(
                f"The strongest absolute numerical correlation is between {pair[0]} and "
                f"{pair[1]}, with a correlation of {value:.3f}. Correlation should not "
                f"be interpreted as causation."
            )
        else:
            conclusions.append("No meaningful pairwise numerical correlation could be calculated.")

    # 7. Dominant categorical feature
    if categorical:
        usable = [c for c in categorical if df[c].nunique(dropna=True) > 1]
        if usable:
            col = usable[0]
            counts = df[col].value_counts()
            conclusions.append(
                f"For the categorical feature '{col}', the most frequent category is "
                f"'{counts.index[0]}', appearing in {counts.iloc[0]:,} records "
                f"({counts.iloc[0] / len(df) * 100:.2f}%)."
            )
        else:
            conclusions.append("Categorical features were detected, but they do not contain multiple categories.")

    # 8. Outliers
    outliers = outlier_report(df, numeric)
    if not outliers.empty:
        total_potential = int(outliers["Potential Outliers"].sum())
        columns_with_outliers = int((outliers["Potential Outliers"] > 0).sum())
        conclusions.append(
            f"The IQR method identifies {total_potential:,} potential outlier observations "
            f"across {columns_with_outliers} numerical features. These observations require "
            f"domain validation before removal."
        )
    else:
        conclusions.append("The IQR outlier assessment could not be applied to the available numerical features.")

    # 9. Data transformation
    conclusions.append(
        f"Preprocessing changed the dataset from {len(raw_df):,} original rows and "
        f"{len(raw_df.columns)} original columns to {len(df):,} rows and {len(df.columns)} processed columns."
    )

    # 10. ML readiness
    conclusions.append(
        "The processed dataset is suitable for the next stage of a data-science workflow, "
        "such as feature encoding, train-test splitting, feature selection and predictive modelling, "
        "subject to domain-specific validation."
    )

    return conclusions[:10]


def show_conclusion(df, numeric, categorical, raw_df):
    st.header("03 | Conclusion")

    st.write(
        "The following ten conclusions are generated from the processed dataset and "
        "the calculated EDA results. They are observations from the data, not fabricated business claims."
    )

    conclusions = generate_conclusions(df, numeric, categorical, raw_df)

    for i, point in enumerate(conclusions, start=1):
        st.markdown(f"**{i}. {point}**")

    st.subheader("Final Assessment")
    st.write(
        "The analysis establishes the dataset's structure, quality, distributions, "
        "relationships and potential modelling considerations. For future machine-learning "
        "work, the most important next steps are domain validation, appropriate categorical "
        "encoding, feature selection, outlier review and evaluation using a separate validation set."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    st.title("DATA ANALYSIS DASHBOARD")
    st.caption("Professional exploratory analysis based on the supplied dataset")

    with st.sidebar:
        st.header("Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="If no file is uploaded, the application looks for Cars.csv beside app.py."
        )

    raw_df = load_data(uploaded_file)

    if raw_df is None:
        st.error(
            "Dataset not found. Place Cars.csv in the same folder as app.py, "
            "or upload a CSV file using the sidebar."
        )
        return

    if raw_df.empty:
        st.error("The uploaded dataset is empty.")
        return

    df = clean_data(raw_df)

    if df.empty:
        st.error("The dataset became empty after preprocessing.")
        return

    numeric, categorical, datetime, boolean = get_column_types(df)

    # Navigation remains on one page.
    section = st.radio(
        "Dashboard Sections",
        ["01 | Introduction", "02 | Exploratory Data Analysis", "03 | Conclusion"],
        horizontal=True
    )

    if section == "01 | Introduction":
        show_introduction(df, raw_df, numeric, categorical, datetime, boolean)

    elif section == "02 | Exploratory Data Analysis":
        st.header("02 | Exploratory Data Analysis")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Data Quality",
            "Univariate Analysis",
            "Bivariate Analysis",
            "Multivariate Analysis",
            "Statistics",
            "Outliers"
        ])

        with tab1:
            show_data_quality(df, numeric, categorical, datetime, boolean)

        with tab2:
            show_univariate(df, numeric, categorical)

        with tab3:
            show_bivariate(df, numeric, categorical)

        with tab4:
            show_multivariate(df, numeric, categorical)

        with tab5:
            show_statistics(df, numeric, categorical)

        with tab6:
            show_outliers(df, numeric)

    else:
        show_conclusion(df, numeric, categorical, raw_df)


if __name__ == "__main__":
    main()
