# 🚗 Car Data Cleaning & Exploratory Data Analysis

A complete **Data Cleaning and Exploratory Data Analysis (EDA)** project using Python to analyze and understand automotive data.

This project demonstrates a practical data science workflow — from loading and cleaning raw data to performing statistical analysis and creating meaningful visualizations.

---

## 📌 Project Overview

The goal of this project is to clean, preprocess, analyze, and visualize car-related data to identify patterns, relationships, trends, and potential data-quality issues.

### 🔄 Data Science Workflow

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Data Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Statistical Analysis
     ↓
Data Visualization
     ↓
Insights & Conclusions
```

---

## ✨ Features

* 📂 Load and inspect automotive datasets
* 🧹 Handle missing values
* 🔄 Detect and remove duplicate records
* 🔍 Identify data-quality issues
* 📊 Generate descriptive statistics
* 📈 Perform Exploratory Data Analysis
* 📉 Analyze distributions and relationships
* 🔗 Perform correlation analysis
* 📊 Create data visualizations
* 📝 Document findings and conclusions
* 📓 Complete analysis using Jupyter Notebook

---

## 📁 Project Structure

```text
Car/
│
├── CAR DATA CLEANING.ipynb
├── Cars.csv
├── Cars.xls
├── CARS_EDA_PROJECT_DOCUMENT.pdf
├── car.py
├── requirements.txt
└── README.md
```

### 📄 File Description

| File                            | Description                                    |
| ------------------------------- | ---------------------------------------------- |
| `CAR DATA CLEANING.ipynb`       | Main notebook containing data cleaning and EDA |
| `Cars.csv`                      | Car dataset in CSV format                      |
| `Cars.xls`                      | Car dataset in Excel format                    |
| `CARS_EDA_PROJECT_DOCUMENT.pdf` | Detailed project report                        |
| `car.py`                        | Python script containing project-related code  |
| `requirements.txt`              | Required Python libraries                      |
| `README.md`                     | Project documentation                          |

---

## 📊 Dataset

The project uses an automotive dataset containing information related to cars.

The dataset is provided in two formats:

* **CSV** — `Cars.csv`
* **Excel** — `Cars.xls`

The dataset is used for:

* Data inspection
* Data cleaning
* Statistical analysis
* Relationship analysis
* Visualization
* Insight generation

> **Note:** For detailed information about the dataset and analysis, refer to `CARS_EDA_PROJECT_DOCUMENT.pdf`.

---

## 🧹 Data Cleaning

The project covers several important data-cleaning techniques.

### Missing Values

Missing values are checked using:

```python
df.isnull().sum()
```

Missing-value percentages are also calculated to understand the extent of missing data.

### Duplicate Records

Duplicate rows are identified using:

```python
df.duplicated().sum()
```

Duplicate records can then be removed using:

```python
df.drop_duplicates(inplace=True)
```

### Data Type Inspection

The dataset is examined using:

```python
df.info()
```

This helps identify incorrect or inconsistent data types.

### Statistical Inspection

Basic statistical information is generated using:

```python
df.describe()
```

---

## 🔎 Exploratory Data Analysis

The project performs different levels of EDA, including:

### 📌 Univariate Analysis

Analysis of individual variables to understand:

* Distributions
* Central tendency
* Spread
* Outliers
* Frequency patterns

### 📌 Bivariate Analysis

Analysis of relationships between two variables.

### 📌 Correlation Analysis

Correlation analysis is performed to understand relationships between numerical variables.

### 📌 Data Visualization

Visualizations are created using:

* Histograms
* Bar charts
* Box plots
* Scatter plots
* Correlation heatmaps
* Distribution plots

---

## 📈 Key Analysis Areas

The analysis focuses on understanding:

* Car-related feature distributions
* Relationships between numerical variables
* Possible correlations between features
* Data-quality issues
* Duplicate records
* Missing values
* Outliers and unusual observations
* Patterns identified through visualization

Detailed results and visualizations are available in:

```text
CARS_EDA_PROJECT_DOCUMENT.pdf
```

---

## 🛠️ Technologies Used

### Programming Language

* 🐍 Python

### Data Analysis

* 🐼 Pandas
* 🔢 NumPy

### Data Visualization

* 📊 Matplotlib
* 📈 Seaborn

### Development Environment

* 📓 Jupyter Notebook

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/piyush00500/Car.git
```

### 2. Navigate to the Project

```bash
cd Car
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
CAR DATA CLEANING.ipynb
```

and execute the notebook cells sequentially.

---

## 💻 Usage

### Run the Jupyter Notebook

1. Clone the repository.
2. Install the required dependencies.
3. Open Jupyter Notebook.
4. Open `CAR DATA CLEANING.ipynb`.
5. Run the cells in order.
6. Review the cleaning process, visualizations, and analysis results.

---

## 🎯 Learning Outcomes

Through this project, I practiced:

* ✅ Python for data analysis
* ✅ Pandas data manipulation
* ✅ NumPy numerical operations
* ✅ Missing-value analysis
* ✅ Duplicate detection and removal
* ✅ Data preprocessing
* ✅ Exploratory Data Analysis
* ✅ Statistical analysis
* ✅ Data visualization
* ✅ Correlation analysis
* ✅ Jupyter Notebook workflow
* ✅ Data science project documentation

---

## 📚 Project Report

A detailed project report is included in:

```text
CARS_EDA_PROJECT_DOCUMENT.pdf
```

The report provides additional information about:

* Project objectives
* Dataset
* Methodology
* Data cleaning
* Exploratory analysis
* Visualizations
* Findings
* Conclusions

---

## 🔮 Future Improvements

Possible future improvements include:

* 🤖 Building a machine learning model for car price prediction
* 📊 Creating an interactive Streamlit dashboard
* 🗄️ Connecting the project to MySQL
* ☁️ Deploying the application online
* 📈 Adding more advanced statistical analysis
* 🧠 Applying feature engineering and predictive modeling

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

### Contribution Steps

```bash
git checkout -b feature/improvement
```

Make your changes and commit them:

```bash
git add .
git commit -m "Add improvement"
```

Push your branch:

```bash
git push origin feature/improvement
```

Then open a Pull Request.

---

## 👨‍💻 Author

### Piyush

B.Tech Electronics & Communication Engineering Student
Interested in **Data Science, AI/ML, Software Development, and Technology**.

🔗 **GitHub:**
https://github.com/piyush00500

---

## 📚 References

* [Pandas Documentation](https://pandas.pydata.org/)
* [NumPy Documentation](https://numpy.org/)
* [Matplotlib Documentation](https://matplotlib.org/)
* [Seaborn Documentation](https://seaborn.pydata.org/)
* [Jupyter Documentation](https://jupyter.org/)

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

For questions, suggestions, or collaboration, feel free to open an issue or submit a pull request.

---

### 📌 Project Status

**Status:** 🟢 Active Development

**Last Updated:** September 2026
