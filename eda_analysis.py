import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# STEP 1: Set project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2: Load the dataset
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# STEP 3: Display dataset dimensions
print("\n1. DATASET SHAPE")
print("Rows and columns:", df.shape)
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

# STEP 4: Display first five rows
print("\n2. FIRST FIVE ROWS")
print(df.head())

# STEP 5: Display column information
print("\n3. DATASET INFORMATION")
df.info()

# STEP 6: Display statistical summary
print("\n4. STATISTICAL SUMMARY")
print(df.describe(include="all").to_string())

# STEP 7: Check missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum())

# STEP 8: Check duplicate rows
print("\n6. DUPLICATE ROWS")
print("Duplicate rows:", df.duplicated().sum())

# STEP 9: Analyze student marks
if "Marks" in df.columns:
    marks = pd.to_numeric(df["Marks"], errors="coerce").dropna()

    if not marks.empty:
        print("\n7. MARKS ANALYSIS")
        print("Average marks:", round(marks.mean(), 2))
        print("Median marks:", round(marks.median(), 2))
        print("Highest marks:", marks.max())
        print("Lowest marks:", marks.min())
        print("Standard deviation:", round(marks.std(), 2))

        # Create marks histogram
        plt.figure(figsize=(8, 5))
        plt.hist(marks, bins=5, edgecolor="black")
        plt.title("Distribution of Student Marks")
        plt.xlabel("Marks")
        plt.ylabel("Number of Students")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "marks_distribution.png")
        plt.close()

        # Create marks boxplot
        plt.figure(figsize=(7, 4))
        plt.boxplot(marks)
        plt.title("Student Marks Boxplot")
        plt.ylabel("Marks")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "marks_boxplot.png")
        plt.close()

# STEP 10: Analyze departments
if {"Department", "Marks"}.issubset(df.columns):
    department_data = df[["Department", "Marks"]].copy()
    department_data["Marks"] = pd.to_numeric(
        department_data["Marks"], errors="coerce"
    )
    department_data = department_data.dropna(
        subset=["Department", "Marks"]
    )

    department_average = department_data.groupby(
        "Department"
    )["Marks"].mean()

    print("\n8. AVERAGE MARKS BY DEPARTMENT")
    print(department_average.round(2).to_string())

    if not department_average.empty:
        department_average.plot(kind="bar", figsize=(8, 5))
        plt.title("Average Marks by Department")
        plt.xlabel("Department")
        plt.ylabel("Average Marks")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "department_average.png")
        plt.close()

# STEP 11: Save a text report
report_path = OUTPUT_DIR / "eda_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    report.write("=" * 40 + "\n")
    report.write(f"Dataset rows: {df.shape[0]}\n")
    report.write(f"Dataset columns: {df.shape[1]}\n")
    report.write(f"Duplicate rows: {df.duplicated().sum()}\n")
    report.write("\nMissing values:\n")
    report.write(df.isnull().sum().to_string())
    report.write("\n\nStatistical summary:\n")
    report.write(df.describe(include="all").to_string())

    if "Marks" in df.columns:
        valid_marks = pd.to_numeric(
            df["Marks"], errors="coerce"
        ).dropna()

        if not valid_marks.empty:
            report.write("\n\nMarks analysis:\n")
            report.write(
                f"Mean: {valid_marks.mean():.2f}\n"
            )
            report.write(
                f"Median: {valid_marks.median():.2f}\n"
            )
            report.write(
                f"Highest: {valid_marks.max()}\n"
            )
            report.write(
                f"Lowest: {valid_marks.min()}\n"
            )

print("\n" + "=" * 50)
print("EDA COMPLETED SUCCESSFULLY!")
print("Results saved in:", OUTPUT_DIR)
print("=" * 50)