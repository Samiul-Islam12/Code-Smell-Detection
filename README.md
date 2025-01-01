# Code Smell Detection

---

## 📝 Project Overview

Code smells are subtle indicators of poor code quality that can lead to technical debt. This project identifies code smells using machine learning techniques, emphasizing:
- Data preprocessing and feature engineering.
- Detecting code smells like Long Method, Large Class, and God Object.
- Leveraging correlation analysis and machine learning methods for feature selection.

---

## 📂 Project Workflow

1. **Data Overview & Preprocessing**:
   - Dataset loaded from `attribute-details.csv`.
   - Preprocessing involves cleaning data, removing missing values, and encoding categorical features.

2. **Feature Selection**:
   - Correlation analysis identifies relationships between features.
   - Features are refined using a threshold of 0.1 or higher for relevance to the "CodeSmell" target variable.

3. **Model Training**:
   - Multiple machine learning models are trained, including:
     - Random Forest
     - Bagging Classifier
     - KNN
     - XGBoost
   - Hyperparameter tuning optimizes model performance.

4. **Model Evaluation**:
   - Models are evaluated using accuracy, precision, recall, and F1-score.
   - Visualizations include confusion matrices and feature importance plots.

---

## 🚀 Features

1. **Comprehensive Data Analysis**:
   - Automatic feature correlation analysis.
   - Thresholding for selecting optimal features.

2. **Multiple Feature Selection Techniques**:
   - Correlation Based
   - Select From Model
   - Recursive Feature Elimination

3. **Machine Learning Implementation**:
   - Training and comparison of models like Random Forest, Bagging Classifier, KNN and XGBoost.
   - Hyperparameter tuning for enhanced performance.

4. **Detailed Visualizations**:
   - Correlation heatmaps for feature relationships.
   - Confusion matrices and performance metrics.
   - AUC, ROC, and Precision-Recall Curve.

---

## 📂 Dataset

The dataset used in this project contains various code metrics and labeled examples of code smells. It includes features like cyclomatic complexity, line length, and nested block depth.

- **Source**: [Download the Dataset](https://github.com/sayedmohsinreza/CSIQ)

---

## 🛠️ Dependencies

Install the required libraries with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📂 Project Structure

```plaintext
├── Code Smell Detection.ipynb                # Jupyter notebooks for experimentation
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
└── LICENSE                                   # License file
```

---

## 🔍 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Samiul-Islam12/Code-Smell-Detection.git
   cd code-smell-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook notebooks/Code_Smell_Detection.ipynb
   ```

4. Use the provided scripts for preprocessing:
   ```bash
   python scripts/preprocess_data.py
   ```

---

## 🧪 Results

- The **Bagging Classifier** and **Random Forest** demonstrated consistent performance across all feature selection techniques, achieving accuracy rates exceeding **82%**.
- **Correlation-based feature selection** resulted in the lowest misclassification rates, effectively retaining relevant and non-redundant features.
- Models struggled with detecting positive instances, as evidenced by higher counts of **false negatives**, despite class imbalance mitigation using SMOTE.
- Among all models, **KNN** exhibited the lowest accuracy (**70.1%**), indicating sensitivity to feature selection.
- The **Bagging Classifier** emerged as the most robust model, achieving a **Matthews Correlation Coefficient (MCC)** of **0.66**, followed by Random Forest at **0.64**.

These results highlight the effectiveness of ensemble learning methods and strategic feature selection in improving the reliability of code smell detection.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or create pull requests.

---



## 💬 Contact

For inquiries or feedback:
- **Md. Samiul Islam**: [samiul.islam2691@gmail.com](mailto:samiul.islam2691@gmail.com)
- GitHub: [Md. Samiul Islam](https://github.com/Samiul-Islam12)
