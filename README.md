# Virtual Logistics Data Analytics Internship

This repository tracks project progress, strategic planning reports, data cleaning scripts, exploratory visualizations and predictive analytical models across all four internship modules.

---

## Detailed Task Summaries

### Task 1: Strategic Planning and Analytics Roadmap
* **Objective:** Define a realistic logistics scenario, set performance metrics, outline an analytics roadmap and provide foundational Python code.
* **Scenario:** Focused on Apex Regional Logistics, operating across five Midwest hubs with 120 trucks handling roughly 15,000 daily orders.
* **Target Metrics:** On-time in-full delivery rate above 95%, stockout rate below 3% and a 12% reduction in fuel expenses.
* **Methodology:** Application of regression for demand, K-Means for spatial clustering and statistical safety stock formulas.
* **Folder Files:**
  * `Task_1/Strategic_Planning_Report.docx`: Full strategic report detailing project scope, research background and execution roadmap.
  * `Task_1/src/logistics_analytics.py`: Python code covering basic data cleaning, demand regression, spatial clustering and safety stock formulas.

### Task 2: Data Collection and Preprocessing Pipeline
* **Objective:** Build a data cleaning pipeline using a public benchmark dataset to prepare raw records for further analysis.
* **Reference Dataset:** DataCo Smart Supply Chain Dataset from Kaggle containing 180,519 records and 53 attributes.
* **Data Quality Fixes:**
  * Dropped sparse attributes missing over 50% of values and used median imputation for missing lead times to prevent outlier distortion.
  * Used the Interquartile Range method to trim extreme transit anomalies like negative days or values over 60 days.
  * Applied Min-Max Scaling to compress numeric features into a 0 to 1 range for spatial algorithms.
* **Folder Files:**
  * `Task_2/Task_2_Preprocessing_Report.docx`: Detailed document covering data flaws, cleaning strategies and decision impact.
  * `Task_2/src/data_preprocessing.py`: Python script implementing missing value imputation, IQR outlier filtering and Min-Max scaling.

### Task 3: Exploratory Data Analysis and Visualization
* **Objective:** Analyze core metrics within the cleaned dataset using Python to spot shipping bottlenecks and evaluate cost drivers.
* **Key Findings:**
  * Average actual transit duration of 3.48 days consistently exceeds promised scheduled targets of 2.93 days, causing delays in over 42% of shipments.
  * Scatter plot analysis revealed high-revenue orders generating negative net profits due to expedited freight fees on delayed shipments.
  * Stacked bar charts showed that First Class express shipping had the worst relative delay rate percentage compared to other modes.
* **Visualizations:** Histogram of delivery delay variance, scatter plot of sales vs profit margins and a stacked bar chart of delivery status across shipping modes.
* **Folder Files:**
  * `Task_3/Task_3_EDA_Visualization_Report.docx`: Analytical report with summary tables, chart interpretations and supply chain recommendations.
  * `Task_3/src/eda_visualization.py`: Python script automating statistical summaries and Matplotlib plot generation.

### Task 4: Predictive Analytics and System Optimization
* **Objective:** Forecast actual shipment transit durations in days, evaluate model parameters and propose actionable logistics optimizations.
* **Methodology:** Trained baseline linear models alongside decision tree models using an 80/20 train-test split, 5-fold cross validation and parameter tuning.
* **Model Results:** The decision tree model achieved a Mean Absolute Error of 0.41 days, a Root Mean Squared Error of 0.63 days and an R-squared score of 0.76.
* **Optimization Recommendations:**
  * Add a dynamic 1-day safety buffer to promised dates when predicted delay risk is high.
  * Assign priority dispatch slots and faster delivery trucks to high-volume long transit routes.
  * Re-route orders with severe delay risks to secondary regional hubs closer to destination zip codes.
* **Folder Files:**
  * `Task_4/Task_4_Predictive_Modeling_Report.docx`: Final project report covering model selection, evaluation metrics, tuning and optimization plans.
  * `Task_4/src/predictive_optimization.py`: Python script handling model training, cross-validation, parameter tuning and evaluation plots.

---

## Tech Stack and Tools
* **Programming Language:** Python
* **Data Processing and Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib

* **Documentation:** Microsoft Word, Markdown

---
 Completed as part of the Virtual Logistics Data Analytics Internship.
