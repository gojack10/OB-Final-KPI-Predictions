import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Baseline data for 2015-2024 (includes hypotheticals as was instructed in assingment)
years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
student_satisfaction = np.array([90, 88, 87, 85, 83, 82, 78, 80, 75, 72])  # Variable trend
faculty_satisfaction = np.array([85, 83, 81, 80, 77, 75, 72, 74, 70, 68])  # Variable trend
graduation_rates = np.array([70, 68, 67, 65, 62, 64, 61, 63, 60, 58])  # Variable trend
student_employment_rates = np.array([90, 88, 86, 85, 82, 80, 76, 78, 75, 72])  # Variable trend

# Polynomial regression model
poly = PolynomialFeatures(degree=2)

years_poly = poly.fit_transform(years)

model_student_satisfaction = LinearRegression()
model_student_satisfaction.fit(years_poly, student_satisfaction)

model_faculty_satisfaction = LinearRegression()
model_faculty_satisfaction.fit(years_poly, faculty_satisfaction)

model_graduation_rates = LinearRegression()
model_graduation_rates.fit(years_poly, graduation_rates)

model_student_employment_rates = LinearRegression()
model_student_employment_rates.fit(years_poly, student_employment_rates)

# Predictions for 2025-2030
future_years = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
future_years_poly = poly.transform(future_years)

pred_student_satisfaction = model_student_satisfaction.predict(future_years_poly)
pred_faculty_satisfaction = model_faculty_satisfaction.predict(future_years_poly)
pred_graduation_rates = model_graduation_rates.predict(future_years_poly)
pred_student_employment_rates = model_student_employment_rates.predict(future_years_poly)

# Plotting the predictions
plt.figure(figsize=(14, 8))

plt.subplot(221)
plt.plot(years, student_satisfaction, 'bo-', label='Actual')
plt.plot(future_years, pred_student_satisfaction, 'r--', label='Predicted')
plt.xticks(np.arange(2015, 2031, 1), fontsize=8, rotation=45)
plt.title('Student Satisfaction')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.ylim(40, 100)
plt.legend()

plt.subplot(222)
plt.plot(years, faculty_satisfaction, 'bo-', label='Actual')
plt.plot(future_years, pred_faculty_satisfaction, 'r--', label='Predicted')
plt.xticks(np.arange(2015, 2031, 1), fontsize=8, rotation=45)
plt.title('Faculty Satisfaction')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.ylim(40, 100)
plt.legend()

plt.subplot(223)
plt.plot(years, graduation_rates, 'bo-', label='Actual')
plt.plot(future_years, pred_graduation_rates, 'r--', label='Predicted')
plt.xticks(np.arange(2015, 2031, 1), fontsize=8, rotation=45)
plt.title('Graduation Rates')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.ylim(40, 100)
plt.legend()

plt.subplot(224)
plt.plot(years, student_employment_rates, 'bo-', label='Actual')
plt.plot(future_years, pred_student_employment_rates, 'r--', label='Predicted')
plt.xticks(np.arange(2015, 2031, 1), fontsize=8, rotation=45)
plt.title('Student Employment Rates')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.ylim(40, 100)
plt.legend()

plt.tight_layout()
plt.savefig('kpi_predictions.png')
plt.show()
