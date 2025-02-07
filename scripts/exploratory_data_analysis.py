# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the engineered dataset
file_path = 'engineered_daily_cp_activity.csv'
data = pd.read_csv(file_path)

# Preview the dataset
print("First 5 Rows of the Dataset:")
print(data.head())

# 1. Distribution of Courier Partners Online
plt.figure(figsize=(8, 6))
sns.histplot(data['courier_partners_online'], bins=20, kde=True)
plt.title("Distribution of Courier Partners Online")
plt.xlabel("Number of Courier Partners Online")
plt.ylabel("Frequency")
plt.show()

# 2. Average Courier Activity by Day of the Week
plt.figure(figsize=(8, 6))
sns.barplot(x='day_of_week', y='courier_partners_online', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title("Average Courier Activity by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Average Courier Partners Online")
plt.xticks(rotation=45)
plt.show()

# 3. Courier Activity by Season
plt.figure(figsize=(8, 6))
sns.boxplot(x='season', y='courier_partners_online', data=data, order=['Spring', 'Summer', 'Autumn', 'Winter'])
plt.title("Courier Activity by Season")
plt.xlabel("Season")
plt.ylabel("Courier Partners Online")
plt.show()

# 4. Impact of Rain Intensity on Courier Activity
plt.figure(figsize=(8, 6))
sns.barplot(x='rain_intensity', y='courier_partners_online', data=data, order=['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain'])
plt.title("Impact of Rain Intensity on Courier Activity")
plt.xlabel("Rain Intensity")
plt.ylabel("Average Courier Partners Online")
plt.show()

# 5. Courier Activity by Temperature Category
plt.figure(figsize=(8, 6))
sns.boxplot(x='temperature_category', y='courier_partners_online', data=data, order=['Cold', 'Moderate', 'Hot'])
plt.title("Courier Activity by Temperature Category")
plt.xlabel("Temperature Category")
plt.ylabel("Courier Partners Online")
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data[['courier_partners_online', 'temperature', 'relative_humidity', 'precipitation']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# 7. Courier Activity Over Time
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['courier_partners_online'], label='Courier Partners Online', alpha=0.7)
plt.title("Courier Activity Over Time")
plt.xlabel("Date")
plt.ylabel("Courier Partners Online")
plt.xticks(rotation=45)
plt.legend()
plt.show()
