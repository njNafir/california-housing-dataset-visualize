import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

housing_url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
df = pd.read_csv(housing_url)
df.to_csv("housing.csv")

print(df.head())
print(df.describe())

# Histogram of Housing Prices

plt.hist(df['median_house_value'], bins=30)
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Distribution of Housing Prices')
plt.savefig('housing_prices.png')
plt.close()

# Scatter Plot of Longitude and Latitude

plt.scatter(df['longitude'], df['latitude'], alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Distribution of Housing')
plt.savefig('longitude_latitude.png')
plt.close()

# Box Plot of Median Income by Ocean Proximity

sns.boxplot(x='ocean_proximity', y='median_income', data=df)
plt.xlabel('Ocean Proximity')
plt.ylabel('Median Income')
plt.title('Median Income by Ocean Proximity')
plt.savefig('median_income_by_ocean_proximity.png')
plt.close()

# Bar Plot of Average Rooms by Ocean Proximity

avg_rooms = df.groupby('ocean_proximity')['total_rooms'].mean()
avg_rooms.plot(kind='bar')
plt.xlabel('Ocean Proximity')
plt.ylabel('Average Rooms')
plt.title('Average Rooms by Ocean Proximity')
plt.savefig('average_rooms.png')
plt.close()

# Violin Plot of Bedrooms and Population

sns.violinplot(x='total_bedrooms', y='population', data=df)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Population')
plt.title('Distribution of Bedrooms and Population')
plt.savefig('bedrooms_and_population.png')
plt.close()

# Stacked Bar Chart of Ocean Proximity and House Type

ocean_house_counts = df.groupby(['ocean_proximity', 'ocean_proximity'])['ocean_proximity'].size().unstack()
ocean_house_counts.plot(kind='bar', stacked=True)
plt.xlabel('Ocean Proximity')
plt.ylabel('Count')
plt.title('Distribution of House Types by Ocean Proximity')
plt.savefig('house_types.png')
plt.close()

# Pie Chart of House Types

house_type_counts = df['ocean_proximity'].value_counts()
plt.pie(house_type_counts, labels=house_type_counts.index, autopct='%1.1f%%')
plt.title('Proportion of House Types')
plt.savefig('house_types_pie.png')
plt.close()

# Pair Plot of Select Features

features = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
sns.pairplot(df[features])
plt.savefig('pair_plot_of_features.png')
plt.close()
