import pandas as pd
import matplotlib.pyplot as plt
import os

def preprocess_data(df):
    """
    Preprocesses the DataFrame by dropping unnecessary columns, replacing null values with row-wise means.

    Parameters:
    df (DataFrame): Input DataFrame containing numeric data.

    Returns:
    DataFrame: Processed DataFrame with null values replaced by row-wise means.
    """
    # Drop unnecessary columns
    df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
    
    # Calculate row-wise mean ignoring NaN values
    row_means = df.iloc[:, 1:].mean(axis=1, skipna=True)
    
    # Iterate over each row and fill NaN values with corresponding row mean
    for idx, row in df.iterrows():
        for col in df.columns[1:]:
            if pd.isnull(row[col]):
                df.at[idx, col] = row_means[idx]
    
    return df

def plot_line_chart(df, indicator_name, countries, start_year, end_year,):
    """
    Plots a line chart for given indicator and countries over a specific time period and saves the plot.

    Parameters:
    df (DataFrame): Input DataFrame containing indicator data.
    indicator_name (str): Name of the indicator column.
    countries (list of str): List of country names to plot.
    start_year (int): Start year for the plot (inclusive).
    end_year (int): End year for the plot (inclusive).
    

    Returns:
    None
    """
    # Filter columns for the specified year range
    columns_to_plot = ['Country Name'] + [str(year) for year in range(start_year, end_year + 1)]
    df_filtered = df[columns_to_plot]

    years = columns_to_plot[1:]  # Exclude 'Country Name' for the x-axis

    plt.figure(figsize=(10, 6))

    for country in countries:
        data = df_filtered[df_filtered['Country Name'] == country].iloc[:, 1:].values.flatten()
        plt.plot(years, data,  label=country)

    plt.title(f'{indicator_name} from {start_year} to {end_year}')
    plt.xlabel('Year')
    plt.ylabel(indicator_name)
    plt.legend()
    plt.grid(True)
    plt.xticks(years[::2])  # Adjust the x-ticks for better readability
    plt.tight_layout()

    plt.savefig('linechart.png', dpi=dpi)
    plt.close()  # Close the figure to free memory

def plot_bar_chart(df, indicator_name, year, countries):
    """
    Plots a bar chart for given indicator and a specific year for selected countries and saves the plot.

    Parameters:
    df (DataFrame): Input DataFrame containing indicator data.
    indicator_name (str): Name of the indicator column.
    year (int): Year for which data is plotted.
    countries (list of str): List of country names to include in the plot.
    output_dir (str): Directory to save the plots (default is 'plots').
    dpi (int): Dots per inch for the output resolution (default is 300).

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    data = df.loc[df['Country Name'].isin(countries), ['Country Name', str(year)]].dropna()
    plt.bar(data['Country Name'], data[str(year)])
    
    plt.title(f'{indicator_name} in {year}')
    plt.xlabel('Country')
    plt.ylabel(indicator_name)
    plt.xticks(rotation=90)
    plt.tight_layout()

    
    plt.savefig('barchart.png', dpi=300)
    plt.close()  # Close the figure to free memory

def plot_histogram(df, countries, output_dir='plots', dpi=300):
    """
    Plots histograms of life expectancy for the given countries and saves them.

    Parameters:
    df (DataFrame): Input DataFrame containing life expectancy data.
    countries (list): List of countries to plot histograms for.
    output_dir (str): Directory to save the plots (default is 'plots').
    dpi (int): Dots per inch for the output resolution (default is 300).

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    
    # Loop through each country
    for country in countries:
        country_data = df[df['Country Name'] == country].iloc[:, 1:]  # Exclude 'Country Name' column for plotting
        # Stack all years into a single series for plotting histogram
        life_expectancy_values = country_data.stack().reset_index(drop=True)
        
        plt.hist(life_expectancy_values.dropna(), bins=20, edgecolor='black', alpha=0.7, label=country)
    
    plt.title('Histogram of Life Expectancy')
    plt.xlabel('Life Expectancy (years)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    
    plt.savefig('histogram.png', dpi=dpi)
    plt.close()  # Close the figure to free memory


# Main program
if __name__ == "__main__":
    #  dataset file paths
    gdp_file = 'GDP per capita.csv'
    life_expectancy_file = 'Life expectancy at birth.csv'
    co2_emissions_file = 'CO2 emissions.csv'
    
    # Load the datasets
    gdp = pd.read_csv(gdp_file, skiprows=4, usecols=lambda x: x.strip() != "Unnamed: 68")
    life_expectancy = pd.read_csv(life_expectancy_file, skiprows=4, usecols=lambda x: x.strip() != "Unnamed: 68")
    co2_emissions = pd.read_csv(co2_emissions_file, skiprows=4, usecols=lambda x: x.strip() != "Unnamed: 68")
    
    # Preprocess data (handling null values)
    gdp = preprocess_data(gdp)
    life_expectancy = preprocess_data(life_expectancy)
    co2_emissions = preprocess_data(co2_emissions)
    
    # Example plots
    countries_to_plot = ['United States', 'China', 'India', 'Germany']
    year_to_plot = 2020

    # Plot line chart for GDP per capita growth from 2000 to 2020
    plot_line_chart(gdp, 'GDP per Capita Growth (%)', countries_to_plot, start_year=2000, end_year=2022)

    # Plot bar chart for life expectancy in a specific year for selected countries
    plot_bar_chart(co2_emissions, 'CO2 Emission', year_to_plot, countries_to_plot)

    # Plot histograms for life expectancy and save them
    plot_histogram(life_expectancy, countries_to_plot)
