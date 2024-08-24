import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

dataset_path = 'opsd_germany_daily.csv'
WIND_SOLAR = 'Wind+Solar'

if __name__ == "__main__":
    # Read data from .csv file
    opsd_daily = pd.read_csv(dataset_path)

    print("shape = ", opsd_daily.shape)
    print("dtype = ", opsd_daily.dtypes)
    print(opsd_daily.head())

    opsd_daily = opsd_daily.set_index('Date')
    print(opsd_daily.head())

    opsd_daily = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    opsd_daily['Year'] = opsd_daily.index.year
    opsd_daily['Month'] = opsd_daily.index.month
    opsd_daily['Weekday Name'] = opsd_daily.index.day_name()
    print(opsd_daily.sample(5, random_state=0))

    print(opsd_daily.loc['2014-01-20': '2014-01-22'])

    sns.set_theme(rc={'figure.figsize': (11, 4)})
    opsd_daily['Consumption'].plot()

    cols_plot = ['Consumption', 'Solar', 'Wind']
    axes = opsd_daily[cols_plot].plot(
        subplots=True, figsize=(11, 9), grid=True)

    for ax in axes:
        ax.set_ylabel('Daily total (GWh)')

    # plt.show()

    fig, axes = plt.subplots(3, 1, sharex=True)
    for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
        sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
        ax.set_title(name)

        if ax != axes[-1]:
            ax.set_xlabel('')

    # plt.show()

    pd.date_range('1998 -03 -10', '1998 -03 -15 ', freq='D')
    times_sample = pd.to_datetime(
        ['2013 -02 -03', '2013 -02 -06', '2013 -02 -08'])
    consum_sample = opsd_daily.loc[times_sample, ['Consumption']].copy()
    print(consum_sample)

    data_columns = ['Consumption', 'Wind', 'Solar', WIND_SOLAR]
    opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()
    opsd_weekly_mean.head(3)

    # Virtualize daily and weekly time series in 6 month after resampling
    start, end = '2017-01', '2017-06'
    fig, ax = plt.subplots()
    ax.plot(opsd_daily.loc[start:end, 'Solar'])
    ax.plot(opsd_weekly_mean.loc[start:end])
    # plt.show()

    opsd_annual = opsd_daily[data_columns].resample('YE').sum(min_count=360)
    opsd_annual = opsd_annual.set_index(opsd_annual.index.year)
    opsd_annual.index.name = 'Year'

    opsd_annual['Wind+Solar/Consumption'] = opsd_annual[WIND_SOLAR] / \
        opsd_annual['Consumption']
    print(opsd_annual.tail(3))

    opsd_7d = opsd_daily[data_columns].rolling(7, center=True).mean()
    opsd_365d = opsd_daily[data_columns].rolling(
        window=365, center=True, min_periods=360).mean()

    fig, ax = plt.subplots()
    ax.plot(opsd_daily['Consumption'], marker='.', markersize=2,
            color='0.6', linestyle='None', label='Daily')
    ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
    ax.plot(opsd_365d['Consumption'], color='0.2',
            linewidth=3, label='365-d Rolling Mean')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlabel('Year')
    ax.set_ylabel('Consumption')
    ax.set_title('Trends in Electricity')
    # plt.show()

    for nm in ['Wind', 'Solar', WIND_SOLAR]:
        ax.plot(opsd_365d[nm], label=nm)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_ylim(0, 400)
        ax.legend()
        ax.set_ylabel('Production(GWh)')
        ax.set_title('Trends in Electricity Production(365-d Rolling Means)')
    plt.show()
