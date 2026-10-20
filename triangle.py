import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

class Triangle:
    '''
    A class to analyze a cumulative claims triangle using Chain Ladder development factors to
    calculate expected reserves.

    Also supports Bornhuetter-Ferguson reserve calculation, calculates claim development patterns,
    and handles the associated visuals for all calculated values.

    Parameters
    ----------
    data : pd.DataFrame
        Cumulative claims triangle data.
        Accident periods should be rows and development periods should be columns.
        Input claims are assumed to be cumulative, and development period columns must be numbered
        sequentially starting at 0.

    Attributes
    ----------
    data : pd.DataFrame
        Stores the untouched raw data given in the initial data parameter.

    num_dev_periods : int
        Stores the number of development periods given in the data.

    num_acc_periods : int
        Stores the number of accident periods given in the data.

    dev_factors : np.ndarray
        The array of Chain Ladder development factors for the given data.

    prediction : pd.DataFrame
        Completed claims triangle data.
        Fill in null-valued entries in the original data using Chain Ladder development factors.

    last_observations : pd.Series
        Stores the last non-null value in the raw data for each accident period.
        Effectively, the most recently observed cumulative claims amount for each accident period.

    ultimate_claims : pd.Series
        Stores the predicted ultimate cumulative claims amount for each accident period.
        Note that the final development period is treated as ultimate.
    '''
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.num_acc_periods, self.num_dev_periods = self.data.shape
        self.dev_factors = np.zeros(self.num_dev_periods - 1)
        self.prediction = self.data.copy()
        self.last_observations = data.ffill(axis = 1).iloc[:, -1]

        for i in range(self.num_dev_periods - 1):
            # Calculates the age-to-age development factors using the observed data only.
            valid_values = self.data[self.data[i+1].notna() & self.data[i].notna()]
            numerator = valid_values[i + 1].sum()
            denominator = valid_values[i].sum()
            if denominator == 0:
                raise ValueError(f'Cannot calculate development factor {i}: sum of claims at development period {i} is zero.')
            self.dev_factors[i] = numerator/denominator


            # Fills in missing values in the prediction data using the development factors above.
            missing_values = self.prediction[i + 1].isna()
            self.prediction.loc[missing_values, i + 1] = self.prediction.loc[missing_values, i]*self.dev_factors[i]
            
        self.ultimate_claims = self.prediction.iloc[:, -1]

    def reserves(self) -> pd.Series:
        '''Returns the Chain Ladder reserve estimates for each accident period.'''
        return self.ultimate_claims - self.last_observations

    def BF_reserves(self, expectation: np.ndarray) -> pd.Series:
        '''
        Calculates the Bornhuetter-Ferguson reserve estimates for each accident period.

        Parameters
        ----------
        expectation : np.ndarray
            The prior estimates of ultimate claims for each accident period

        Returns
        -------
        pd.Series
            The Bornhuetter-Ferguson reserve estimates for each accident period, based on the given expectation array.
        '''
        if len(expectation) != self.num_acc_periods:
            raise ValueError(f'Invalid array length for Bornhuetter-Ferguson Reserve. Expected length of {self.num_acc_periods}, got {len(expectation)}.')
        return expectation * (1 - self.last_observations/self.ultimate_claims)

    def cumulative_percent_developed(self) -> np.ndarray:
        '''
        Calculates and returns the cumulative percentage of total claims expected to develop by each development period,
        for any arbitrary accident period, based on the calculated Chain Ladder development factors.
        '''
        cumulative_devs = np.ones(self.num_dev_periods)
        i = -2
        for fac in self.dev_factors[::-1]:
            # Work backward from 100% developed to calculate the cumulative proportion developed
            # at each preceding development age.
            cumulative_devs[i] = cumulative_devs[i + 1] / fac
            i -= 1
        return cumulative_devs

    def plot_raw_data(self, filename: str | None = None) -> None:
        '''
        Plots the initial observation data as a heatmap.

        Parameters
        ----------
        filename : str | None
            The filename to save the plot to. A .png or .jpg extension is recommended.
            If None, the plot will instead be displayed on screen rather than saved to a file.
        '''
        rows, columns = self.data.shape
        plt.figure(figsize = (columns*1.4, rows*0.8))
        sns.heatmap(
            data = self.data,
            annot = True,
            fmt = '.0f'
            )
        plt.title('Raw Data')
        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()

    def plot_predicted_data(self, filename: str | None = None) -> None:
        '''
        Plots the predicted data calculated using Chain Ladder development factors as a heatmap.

        Parameters
        ----------
        filename : str | None
            The filename to save the plot to. A .png or .jpg extension is recommended.
            If None, the plot will instead be displayed on screen rather than saved to a file.
        '''
        rows, columns = self.prediction.shape
        plt.figure(figsize = (columns*1.4, rows*0.8))
        sns.heatmap(
            data = self.prediction,
            annot = True,
            fmt = '.0f'
            )
        plt.title('Predicted Data')
        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()

    def plot_dev_factors(self, filename: str | None = None) -> None:
        '''
        Plots the Chain Ladder development factors as a bar graph.

        Parameters
        ----------
        filename : str | None
            The filename to save the plot to. A .png or .jpg extension is recommended.
            If None, the plot will instead be displayed on screen rather than saved to a file.
        '''
        plt.figure(figsize = ((self.num_dev_periods - 1)*1.4, 5))
        bar = plt.bar(np.arange(self.num_dev_periods - 1), self.dev_factors, width = 0.6, color = 'blue')
        plt.title('Development Factors')
        plt.xticks(np.arange(self.num_dev_periods - 1))
        plt.xlabel('Development Period')
        plt.ylabel('Development Factor')
        plt.bar_label(bar, padding=3, fmt = '%.2f')
        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()

    def plot_development_percent(self, filename: str | None = None) -> None:
        '''
        Plots the cumulative percentage data as calculated by the cumulative_percent_developed method as a bar graph.
        Also calculates and plots the corresponding incremental percentage of total claims expected to develop in each
        development period on the same plot.

        Parameters
        ----------
        filename : str | None
            The filename to save the plot to. A .png or .jpg extension is recommended.
            If None, the plot will instead be displayed on screen rather than saved to a file.
        '''
        cumulative = self.cumulative_percent_developed()
        incremental = cumulative.copy()
        incremental[1:] = np.diff(cumulative)
        
        plt.figure(figsize = (self.num_dev_periods*1.4, 5))
        cumul_bar = plt.bar(np.arange(self.num_dev_periods) + 0.3, cumulative*100, width = 0.3, color = 'green', label = 'Cumulative Percent Developed')
        incr_bar = plt.bar(np.arange(self.num_dev_periods), incremental*100, width = 0.3, color = 'blue', label = 'Incremental Percent Developed')
        plt.title('Development Percent')
        plt.xticks(np.arange(self.num_dev_periods))
        plt.xlabel('Development Period')
        plt.ylabel('Percent')
        plt.bar_label(cumul_bar, padding=3, fmt = '%.0f%%')
        plt.bar_label(incr_bar, padding=3, fmt = '%.0f%%')
        plt.legend()
        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()

    def plot_reserves(self, BF_expectation: np.ndarray, filename: str | None = None) -> None:
        '''
        Plots the reserve estimates for both the Chain Ladder method and the Bornhuetter-Ferguson method for each accident
        period as a bar graph.

        Parameters
        ----------
        BF_expectation : np.ndarray
            The prior estimates of ultimate claims for each accident period used to calculate the Bornhuetter-Ferguson method.
            Refer to the BF_reserves method; this parameter is fed directly as a parameter into that method.
        
        filename : str | None
            The filename to save the plot to. A .png or .jpg extension is recommended.
            If None, the plot will instead be displayed on screen rather than saved to a file.
        '''
        plt.subplots(figsize = (self.num_acc_periods*1.4, 5))
        res_bars = plt.bar(np.arange(self.num_acc_periods), self.reserves(), width = 0.3, color = 'red', label = 'Chain Ladder')
        BF_res_bars = plt.bar(np.arange(self.num_acc_periods) + 0.4, self.BF_reserves(BF_expectation), width = 0.3, color = 'purple', label = 'Bornhuetter-Ferguson')
        plt.title('Reserves')
        plt.xticks(np.arange(self.num_acc_periods))
        plt.xlabel('Accident Period')
        plt.ylabel('Reserve')
        plt.bar_label(res_bars, padding=3, fmt = '%.0f')
        plt.bar_label(BF_res_bars, padding=3, fmt = '%.0f')
        plt.legend()
        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()

    def plot_all(self, BF_expectation: np.ndarray, filepath: str | None = None, overwrite: bool = False) -> None:
        '''
        Creates all the plots in the previous plot methods all at once. Automatically picks file names for the plots
        if a filepath is given, rather than having the user create names for each plot individually.

        Parameters
        ----------
        BF_expectation : np.ndarray
            The prior estimates of ultimate claims for each accident period used to calculate the Bornhuetter-Ferguson method.
            Refer to the BF_reserves method; this parameter is fed directly as a parameter into that method.
        
        filepath : str | None
            The directory to which all plots will be saved. Note that if overwrite (below) is set to False, the actual directory
            may not exactly match this parameter. If None, the plots will instead be displayed on screen rather than saved.

        overwrite : bool
            If False and the specified directory already exists, a number is appended to create a new directory. This ensures that
            no files are overwritten when creating the plots.

            If True, the method will write to the given directory even if it already exists. If any existing files with the same
            name as the plots already exists, the plots will permanently overwrite them. Use with caution.

            Note that if filepath is None, this parameter has no effect.
        '''
        if filepath != None:
            if os.path.isdir(filepath) and not overwrite:
                addon = 1
                while os.path.isdir(filepath + ' ' + str(addon)):
                    addon += 1
                filepath = filepath + ' ' + str(addon)
            if not os.path.isdir(filepath):
                os.makedirs(filepath)
        raw_data_fn = None
        predicted_fn = None
        dev_factors_fn = None
        perc_dev_fn = None
        reserves_fn = None
        if filepath != None:
            raw_data_fn = os.path.join(filepath, 'raw_data.png')
            predicted_fn = os.path.join(filepath, 'prediction.png')
            dev_factors_fn = os.path.join(filepath, 'development_factors.png')
            perc_dev_fn = os.path.join(filepath, 'development_percent.png')
            reserves_fn = os.path.join(filepath, 'reserves.png')
        self.plot_raw_data(raw_data_fn)
        self.plot_predicted_data(predicted_fn)
        self.plot_dev_factors(dev_factors_fn)
        self.plot_development_percent(perc_dev_fn)
        self.plot_reserves(BF_expectation, reserves_fn)
