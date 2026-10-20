# Bornhuetter-Ferguson Reserving Method

This program is a Python implementation of the Bornhuetter-Ferguson method to estimate reserves using cumulative claims triangles.

The project calculates development factors using the Chain Ladder technique, and then uses these to estimate a completed claims triangle and reserves for each accident period. Then, when given a prior estimate of ultimate claims, it computes the Bornhuetter-Ferguson reserves for each period. Finally, it analyzes the claims patterns predicted, and creates visuals to help quantify them.

## Features
- Automatically creates the Chain Ladder development factors for each development period, as well as the corresponding predicted data.
- Calculates the Chain Ladder reserves estimate.
- Given a list of expected ultimate claims, calculates the Bornhuetter-Ferguson reserves estimate for that list of expected reserves.
- Determines the cumulative percentage of total claims that are expected to develop by a specific development period.
- Creates plots for:
	- Observed triangle data (heatmap)
	- Predicted claims data (heatmap)
	- Development factors for each period (bar graph)
	- Cumulative and incremental percentages of the total claims predicted to emerge by the development factors (bar graph)
	- Reserves for each accident period (bar graph)
- Includes methods to create each plot individually, and a master method to plot everything at once.
- Options when plotting to save the plots to a given file location, or show them on screen without saving.


## Methodology

### Chain Ladder
The Chain ladder method predicts development factors by assuming they are uniform across accident periods. With this assumption in place, the development factor for development period j can be calculated as the ratio of claims at development period j+1 to the claims at development period j, excluding any claims occuring on accident periods where the claims data for either development period is not present.

For a given set of cumulative triangle claims data, and assuming the data in question does not have any missing values in the upper triangle, the development factor for period j then becomes precisely:

f_j = Σ C_(i, j + 1) / Σ C_(i, j), with the sum ranging from i = 0 to n - j - 1.

Where C_(i, j) denotes the claims data for accident period i and development period j, and n denotes the total number of accident periods.

To accommodate for triangle data with missing observations, the implementation instead includes only accident periods for which both C_(i, j) and C_(i, j + 1) are observed. Projected values are not used when estimating development factors.

After development factors are calculated, predictions for the unknown data are calculated recursively by multiplying the previous value in that development cycle by the corresponding development factor, or equivalently:

C_(i, j) ~= C_(i, j - 1) * f_(j - 1)

The ultimate claims for accident period i are then taken to be:

C_(i, inf) = C_(i, m - 1)

Where m is the total number of development periods.

This allows computation of the total reserve. For a fully intact traingle, it is computed as:

R_i = C_(i, inf) - C_(i, n - i))

To incorporate missing data, C_(i, n - i) is replaced instead with K_i, the most recent observation of claims data for accident period i:

R_i = C_(i, inf) - K_i

The value of R_i then represents the total expected value of claims yet to develop in accident period i based on the observed data.

### Bornhuetter-Ferguson

The Bornhuetter-Ferguson method expands on the Chain Ladder method by introducing a prior expectation of what the ultimate claims should be. From there, a new reserve is calculated, using the formula:

R_i = E_i * (1 - K_i / C_(i, inf))

Where E_i is the expected ultimate claims value for accident period i, K_i represent the last observed value of claims for accident period i, and C_(i, inf) represents the ultimate claims estimated for accident period i as computed by the Chain Ladder method.

In this way, the Bornhuetter-Ferguson method applies the prior ultimate expectation to the claims yet to be developed only. Thus as more of the claims are observed, the reserves are calculated more and more based on the observed data and less on the prior estimate.

### Development Analysis

After reserves and development factors are calculated, the program computes the cumulative percentage of total claims expected to develop by each development period. These are calculated recursively using the formula:

P_(m - 1) = 1, where m is the total number of development periods
P_(i - 1) = P_i/f_(i - 1) for i = 1 through m - 2.

The corresponding incremental percentages are also calculated. Given the cumulative percentages above, this becomes simply:

IP_(i) = P_i - P_(i - 1)

## Example Output
<img width="1400" height="800" alt="prediction" src="https://github.com/user-attachments/assets/f4bd28ba-0db6-410e-9ae0-c6ef7d379914" />

<img width="1400" height="500" alt="development_percent" src="https://github.com/user-attachments/assets/39d0bd19-709f-4fc8-b1e9-723078d2d9d3" />

<img width="1400" height="500" alt="reserves" src="https://github.com/user-attachments/assets/e690da86-c4df-4c4e-8e71-fb5285d8a9ac" />

## Input Data

The Triangle class included expects a pd.DataFrame object as input, representing a claims triangle. The data should be organized with one axis being accident periods, and the other development periods.

Included is a block of code that will convert a CSV file into the appropriate pd.DataFrame object the class can read. By default, the given example data 'triangle_data.csv' is used as a filename, but it should work with any data, so long as the CSV is formatted with columns 'development' and 'origin', representing the development years and accident years, respectively. The script then converts these fields into accident and development periods, then pivots the data into a claims triangle.

## Assumptions and Limitations

- Input claims are assumed to be cumulative
- Development periods and accident periods must be sequentially numbered beginning at 0.
- Observed development patterns are assumed to be representative of future patterns.
- In this model, the final development pattern is treated as ultimate.
- The claims triangle need not be complete - missing values are supported, in which case development factors will be estimated only using observed claim pairs. Projected data for missing values will never be used to predict other missing values.
- Since they rely on a prior computed estimate of ultimate claims, Bornhuetter-Ferguson results depend on the quality of the supplied estimates.
- Reserve estimates are calculated as a single point number, not a range. They do not take into account any reserve uncertainty.

## Usage Example

```python
filename = 'triangle_data.csv'
data = pd.read_csv(filename)

data['development_period'] = data['development'] - data['origin']
data['accident_period'] = data['origin'] - data['origin'].min()

triangle_data = data.pivot(
	index = 'accident_period',
	columns = 'development_period',
	values = 'values'
)

triangle = Triangle(triangle_data)

prior_estimates = np.ones(triangle.num_acc_periods) * 10000

triangle.plot_all(
	prior_estimates,
	filepath = 'Output',
	overwrite = False
)
