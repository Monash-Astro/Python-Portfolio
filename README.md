# Python-Portfolio
My code portfolio

###########  Best_Fit_Line.py  ###########

This code is used to construct a best fit line for data points with arbitrary two dimensional uncertainties. 
Also uses mixture models and posterior probabilities to identify outliers, plottimng best fit draws for points 
coloured by posterior probability of being outliers.

To run:
Download python file.
Install necessary modules using command such as:
  - pip install numpy matplotlib scipy emcee corner

Run python file, i.e
  >>> Import Best_Fit_Line


###########  Expectation_maximisation.py  ###########

This code generates a set of randomly clustered points. Using an expectation maximisation algorithm to increase 
the logliklihood that a random set of initial points corresponds to the provided data points. The log liklihood
as well as change in logliklihood are plotted as a function of iterations. The best fit points are also plotted 
with three degrees of uncertainty at each iteration.

To run:
Install necessary modules using command such as:
  - pip install pandas numpy matplotlib scikit-learn

Run python file, i.e
  >>> Import Expectation_maximisation


###########  Gaussian_model_prediction.py  ###########

This code uses recorded data for the brightness of a star over a period of three months. I use a lombscargle 
periodogram to identify the maximal frequencies present in the data. I then use Gaussian processes to
predict the future brightness of the star for another two month period. I use the kernels
  -  Mattern 32
  -  Exponential Sine
  -  Rational Quadratic (Optional: reduces misrepresentation of data)
I then plot the mean expected brightness for the next two months,, as well as 2 standard deviations from the mean
path. Finally I plot the draws sampled using emcee, predicting the brightness for the next two months.

To run:

!!! NOTE THIS WILL TAKE 10 - 30 MINUTES TO RUN !!!

Install necessary modules using command such as:
  - pip install numpy matplotlib astropy george scipy emcee

Run python file, i.e
  >>> Import Gaussian_model_prediction






