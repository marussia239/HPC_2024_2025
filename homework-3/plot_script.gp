#Note: in order to plot, include x-values in the first column, and the corresponding y-values in the following columns in a csv file.

# Define variables
datafile = 'res_ref.csv'
outputfile = 'plot.png'
title_main = 'Execution Time as a Function of Data Size'
x_label = 'Data Size (log scale)'
y_label = 'Execution time (s)'

# Set terminal and output
set terminal pngcairo size 800,600 enhanced background rgb 'white'
set output outputfile

# Main title and subtitle
set title title_main

# Axes labels
set xlabel x_label
set ylabel y_label

# Set x-axis to logarithmic scale
set logscale x

# Set grid for better readability
set grid

# Place the key outside the plot for better use of space
set key outside right

# Plot each series using the variable for the data file
plot datafile using 1:2 title 'Sequential' with linespoints, \
     '' using 1:3 title '1 thread' with linespoints, \
     '' using 1:4 title '2 threads' with linespoints, \
     '' using 1:5 title '5 threads' with linespoints, \
     '' using 1:6 title '10 threads' with linespoints, \
     '' using 1:7 title '20 threads' with linespoints, \
     '' using 1:8 title '40 threads' with linespoints, \
     '' using 1:9 title '60 threads' with linespoints, \
     '' using 1:10 title '100 threads' with linespoints