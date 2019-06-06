# AI-Project-4

## How to Compile
Type "make" in the directory containing the makefile to compile the project wit no debug output enabled.

You can enable debug output by typing `make dbg`.

## How to Run
kmeans.exe takes five command-line arguments:
1. Integer seed for random numbers
2. The number of clusters used in classification
3. The number of features in each feature vector of a dataset
4. Text file containing training data
5. Text file containing testing data

Each feature vector in the iris-data dataset contains four attributes.  Here's an example run of the program: `./kmeans 0 7 4 training.txt testing.txt"

## Output
kmeans.exe outputs the number of testing vectors it correctly identified.

## Interpreting the Results
Refer to the report PDF.
