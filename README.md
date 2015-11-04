PyIBP
==========

PyIBP is an Indian Buffet Process package, developed by the Cloud Computing Research Team in [University of Maryland, College Park] (http://www.umd.edu).

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyIBP).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy, matplotlib and nltk.

Launch and Execute
----------

Assume the PyIBP package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PyIBP

To prepare the example dataset,

	tar zxvf cambridge-bars.tar.gz

To launch PyIBP, first redirect to the directory of PyIBP source code,

	cd $PROJECT_SPACE/src/PyIBP

and run the following command on example dataset,

	python -m launch_train --input_directory=./cambridge-bars --output_directory=./ --training_iterations=100

The generic argument to run PyIBP is

	python -m launch_train --input_directory=$INPUT_DIRECTORY/$DATASET_NAME --output_directory=$OUTPUT_DIRECTORY --training_iterations=$NUMBER_OF_ITERATIONS

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$DATASET_NAME```.

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -m launch_train --help
