This software should run on any Python 3.x version, but was developed in Python 3.9.1.

In order to run the algorithms, it requires input files in the directory mentioned in the file_path variable. These files should be:
- avg.csv: the preference value per time step and per holon.
- upper.csv: the upper bounds per time step and per holon (should be the absolute difference between the value of the upper bound and value of the preference power).
- lower.csv: the lower bounds per time step and per holon (should be the absolute difference between the value of the lower bound and value of the preference power).
- capacity.csv: the capacities of each of the batteries connected to the holons.
- initial_soc.csv: the values of the initial state of charge for each of those aforementioned batteries.

All the other parameters can be set in main.py.

