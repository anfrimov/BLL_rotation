Use of terminal/neurogrids.py

python /path/to/neurogrids.py /path/to/dynamic_pattern.txt \
-s /path/to/static_pattern.txt -p /path/to/primary_pattern.txt \
-o /path/to/output/directory -D dynamic-color -S static-color \
-P primary-color -g grid-size -d

Example use:

cd "/Users/anthony819/Documents/Python/Brain Language Lab/code"
python3 terminal/neurogrids.py ../Inputs/03b_dynamic_patterns.txt \
-s ../Inputs/02b_static_patterns.txt -p ../Inputs/01b_primary_areas_patterns.txt \
-n test_01 --mp4 -o /Users/anthony819/Documents/Python/neuro_test

cd "/Users/anthony819/Documents/Python/Brain Language Lab/code"
python3 terminal/neurogrids.py --palette-options