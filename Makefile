data_url = http://archive.ics.uci.edu/ml/machine-learning-databases/00359/
data_file = NewsAggregatorDataset.zip

.PHONY: all 

all: model/*.py __main__.py data/*.csv
	python .

%.csv:
	wget $(data_url)/$(data_file) -O data/$(data_file)
	unar -D data/$(data_file) -o data/
