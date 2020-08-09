data_url = http://archive.ics.uci.edu/ml/machine-learning-databases/00359/
data_file = NewsAggregatorDataset.zip


all: model/*.py data/*.csv
	news-classifier --path data/newsCorpora.csv
	

%.csv:
	curl $(data_url)/$(data_file) -O data/$(data_file)
	unzip data/$(data_file) -p data/

.PHONY: all
