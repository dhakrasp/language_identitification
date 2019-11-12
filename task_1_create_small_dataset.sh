mkdir -p langid-data-small/task1/train
shuf -n $1 langid-data/task1/data.en > langid-data-small/task1/train/data.en
shuf -n $1 langid-data/task1/data.es > langid-data-small/task1/train/data.es
shuf -n $1 langid-data/task1/data.pt > langid-data-small/task1/train/data.pt

mkdir -p langid-data-small/task1/test
shuf -n $2 langid-data/task1/data.en > langid-data-small/task1/test/data.en
shuf -n $2 langid-data/task1/data.es > langid-data-small/task1/test/data.es
shuf -n $2 langid-data/task1/data.pt > langid-data-small/task1/test/data.pt