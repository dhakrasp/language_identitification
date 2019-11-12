mkdir -p langid-data-small/task2/train
shuf -n $1 langid-data/task2/data.pt-pt > langid-data-small/task2/train/data.pt-pt
shuf -n $1 langid-data/task2/data.pt-br > langid-data-small/task2/train/data.pt-br

mkdir -p langid-data-small/task2/test
shuf -n $1 langid-data/task2/data.pt-pt > langid-data-small/task2/test/data.pt-pt
shuf -n $1 langid-data/task2/data.pt-br > langid-data-small/task2/test/data.pt-br
