def str_to_int(s):
    return 1 if s == "True" else 0

gold = []
with open('data/test.complete.csv') as reader:
    for i, line in enumerate(reader):
        if i > 0:
            gold.append(str_to_int(line.strip().split(',')[-1]))
    
preds = [] 
import sys
with open(sys.argv[1]) as reader:
    for line in reader:
        preds.append(int(line.strip()))
      
correct = 0
for x, y in zip(gold, preds):
    if x == y:
        correct += 1
print(correct/len(gold))
     