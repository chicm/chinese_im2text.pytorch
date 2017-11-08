from eval_utils import language_eval_chinese
import json
import __future__

INPUT_FILE = 'results/sat_224_1105/val_out_model.json'
preds = json.load(open(INPUT_FILE, 'r'))

ref_files = ['refs/coco_val_ref_'+str(i)+'.json' for i in range(5)]

print(ref_files)
scores = []

for ref_file in ref_files:
    scores.append(language_eval_chinese(preds, ref_file))

sum_score = scores[0]
for score in scores[1:]:
    for k in score.keys():
        sum_score[k] += score[k]

for k in sum_score.keys():
    sum_score[k]  /= 5

print('AVERAGE SCORE:' + str(sum_score))