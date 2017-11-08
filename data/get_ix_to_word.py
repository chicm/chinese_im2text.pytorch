import json

info = json.load(open('ai_challenger.json', 'r'))
itow = info['ix_to_word']
print(len(itow.items()))

json.dump(itow, open('ix_to_w.json', 'w'), ensure_ascii=False, indent=4)
