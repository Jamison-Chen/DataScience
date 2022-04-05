import json

tse_stage3 = {
    "type": "tse",
    "stock": {},
}

with open("tse_stage2.json", "r") as f:
    data = json.loads(f.read())["stock"]
    for sid in data:
        tse_stage3["stock"][sid] = sum(data[sid], [])

with open("tse_stage3.json", "w", encoding="utf-8") as f:
    json.dump(tse_stage3, f, ensure_ascii=False, indent=4)

otc_stage3 = {
    "type": "otc",
    "stock": {},
}

with open("otc_stage2.json", "r") as f:
    data = json.loads(f.read())["stock"]
    for sid in data:
        otc_stage3["stock"][sid] = sum(data[sid], [])

with open("otc_stage3.json", "w", encoding="utf-8") as f:
    json.dump(otc_stage3, f, ensure_ascii=False, indent=4)
