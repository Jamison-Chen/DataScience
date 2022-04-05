import json

tse_stage1 = {
    "type": "tse",
    "fields": [
        "日期",
        "成交千股",
        "成交千元",
        "開盤價",
        "最高價",
        "最低價",
        "收盤價",
        "漲跌價差",
        "成交筆數",
    ],
    "stock": {},
}

with open("tse.json", "r") as f:
    data = json.loads(f.read())
    for sid in data:
        for i in range(len(data[sid]["data"])):
            del data[sid]["data"][i][10]
            del data[sid]["data"][i][9]
        tse_stage1["stock"][sid] = data[sid]["data"]

with open("tse_stage1.json", "w", encoding="utf-8") as f:
    json.dump(tse_stage1, f, ensure_ascii=False, indent=4)

otc_stage1 = {
    "type": "otc",
    "fields": [
        "日期",
        "成交千股",
        "成交千元",
        "開盤價",
        "最高價",
        "最低價",
        "收盤價",
        "漲跌價差",
        "成交筆數",
    ],
    "stock": {},
}

with open("otc.json", "r") as f:
    data = json.loads(f.read())
    for sid in data:
        for i in range(len(data[sid]["data"])):
            del data[sid]["data"][i][10]
            del data[sid]["data"][i][9]
        otc_stage1["stock"][sid] = data[sid]["data"]

with open("otc_stage1.json", "w", encoding="utf-8") as f:
    json.dump(otc_stage1, f, ensure_ascii=False, indent=4)
