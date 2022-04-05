import json

# tse_stage2 = {
#     "type": "tse",
#     "fields": ["成交千股", "成交千元", "成交筆數", "漲跌百分比", "高低價差百分比"],
#     "stock": {},
# }

# with open("tse.json", "r") as f:
#     data = json.loads(f.read())
#     for sid in data:
#         if len(data[sid]["data"]) != 56:
#             print(len(data[sid]["data"]), sid)
#             continue
#         for i in range(len(data[sid]["data"])):
#             data[sid]["data"][i] = [
#                 data[sid]["data"][i][1],
#                 data[sid]["data"][i][2],
#                 data[sid]["data"][i][8],
#                 round(data[sid]["data"][i][7] / data[sid]["data"][i][6], 4),
#                 round(
#                     (data[sid]["data"][i][4] - data[sid]["data"][i][5])
#                     / data[sid]["data"][i][5],
#                     4,
#                 ),
#             ]
#         tse_stage2["stock"][sid] = data[sid]["data"]

# with open("tse_stage2.json", "w", encoding="utf-8") as f:
#     json.dump(tse_stage2, f, ensure_ascii=False, indent=4)

otc_stage2 = {
    "type": "otc",
    "fields": ["成交千股", "成交千元", "成交筆數", "漲跌百分比", "高低價差百分比"],
    "stock": {},
}

with open("otc.json", "r") as f:
    data = json.loads(f.read())
    for sid in data:
        if len(data[sid]["data"]) != 56:
            print(len(data[sid]["data"]), sid)
            continue
        for i in range(len(data[sid]["data"])):
            data[sid]["data"][i] = [
                data[sid]["data"][i][1],
                data[sid]["data"][i][2],
                data[sid]["data"][i][8],
                round(data[sid]["data"][i][7] / data[sid]["data"][i][6], 4),
                round(
                    (data[sid]["data"][i][4] - data[sid]["data"][i][5])
                    / data[sid]["data"][i][5],
                    4,
                ),
            ]
        otc_stage2["stock"][sid] = data[sid]["data"]

with open("otc_stage2.json", "w", encoding="utf-8") as f:
    json.dump(otc_stage2, f, ensure_ascii=False, indent=4)
