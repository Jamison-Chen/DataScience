from requests import post
from pyquery import PyQuery
import time
import json

otc_list_end_point = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=2&issuetype=4&industry_code=&Page=1&chklike=Y"
tse_list_end_point = "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=1&industry_code=&Page=1&chklike=Y"
tse_stock_info_end_point = (
    "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={}&stockNo={}"
)
# e.g. date=20220301&stockNo=2330
otc_stock_info_end_point = "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d={}&stkno={}"
# e.g. d=111/03/01&stkno=3105

res1 = post(tse_list_end_point)
doc1 = PyQuery(res1.text)
tse_list = doc1.find("tr:not(:first-child)>td:nth-child(3)").text().split(" ")

res2 = post(otc_list_end_point)
doc2 = PyQuery(res2.text)
otc_list = doc2.find("tr:not(:first-child)>td:nth-child(3)").text().split(" ")

# dateListForTse = ["20220101", "20220201", "20220301"]
# allTseData = {}
# for sid in tse_list:
#     allTseData[sid] = {
#         "type": "tse",
#         "fields": [
#             "日期",
#             "成交千股",
#             "成交千元",
#             "開盤價",
#             "最高價",
#             "最低價",
#             "收盤價",
#             "漲跌價差",
#             "成交筆數",
#             "漲跌百分比",
#             "高低價差百分比",
#         ],
#         "data": [],
#     }
#     try:
#         for date in dateListForTse:
#             res = post(tse_stock_info_end_point.format(date, sid))
#             res = PyQuery(res.text).text()
#             modifiedData = json.loads(res)["data"]
#             for j in range(len(modifiedData)):
#                 for i in range(len(modifiedData[j])):
#                     if i != 0:
#                         modifiedData[j][i] = "".join(modifiedData[j][i].split(","))
#                         try:
#                             modifiedData[j][i] = round(float(modifiedData[j][i]), 2)
#                         except:
#                             if i == 1 or i == 2 or i == 8:
#                                 modifiedData[j][i] = 0
#                             else:
#                                 if j > 0:
#                                     modifiedData[j][i] = modifiedData[j - 1][i]
#                                 else:
#                                     raise Exception("")
#                     if i == 1 or i == 2:
#                         modifiedData[j][i] = round(modifiedData[j][i] / 1000, 2)
#                 modifiedData[j].append(
#                     round(modifiedData[j][7] / modifiedData[j][6], 4)
#                 )
#                 modifiedData[j].append(
#                     round(
#                         (modifiedData[j][4] - modifiedData[j][5]) / modifiedData[j][5],
#                         4,
#                     )
#                 )
#             allTseData[sid]["data"].extend(modifiedData)
#             time.sleep(6)
#             print("stock:{} / date {} fetch completed.".format(sid, date))
#     except:
#         del allTseData[sid]
#         print("Something wrong happened when fetching {}".format(sid))
#         continue

# with open("tse.json", "w", encoding="utf-8") as f:
#     json.dump(allTseData, f, ensure_ascii=False, indent=4)


# dateListForOtc = ["111/01/01", "111/02/01", "111/03/01"]
# allOtcData = {}
# for sid in otc_list:
#     allOtcData[sid] = {
#         "type": "otc",
#         "fields": [
#             "日期",
#             "成交千股",
#             "成交千元",
#             "開盤價",
#             "最高價",
#             "最低價",
#             "收盤價",
#             "漲跌價差",
#             "成交筆數",
#             "漲跌百分比",
#             "高低價差百分比",
#         ],
#         "data": [],
#     }
#     try:
#         for date in dateListForOtc:
#             res = post(otc_stock_info_end_point.format(date, sid))
#             res = PyQuery(res.text).text()
#             modifiedData = json.loads(res)["aaData"]
#             for j in range(len(modifiedData)):
#                 for i in range(len(modifiedData[j])):
#                     if i != 0:
#                         modifiedData[j][i] = "".join(modifiedData[j][i].split(","))
#                         try:
#                             modifiedData[j][i] = round(float(modifiedData[j][i]), 2)
#                         except:
#                             if i == 1 or i == 2 or i == 8:
#                                 modifiedData[j][i] = 0
#                             else:
#                                 if j > 0:
#                                     modifiedData[j][i] = modifiedData[j - 1][i]
#                                 else:
#                                     raise Exception("")
#                 modifiedData[j].append(
#                     round(modifiedData[j][7] / modifiedData[j][6], 4)
#                 )
#                 modifiedData[j].append(
#                     round(
#                         (modifiedData[j][4] - modifiedData[j][5]) / modifiedData[j][5],
#                         4,
#                     )
#                 )
#             allOtcData[sid]["data"].extend(modifiedData)
#             # time.sleep(6)
#             print("stock:{} / date {} fetch completed.".format(sid, date))
#     except:
#         del allOtcData[sid]
#         print("Something wrong happened when fetching {}".format(sid))
#         continue

# with open("otc.json", "w", encoding="utf-8") as f:
#     json.dump(allOtcData, f, ensure_ascii=False, indent=4)
