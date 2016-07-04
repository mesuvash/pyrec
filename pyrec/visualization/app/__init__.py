from flask import Flask, render_template
import random
from helper import loadItemInfo, getTopKItems, getItemsInRange
import json

item_info_path = "/Users/sedhain/Desktop/adobe-demo-files/item_name_imgurl.rpt"
item_sim_learned_file = "/Users/sedhain/Desktop/adobe-demo-files/item_sim_ulowlinear_sorted.json"
item_sim_knn_file = "/Users/sedhain/Desktop/adobe-demo-files/item_sim_sorted.json"

item_purchase_count_file = "/Users/sedhain/Desktop/adobe-demo-files/item_count.json"
similarities_learned = json.load(open(item_sim_learned_file))
similarities_knn = json.load(open(item_sim_knn_file))
icount = json.load(open(item_purchase_count_file))

k = 5000
# k_popular_items = getTopKItems(icount, k)
k_popular_items = getItemsInRange(icount, 30, 80)
item_info = loadItemInfo(item_info_path)
# items_with_info = list(set(similarities.keys()).intersection(item_info.keys()))
items_with_info = list(set(similarities_knn.keys()).intersection(
    item_info.keys()).intersection(k_popular_items))

all_items_with_url = item_info.keys()
app = Flask(__name__)


@app.route("/")
def main():
    sample_items = random.sample(items_with_info, 20)
    random_display_items = []
    for item in sample_items:
        name, url = item_info[item]
        random_display_items.append((item, name, url))
    return render_template("home.html", items=random_display_items, icount=icount)


@app.route("/similar_items/<itemid>")
def similar(itemid):
    # similar_items = item_similar[itemid]
    name, url = item_info[itemid]
    source = [(itemid, name, url)]
    similar_items_learned = []
    similar_items_knn = []
    if itemid in similarities_learned:
        for item in similarities_learned[itemid][:5]:
            if item in item_info:
                name, url = item_info[item]
                similar_items_learned.append((item, name, url))
        for item in similarities_knn[itemid][:5]:
            if item in item_info:
                name, url = item_info[item]
                similar_items_knn.append((item, name, url))

    else:
        print "No Images for similar items found"

    return render_template("similar.html", source=source,
                           similar_items_learned=similar_items_learned,
                           similar_items_knn=similar_items_knn,
                           icount=icount)
