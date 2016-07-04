def loadItemInfo(fpath, delim="\t"):
    item_info = {}
    with open(fpath) as fp:
        for line in fp:
            try:
                item, name, url = line.strip().split("\t")
                item_info[item] = (name.decode("utf-8"), url.decode("utf-8"))
            except:
                pass
    return item_info


def getTopKItems(icount, k):
    sorted_items = sorted(icount.items(), key=lambda x: x[1])[::-1]
    return map(lambda x: x[0], sorted_items[:k])

def getItemsInRange(icount, low, high):
    filtered_items = filter(lambda x: True if ((x[1] >= low) and (x[1] <= high)) else False, icount.items())
    return map(lambda x: x[0], filtered_items)
