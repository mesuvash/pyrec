def loadItemInfo(fpath, delim="\t"):
	item_info = {}
	with open(fpath) as fp:
		for line in fp:
			try:
				item, name, url = line.strip().split("\t")
				item_info[item] = (name, url)
			except:
				pass
	return item_info