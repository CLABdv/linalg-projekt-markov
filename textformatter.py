#!/usr/bin/python

fr = open("asv.txt", "r")
fw = open("asv_formatted.txt", "w")

text = fr.read().splitlines()
fr.close()

text = list(map(lambda x: " ".join(x.split()[2:]), text))
text = "\n".join(text)
fw.write(text)
fw.close()


