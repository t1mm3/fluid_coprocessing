#! /bin/python
import fileinput


def autofix_unit(x):
    units = [
            (1024*1024*1024, "{div}~Gi"),
            (1024*1024, "{div}~Mi"),
            (1024, "{div}~Ki")
        ]

    flt = False

    txt = "{}".format(x)
    try:
        for scale, label in units:
            if flt:
                num = float(txt)
            else:
                num = int(txt)

            if num >= scale:
                return label.format(div=num / scale)
    except ValueError:
        # Cannot cast
        return txt

    # Not found
    return txt

str_cells = {}
slowdowns = set()
filtersizes = set()

for line in fileinput.input():
	a = line.split()
	# print(a)
	w = int(a[1])
	s = int(a[3])
	z = int(a[5])
	k = int(a[7])
	m = int(a[9])

	fpr = a[21]
	fpr = fpr.replace("fpr", "")
	fpr = float(fpr)

	# print(fpr)

	slowdown = int(a[9+3])
	filtersize = int(a[9+5]) / 8

	# s = "({m}, {k}, {w}, {s}, {z})".format(m=autofix_unit(m), s=s, w=w, z=z, k=k,)
	# s = "({m}, {k})".format(m=autofix_unit(m), s=s, w=w, z=z, k=k,)

	if fpr * 100.0 < 0.05:
		s = " < 0.1 \\%"
	else:
		s = "{:10.1f}\\%".format(fpr*100.0)

	# print("str={} slowdnow{} filter{}".format(s,slowdown,filtersize))

	str_cells[(slowdown, filtersize)] = s

	slowdowns.add(slowdown)
	filtersizes.add(filtersize)

filtersizes = list(filtersizes)
slowdowns = list(slowdowns)

filtersizes.sort(reverse=True)
slowdowns.sort()

num_cols = len(slowdowns)
num_rows = len(filtersizes)
colformat = ""
for a in range(0, num_cols):
	colformat = colformat + " r "

print("""\\begin{{tabular}}{{ c  c | {fmt} }}
	\\multicolumn{num}{{c}}{{ Additional Pipeline Cost $c_A$ }}\\\\
	""".format(fmt=colformat, num=num_cols+2))

line = " \\parbox[t]{2mm}{\\multirow{" + "{}".format(num_rows+1)+ "}{*}{\\rotatebox[origin=c]{90}{Inner Relation Size}}} & "
for col in slowdowns:
	line = line + " & {}".format(col)
line = line + "\\\\"
print(line)
print("\\hline\\")

for size in filtersizes:
	line = "& {}".format(autofix_unit(size))
	for col in slowdowns:
		line = line + " & "
		line = line + str_cells[(col, size)]
		# print(str_cells[(col, size)])
		first = False

	print("{}\\\\".format(line))

print("\\end{tabular}")