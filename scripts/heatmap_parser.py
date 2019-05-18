#!/bin/env python3


def write_row(file_, *columns):
    print(*columns, sep=" & ", end=" \\\\ \\cmidrule(l){2-9} \n", file=file_)

def eliminate_duplicate_lines():
	input_file = "heatmap.tbl"
	output_file = "new_heatmap.tbl"
	with open(input_file, "r") as in_file:
		with open(output_file, "w") as out_file:
			lines = in_file.readlines();
			for index in range(0, len(lines),4):
				out_file.write(lines[index])
				out_file.write(lines[index + 1])

def main():
	input_file = "new_heatmap.tbl"
	output_file = "latex_table.txt"
	#eliminate_duplicate_lines()
	with open(input_file, "r") as in_file:
		with open(output_file, "w") as out_file:
			lines = in_file.readlines();
			write_row(out_file, "Inner Relation Size (MiTuples)", "Additional Pipeline Cost", "BF size (Mbits)", "w", "s", "z", "k", "tw", "Device")
			for line in lines:
				words = line.split()
				write_row(out_file, int(int(words[20])/(1024*1024)), words[12], int(int(words[9])/(1024*1024*8)), words[1], words[3], words[5], words[7], words[16], words[10]);




if __name__ == "__main__":
	main()