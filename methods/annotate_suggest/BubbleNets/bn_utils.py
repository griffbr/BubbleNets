# Utils for BubbleNets.

import numpy as np
import os; import glob; import IPython

def num_unique(video_list):
	unique_list = []
	for _, video in enumerate(video_list):
		if not video[:-1] in unique_list:
			unique_list.append(video[:-1])
	return len(unique_list), unique_list

def read_list_file(file_name):
	read_list = open(file_name,'r').readlines()
	for i in range(len(read_list)):
		read_list[i] = read_list[i].strip('\n')
	return read_list

def print_out_text(file_name, text, arg='a'):
	output_file = open(file_name, arg)
	output_file.write(str(text))
	output_file.close()

def print_statements(file_name, statements):
	for _, statement in enumerate(statements):
		print(statement)
		print_out_text(file_name, statement)
