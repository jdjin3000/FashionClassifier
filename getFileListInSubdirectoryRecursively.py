import glob

for filename in glob.iglob('img/' + '**/*.jpg', recursive=True):
	print(filename)