.PHONY: plots publish

plots:
	src/python/plot/plot.py -p total-training-time -d stats -o src/tex/total-training-time.pdf
	src/python/plot/plot.py -p avg-epoch-time -d stats -o src/tex/avg-epoch-time.pdf

publish:
	git subtree push --prefix src/ui origin gh-pages
