DOCS=\
	_output/midterm.html _output/midterm.pdf \
	_output/finalp.html _output/finalp.pdf \

all: $(DOCS)

_output/%.html _output/%.pdf: %.qmd
	quarto render $< --to all

_output/%.html: %.qmd
	quarto render $<
