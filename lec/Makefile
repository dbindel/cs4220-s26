DOCS=\
	_output/2026-01-21.pdf _output/2026-01-21.html \
	_output/2026-01-21-intro.html _output/2026-01-21-blas.html \
	_output/2026-01-23.pdf _output/2026-01-23.html \
	_output/2026-01-23-slides.html

all: $(DOCS)

_output/%.html _output/%.pdf: %.qmd
	quarto render $< --to all

_output/%.html: %.qmd
	quarto render $<
