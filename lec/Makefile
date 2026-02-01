DOCS=\
	_output/2026-01-21.pdf _output/2026-01-21.html \
	_output/2026-01-21-intro.html _output/2026-01-21-blas.html \
	_output/2026-01-23.pdf _output/2026-01-23.html \
	_output/2026-01-23-slides.html \
	_output/2026-01-26-slides.html \
	_output/2026-01-30.pdf _output/2026-01-30.html \
	_output/2026-02-02.pdf _output/2026-02-02.html \
	_output/2026-02-02-slides-err.html \
	_output/2026-02-02-slides-fp.html \
	_output/2026-02-04.pdf _output/2026-02-04.html

all: $(DOCS)

_output/%.html _output/%.pdf: %.qmd
	quarto render $< --to all

_output/%.html: %.qmd
	quarto render $<
