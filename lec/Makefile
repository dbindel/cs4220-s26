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
	_output/2026-02-04.pdf _output/2026-02-04.html \
	_output/2026-02-06.pdf _output/2026-02-06.html \
	_output/2026-02-09.pdf _output/2026-02-09.html \
	_output/2026-02-11.pdf _output/2026-02-11.html \
	_output/2026-02-13.pdf _output/2026-02-13.html \
	_output/2026-02-18.pdf _output/2026-02-18.html \
	_output/2026-02-18-slides.html \
	_output/2026-02-20.pdf _output/2026-02-20.html \
	_output/2026-02-23.pdf _output/2026-02-23.html \
	_output/2026-02-25.pdf _output/2026-02-25.html \
	_output/2026-03-02.pdf _output/2026-03-02.html \
	_output/2026-03-04.pdf _output/2026-03-04.html \
	_output/2026-03-06.pdf _output/2026-03-06.html \
	_output/2026-03-11.pdf _output/2026-03-11.html \
	_output/2026-03-13.pdf _output/2026-03-13.html \
	_output/2026-03-16.pdf _output/2026-03-16.html \
	_output/2026-03-18.pdf _output/2026-03-18.html \
	_output/2026-03-20.pdf _output/2026-03-20.html \
	_output/2026-03-23.pdf _output/2026-03-23.html \
	_output/2026-03-25.pdf _output/2026-03-25.html \
	_output/2026-03-27.pdf _output/2026-03-27.html \
	_output/2026-04-06.pdf _output/2026-04-06.html \
	_output/2026-04-08.pdf _output/2026-04-08.html \
	_output/2026-04-10.pdf _output/2026-04-10.html \
	_output/2026-04-06-slides.html \
	_output/2026-04-13.pdf _output/2026-04-13.html \
	_output/2026-04-15.pdf _output/2026-04-15.html \
	_output/2026-04-17.pdf _output/2026-04-17.html \
	_output/2026-04-20.pdf _output/2026-04-20.html \
	_output/2026-04-22.pdf _output/2026-04-22.html \
	_output/2026-04-24.pdf _output/2026-04-24.html 

all: $(DOCS)

%.qmd: %.jl
	lua ../util/ldoc.lua -p quarto -highlight julia $< -o $@

_output/%.html _output/%.pdf: %.qmd
	quarto render $< --to all

_output/%.html: %.qmd
	quarto render $<
