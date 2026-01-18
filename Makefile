DEST=cslinux:/courses/cs4220/2026sp

.PHONY: build clean deploy

build:
	(cd web; jekyll build)
	#(cd lec; quarto render)

clean:
	(cd web; rm -rf _site)
	(cd lec; rm -rf _output)

deploy: build
	(cd web; rsync -avzL _site/ $(DEST) || true)
	(cd lec; rsync -avzL _output/ $(DEST)/lec || true)
	(cd hw;  rsync -avzL *.jl *.html $(DEST)/hw  || true)

