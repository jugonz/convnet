PDFLATEX=pdflatex
BIBTEX=bibtex

TEX-FILES=*.tex
BIB-FILES=*.bib

all:	paper.pdf

paper.pdf: ${TEX-FILES} ${BIB-FILES}
	$(PDFLATEX) paper
	$(BIBTEX) paper
	$(PDFLATEX) paper
	$(PDFLATEX) paper
	#$(PDFLATEX) paper

clean:
	rm -f *.aux *.bbl *.blg *.log *.dvi *.out *.idx *.bak *~ *.toc *.lof
	rm -f paper.pdf
