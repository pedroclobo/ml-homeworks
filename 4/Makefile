report:: src/report.tex
	pdflatex -aux-directory=bin -output-directory=bin src/report.tex && mv bin/report.pdf bin/G067.pdf

clean::
	find . -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "report.pdf" \) | xargs rm -f
