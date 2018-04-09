run:
	python drpy.py -n 24 -o datasets-dr/ input.txt
test:
	python drpy.py -n 4 -o datasets-dr/ tests/teste1.txt 
download-datasets:
	@printf ">>>>> Downloading datasets from google-drive ... [1/2] <<<<<\n\n"
	@wget --no-check-certificate -i datasets-csv/link-to-datasets.txt -O datasets-csv/datasets-csv.tar.gz 
	@printf ">>> Download completed <<<\n\n"
	@printf ">>> Decompressing datasets ... [2/2] <<<\n\n"
	@tar -xzvf datasets-csv/datasets-csv.tar.gz -C datasets-csv/
	@rm datasets-csv/datasets-csv.tar.gz
	@printf ">>> Decompressing completed <<<\n\n"
	@printf ">>> Finished <<<\n\n"
run-euler:
	qsub run_euler.sh
