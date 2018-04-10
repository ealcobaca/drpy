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
compress-datasets:
	@printf ">>>>> compressing datasets-dr <<<<<\n\n"
	@tar -czvf datasets-dr.tar.gz datasets-dr/
get-datasets:
	@printf ">>>>> downloading datasets-dr to local machine <<<<<\n\n"
	@scp alcobaca@143.107.232.170:/home/alcobaca/Studies/drpy-virtualenv/drpy/datasets-dr.tar.gz datasets-dr.tar.gz
euler-run:
	@printf ">>>>> Starting Job <<<<<\n\n"
	@make euler-clean
	qsub run_euler.sh
euler-clean:
	@printf ">>>>> removing erros and output files <<<<<\n\n"
	@rm -f drpy.e* 
	@rm -f drpy.o* 
rm-results:
	@printf ">>>>> removing results <<<<<\n\n"
	@rm -rf datasets-dr/
	@rm -rf report/
