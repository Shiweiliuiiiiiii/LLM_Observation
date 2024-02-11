for iteration in 10
do
python convert_hf.py --fairseq_path metaseq/opt-2.7-${iteration}0000-resharded/pytorch_model.bin \
--pytorch_dump_folder_path hf_2.7b-${iteration}0000 \
--hf_config metaseq/opt-2.7-${iteration}0000-resharded/config.json
done