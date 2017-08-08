TDSE_DIR="/Users/cgoldsmith/repos/TDSE"
for i in -20 -10 0 10 20; do
	mkdir tau_$i
	cd ./tau_$i
	pwd
	cp ../input.json .
	python $TDSE_DIR/scripts/run_scripts/change_tau.py $i
	sbatch ../slurm_daisychain.summitcu
	cd ..
done;