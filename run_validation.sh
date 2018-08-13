for i in $(seq -f "%04g" 1 50)
do

python run_SpecSelFunction_Moller.py --data /scratch/midway2/rkessler/SNDATA_ROOT/FIT/DES3YR_v6/DES3YR_DES_VALIDATION/DES3YR_DES_VALIDATION_STATONLY-"$i"/FITOPT000.FITRES --nameout speceff_validation/fits/speceff_AMG10_"$i".DAT  --validation 

done
