file="validation_fit_values.txt"
if [ -f $file ] ; then
    rm $file
fi

for i in $(seq -f "%04g" 1 50)
do

python run_SpecSelFunction_Moller.py --data /scratch/midway2/rkessler/SNDATA_ROOT/FIT/DES3YR_v7/DES3YR_DES_SPECEFF_VALIDATION/DES3YR_DES_VALIDATION_STATONLY-G10-"$i"/FITOPT000.FITRES  --sim /scratch/midway2/rkessler/SNDATA_ROOT/FIT/DES3YR_v7/DES3YR_DES_SPECEFF/DES3YR_DES_SPECEFF_AMG10/FITOPT000.FITRES.gz --nameout speceff_validation/fits/speceff_AMG10_"$i".DAT  --validation

python run_SpecSelFunction_Moller.py --data /scratch/midway2/rkessler/SNDATA_ROOT/FIT/DES3YR_v7/DES3YR_DES_SPECEFF_VALIDATION/DES3YR_DES_VALIDATION_STATONLY-C11-"$i"/FITOPT000.FITRES  --sim /scratch/midway2/rkessler/SNDATA_ROOT/FIT/DES3YR_v7/DES3YR_DES_SPECEFF/DES3YR_DES_SPECEFF_AMC11/FITOPT000.FITRES.gz --nameout speceff_validation/fits/speceff_AMC11_"$i".DAT  --validation 

done
