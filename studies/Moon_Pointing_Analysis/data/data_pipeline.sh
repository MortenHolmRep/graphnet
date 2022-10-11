db=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/08000-08999/
gcd=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/08000-08999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz
pulse="InIceDSTPulses"

outdir=/scratch/mholm/database/

python /data/user/mholm/graphnet/studies/Moon_Pointing_Analysis/data/convert_i3_to_sqlite.py --db ${db} --gcd ${gcd} --pulse ${pulse} --outdir ${outdir}