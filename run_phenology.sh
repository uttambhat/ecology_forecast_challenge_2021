mkdir -p targets submissions
wget https://data.ecoforecast.org/targets/phenology/phenology-targets.csv.gz
gzip -dc phenology-targets.csv.gz > targets/phenology-targets_$(date +"%Y-%m-%d").csv
python3 run_phenology.py
rm phenology-targets.csv.gz
