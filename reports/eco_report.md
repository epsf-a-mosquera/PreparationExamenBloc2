# Rapport d'impact écologique (CodeCarbon)

Source : fichiers fusionnés depuis `reports/emissions/emissions.csv*`

Fichiers pris en compte :
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_0.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_1.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_10.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_11.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_12.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_13.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_14.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_15.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_16.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_17.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_18.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_19.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_2.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_20.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_21.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_22.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_23.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_24.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_25.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_26.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_27.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_28.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_29.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_3.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_4.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_5.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_6.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_7.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_8.bak`
- `/home/ubuntu/PreparationExamenBloc2/reports/emissions/emissions.csv_9.bak`

## Résumé par phase

| project_name                           |   emissions |   energy_consumed |   duration |   emissions_g |   duration_min |
|:---------------------------------------|------------:|------------------:|-----------:|--------------:|---------------:|
| bloc2::kafka_pipeline_send             | 0.000730118 |       0.00251068  |  111.569   |    0.730118   |      1.85948   |
| bloc2::kafka_consumer_ingest_infer     | 0.000658824 |       0.00226552  |  106.876   |    0.658824   |      1.78127   |
| bloc2::kafka_producer_send             | 0.000387803 |       0.00133355  |   44.6874  |    0.387803   |      0.74479   |
| bloc2::sql_ingestion_predictions_batch | 0.000315323 |       0.00108431  |   89.489   |    0.315323   |      1.49148   |
| bloc2::generate_data                   | 5.95605e-06 |       2.04812e-05 |    1.34811 |    0.00595605 |      0.0224685 |
| bloc2::etl_extract_transform           | 2.5144e-06  |       8.64635e-06 |    1.50484 |    0.0025144  |      0.0250807 |
| bloc2::ml_train_classification         | 2.0436e-06  |       7.02738e-06 |    1.31468 |    0.0020436  |      0.0219114 |

## Totaux

- Emissions totales : 0.002103 kgCO2e (2.10 gCO2e)
- Énergie totale : 0.007230 kWh
- Durée totale : 356.8 s (5.9 min)

## Interprétation (guide rapide)

- `project_name` : nom de la phase trackée (une étape du pipeline).
- `emissions` : CO2e estimé (kg) pour la phase.
- `emissions_g` : même info en grammes (plus lisible).
- `energy_consumed` : énergie estimée (kWh) si disponible.
- `duration` : temps total de calcul (secondes) si disponible.
- Compare les phases : la plus forte émission = phase la plus coûteuse.
- Note : sur certaines VM, CodeCarbon estime l'énergie (mode cpu_load/TDP), c'est une estimation.
