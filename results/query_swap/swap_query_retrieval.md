# Query-swap control: swap-query retrieval on the auxiliary set and the held-out batch

## medical: attacker VLM biomedclip

base query: "Any chest X-ray showing pneumonia"  ·  swap query: "Any chest X-ray showing pleural effusion" (tag qeff)

| Query | class | AUROC on aux | prec@10% | base rate | positives in aux |
|---|---|---|---|---|---|
| base | pneumonia | 0.639 | 0.0089 | 0.0038 | 43 |
| base | effusion | 0.542 | 0.1007 | 0.0817 | 916 |
| swap | pneumonia | 0.508 | 0.0027 | 0.0038 | 43 |
| swap | effusion | 0.770 | 0.2665 | 0.0817 | 916 |

| batch image | labels | rank under base query | rank under swap query |
|---|---|---|---|
| 0-12.png | pleural | 2/8 | 3/8 |
| 1-4.png | mass | 3/8 | 8/8 |
| 2-11.png | fibrosis | 7/8 | 6/8 |
| 3-3.png | infiltration | 5/8 | 5/8 |
| 4-6.png | pneumonia **(base target)** | 8/8 | 7/8 |
| 5-6.png | pneumonia **(base target)** | 6/8 | 4/8 |
| 6-2.png | effusion **(swap class)** | 4/8 | 1/8 |
| 7-0.png | atelectasis | 1/8 | 2/8 |

## uav: attacker VLM clip

base query: "aerial drone image showing solar panels on rooftops"  ·  swap query: "aerial drone image showing an airport runway" (tag qair)

| Query | class | AUROC on aux | prec@10% | base rate | positives in aux |
|---|---|---|---|---|---|
| base | solar_board | 0.916 | 0.5594 | 0.0765 | 154 |
| base | airstrip | nan | 0.0000 | 0.0000 | 0 |
| swap | solar_board | 0.410 | 0.0050 | 0.0765 | 154 |
| swap | airstrip | nan | 0.0000 | 0.0000 | 0 |

| batch image | labels | rank under base query | rank under swap query |
|---|---|---|---|
| 0-0_1_2_6_7_8_9_10_16.png | roof, dirt_motor_road, paved_motor_road, container, airstrip, traffic_barrier, green_field, wild_field, sedan **(swap class)** | 8/8 | 2/8 |
| 1-0_1_2_9_11_15_16.png | roof, dirt_motor_road, paved_motor_road, green_field, solar_board, paved_walk, sedan **(base target)** | 2/8 | 5/8 |
| 2-0_1_9.png | roof, dirt_motor_road, green_field | 6/8 | 6/8 |
| 3-9.png | green_field | 3/8 | 3/8 |
| 4-0_1_2_9_11_15_16.png | roof, dirt_motor_road, paved_motor_road, green_field, solar_board, paved_walk, sedan **(base target)** | 1/8 | 4/8 |
| 5-0_1_2_6_7_8_9_10_14_16.png | roof, dirt_motor_road, paved_motor_road, container, airstrip, traffic_barrier, green_field, wild_field, car_park, sedan **(swap class)** | 7/8 | 1/8 |
| 6-0_1_9_16.png | roof, dirt_motor_road, green_field, sedan | 5/8 | 7/8 |
| 7-0_1_9_16.png | roof, dirt_motor_road, green_field, sedan | 4/8 | 8/8 |

