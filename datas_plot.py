d1 = [
	[91.4,91.05,91.03333333333333,91.02,91.052,90.995,90.33571428571429,90.82125,90.99222222222222,90.375,],
	[91.52,90.89,90.76666666666667,90.81,90.232,90.90166666666667,91.05857142857143,90.95,90.81111111111112,90.422],
	[91.57,90.87,90.77333333333333,91.1075,90.99,90.865,90.84285714285714,90.11375,90.89444444444445,91.04],
	[91.5,91.125,90.87333333333333,91.09,91.122,90.60666666666667,90.87,90.39,90.87777777777778,90.958],
	[90.83,90.61,91.15666666666667,90.985,90.348,90.325,90.92857142857143,90.98875,90.94555555555556,90.722],
	
]

d2 = [
	[99.4410569105691,98.98477157360406,72.7944800394283,77.8829604130809,73.99,],
	[96.67338709677419,98.79547689282202,93.03170409511229,95.29156565012121,86.42,],
	[98.55670103092784,68.69009584664536,98.30231798889977,90.84420379030273,86.47,],
	[99.26028663892741,97.76011560693641,90.23496083986002,93.60916613621897,90.11,],
	[94.80171489817792,92.70343804105862,94.26566244436837,87.61617683998995,88.0],
	[93.03322615219722,98.27147941026945,96.74983585029547,81.04259534422982,95.12],
	[99.12321181356714,95.36847492323439,98.68787928489421,90.65563335455124,92.36],
	[99.07706506691278,96.99230028873917,94.20939107350257,96.48129921259843,96.13],
	
]

# As if we are not able to run it for five times once, it will use up all memory. 
# We ran it once by once and record it here.

import matplotlib.pyplot as plt

ac = d2
l = len(ac[0])

mean_accuracies = []
for i in range(l):
	di = [d[i] for d in ac]
	mean_accuracies.append(sum(di)/len(di))

print(mean_accuracies)

# Setting up the plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, l+1), mean_accuracies, marker='')  
plt.xlabel('Number of Tasks')
plt.ylabel('Accuracy')
plt.title('Accuracy per Number of Tasks')
plt.grid(False)
plt.xticks(range(1, l+1))
plt.yticks(range(80, 100))
plt.show()


'''
NUM_TASK:1 Started!
100%
 200/200 [00:10<00:00, 19.83it/s]
BATCH_SIZE:300 NUM_TASK:1 train done!
Test Loss: 0.3534327568486333
Accuracy: 91.52%
NUM_TASK:1 Finished!------------------
NUM_TASK:2 Started!
100%
 400/400 [00:21<00:00, 19.19it/s]
BATCH_SIZE:300 NUM_TASK:2 train done!
Test Loss: 0.3925085614310272
Accuracy: 90.89%
NUM_TASK:2 Finished!------------------
NUM_TASK:3 Started!
100%
 600/600 [00:33<00:00, 20.85it/s]
BATCH_SIZE:300 NUM_TASK:3 train done!
Test Loss: 0.40495866959914567
Accuracy: 90.76666666666667%
NUM_TASK:3 Finished!------------------
NUM_TASK:4 Started!
100%
 800/800 [00:43<00:00, 18.92it/s]
BATCH_SIZE:300 NUM_TASK:4 train done!
Test Loss: 0.3943900408487378
Accuracy: 90.81%
NUM_TASK:4 Finished!------------------
NUM_TASK:5 Started!
100%
 1000/1000 [00:58<00:00, 17.30it/s]
BATCH_SIZE:300 NUM_TASK:5 train done!
Test Loss: 0.40078601216573917
Accuracy: 90.232%
NUM_TASK:5 Finished!------------------
NUM_TASK:6 Started!
100%
 1200/1200 [01:08<00:00, 18.40it/s]
BATCH_SIZE:300 NUM_TASK:6 train done!
Test Loss: 0.40116026090923695
Accuracy: 90.90166666666667%
NUM_TASK:6 Finished!------------------
NUM_TASK:7 Started!
100%
 1400/1400 [01:20<00:00, 17.75it/s]
BATCH_SIZE:300 NUM_TASK:7 train done!
Test Loss: 0.39115013378377783
Accuracy: 91.05857142857143%
NUM_TASK:7 Finished!------------------
NUM_TASK:8 Started!
100%
 1600/1600 [01:33<00:00, 18.35it/s]
BATCH_SIZE:300 NUM_TASK:8 train done!
Test Loss: 0.4004684590179934
Accuracy: 90.95%
NUM_TASK:8 Finished!------------------
NUM_TASK:9 Started!
100%
 1800/1800 [01:48<00:00, 17.55it/s]
BATCH_SIZE:300 NUM_TASK:9 train done!
Test Loss: 0.4038088215018312
Accuracy: 90.81111111111112%
NUM_TASK:9 Finished!------------------
NUM_TASK:10 Started!









UM_TASK:1 Started!
split classes: [[6, 4]]
100%
 12/12 [00:02<00:00,  5.92it/s]
BATCH_SIZE:1024 NUM_TASK:1 train done!
Test Loss: 0.0493000028654933
Accuracy: 98.65979381443299%
NUM_TASK:1 Finished!------------------
NUM_TASK:2 Started!
split classes: [[3, 8], [1, 7]]
100%
 25/25 [00:06<00:00,  4.68it/s]
BATCH_SIZE:1024 NUM_TASK:2 train done!
Test Loss: 1.5459524154663087
Accuracy: 43.86303351820593%
NUM_TASK:2 Finished!------------------
NUM_TASK:3 Started!
split classes: [[2, 1], [0, 4], [6, 3]]
100%
 36/36 [00:07<00:00,  6.76it/s]
BATCH_SIZE:1024 NUM_TASK:3 train done!
Test Loss: 0.06651805993169546
Accuracy: 98.44185665081187%
NUM_TASK:3 Finished!------------------
NUM_TASK:4 Started!
split classes: [[6, 4], [2, 8], [9, 5], [0, 7]]
100%
 47/47 [00:08<00:00,  7.63it/s]
BATCH_SIZE:1024 NUM_TASK:4 train done!
Test Loss: 0.39153482392430305
Accuracy: 85.43602800763844%
NUM_TASK:4 Finished!------------------
NUM_TASK:5 Started!
split classes: [[9, 0], [2, 8], [6, 5], [3, 7], [4, 1]]
100%
 59/59 [00:15<00:00,  6.63it/s]
BATCH_SIZE:1024 NUM_TASK:5 train done!
Test Loss: 0.49784830808639524
Accuracy: 86.11%
NUM_TASK:5 Finished!------------------
'''