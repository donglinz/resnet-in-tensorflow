import numpy as np
import matplotlib.pyplot as plt

lr_list = ['4e-4', '8e-4', '16e-4']
legend_list = ['rr', 'rf', 'fr', 'ff']
pos = [-0.45, -0.15, 0.15, 0.45]
color = ['#D7191C', '#2C7BB6', '#dd1c77', '#31a354']
legends = ['Random inits, random batches', 'Random inits, fixed batches', 'Fixed inits, ramdom batches', 'Fixed inits, fixed batches']

trials = 9
plt.figure()

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

for idx, legend in enumerate(legend_list):
    plot_data = []
    for lr in lr_list:
        arr = []
        for trial in range(trials):
            arr.append(np.loadtxt('{}{}{}/pred199.txt'.format(lr, legend, trial+1)).astype(int))
            # if lr == '4e-4':
            #     arr.append(np.loadtxt('{}{}{}/pred19.txt'.format(lr, legend, trial+1)).astype(int))
            # if lr == '8e-4':
            #     arr.append(np.loadtxt('{}{}{}/pred9.txt'.format(lr, legend, trial+1)).astype(int))
            # if lr == '16e-4':
            #     arr.append(np.loadtxt('{}{}{}/pred4.txt'.format(lr, legend, trial+1)).astype(int))
        print(lr, legend)
        print("--------------------")
        # print(np.sum(np.logical_or(arr[0] != arr[1], arr[1] != arr[2])))
        num = []
        for i in range(trials):
            for j in range(trials):
                if j <= i:
                    continue
                # print(np.sum(arr[i] != arr[j]))
                num.append(np.sum(arr[i] != arr[j]))
        # print(np.percentile(np.array(num), q=50))
        plot_data.append(np.array(num) / 10000)
    bp = plt.boxplot(plot_data, positions=np.array(range(3))*2.0+pos[idx], sym='', widths=0.2, notch=True)
    set_box_color(bp, color[idx])
    plt.plot([], c=color[idx], label=legends[idx])

plt.legend()

plt.xticks(range(0, len(lr_list) * 2, 2), lr_list)
# plt.xlim(-2, len(ticks)*2)
plt.ylim(0.1, 0.40)
plt.xlabel('Learning rate')
plt.ylabel('Fraction of test labels different')
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(5, 7)
plt.subplots_adjust(left=0.15, bottom=0.07, right=0.98, top=0.98)
plt.show()
plt.savefig('smallcnndete200.png')

# for lr in lr_list:
#     for legend in legend_list:
#         print(lr, legend)
#         print("--------------------")
#         for epoch in range(0, 50):
#             arr = []
#             for trial in range(trials):
#                     arr.append(np.loadtxt('{}{}{}/pred{}.txt'.format(lr, legend, trial+1, epoch)).astype(int))
#             num = []
#             for i in range(trials):
#                 for j in range(trials):
#                     if j <= i:
#                         continue
#                     # print(np.sum(arr[i] != arr[j]))
#                     num.append(np.sum(arr[i] != arr[j]))
#             print(np.percentile(np.array(num), q=50))