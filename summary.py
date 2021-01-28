import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


lr_list = ['4e-4', '8e-4', '16e-4']
legend_list = ['rr', 'rf', 'fr', 'ff']
pos = [-0.45, -0.15, 0.15, 0.45]
color = ['#D7191C', '#2C7BB6', '#dd1c77', '#31a354']
legends = ['Random inits, random batches', 'Random inits, fixed batches', 'Fixed inits, ramdom batches', 'Fixed inits, fixed batches']

summary_l2 = True

trials = 8
plt.figure()

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def get_l2_distance(model1, model2):
    w1 = np.concatenate(list(map(lambda x: x.numpy().reshape(-1), model1.weights)))
    w2 = np.concatenate(list(map(lambda x: x.numpy().reshape(-1), model2.weights)))
    return np.linalg.norm(w1 - w2)
def get_label_disagree(arr1, arr2):
    return np.sum(arr1 != arr2) / 10000.
for epoch in [9, 49, 99, 199]:
    for idx, legend in enumerate(legend_list):
        plot_data = []
        for lr in lr_list:
            arr = []
            models = []
            for trial in range(trials):
                if summary_l2:
                    models.append(tf.keras.models.load_model(f'smallcnnckptdet/{lr}{legend}{trial+1}/ckpt{epoch}'))
                else:
                    arr.append(np.loadtxt('smallcnnckptdet/{}{}{}/pred{}.txt'.format(lr, legend, trial+1, epoch)).astype(int))
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
                    if summary_l2:
                        num.append(get_l2_distance(models[i], models[j]))
                    else:
                        num.append(get_label_disagree(arr[i], arr[j]))
            # print(np.percentile(np.array(num), q=50))

            plot_data.append(np.array(num))
            
        bp = plt.boxplot(plot_data, positions=np.array(range(3))*2.0+pos[idx], sym='', widths=0.2, notch=True)
        set_box_color(bp, color[idx])
        plt.plot([], c=color[idx], label=legends[idx])

    plt.legend()

    plt.xticks(range(0, len(lr_list) * 2, 2), lr_list)
    # plt.xlim(-2, len(ticks)*2)
    plt.ylim(0, 90)
    plt.xlabel('Learning rate')

    if summary_l2:
        plt.ylabel('L2 distance of model parameters')
    else:
        plt.ylabel('Fraction of test label diffferent')
    
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(5, 7)
    plt.subplots_adjust(left=0.10, bottom=0.07, right=0.98, top=0.98)
    plt.show()
    if summary_l2:
        plt.savefig(f'smallcnndetl2dis{epoch+1}.png')
    else:
        plt.savefig(f'smallcnne{epoch+1}.png')
    fig.clear()
    plt.close(fig)

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