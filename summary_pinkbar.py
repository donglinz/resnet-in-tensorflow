import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

    
experiments = {
    'b128e2' : 1,
    'b512e8' : 7,
    'b2048e32': 31,
    'b8192e128': 127,
    'b50000e1000': 999
}

xlabels = ['128', '512', '2048', '8192', '50000']

pos = [-0.45, -0.15, 0.15, 0.45]
color = ['#D7191C', '#2C7BB6']
folder = 'pinkbardettest'

trials = 8
label_dis = []
parameter_dis = []

def get_l2_distance(model1, model2):
    w1 = np.concatenate(list(map(lambda x: x.numpy().reshape(-1), model1.weights)))
    w2 = np.concatenate(list(map(lambda x: x.numpy().reshape(-1), model2.weights)))
    return np.linalg.norm(w1 - w2)
def get_label_disagree(arr1, arr2):
    return np.sum(arr1 != arr2) / 10000.

for e in experiments:
    preds = []
    models = []
    for trial in range(trials):
        models.append(tf.keras.models.load_model(f'{folder}/{e}{trial+1}/ckpt{experiments[e]}'))
        # preds.append(np.loadtxt(f'{folder}/{e}{trial+1}/pred{experiments[e]}.txt').astype(int))

    arr1 = []
    arr2 = []
    for i in range(trials):
        for j in range(trials):
            if j <= i:
                continue
            arr1.append(get_l2_distance(models[i], models[j]))
            # arr2.append(get_label_disagree(preds[i], preds[j]))
    parameter_dis.append(np.array(arr1))
    # label_dis.append(np.array(arr2))

# bp = plt.boxplot(label_dis, sym='', widths=0.2, notch=True, positions=np.array(range(len(xlabels))))
# plt.plot()
# plt.xticks(range(0, len(xlabels)), xlabels)
# plt.xlabel('Batch size')
# plt.ylabel('Fraction of test label different')
# plt.tight_layout()
# fig = plt.gcf()
# fig.set_size_inches(5, 7)
# plt.subplots_adjust(left=0.15, bottom=0.10, right=0.98, top=0.98)
# plt.show()
# plt.savefig('pinkbar_label.png')
# plt.cla()
# plt.clf()

bp = plt.boxplot(parameter_dis, sym='', widths=0.2, notch=True, positions=np.array(range(len(xlabels))))
plt.plot()
plt.xticks(range(0, len(xlabels)), xlabels)
plt.xlabel('Batch size')
plt.ylabel('L2 distance of model parameters')
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(5, 7)
plt.subplots_adjust(left=0.15, bottom=0.10, right=0.98, top=0.98)
plt.show()
plt.savefig('pinkbar_para.png')
plt.cla()
plt.clf()