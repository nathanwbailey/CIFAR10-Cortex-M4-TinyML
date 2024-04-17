import matplotlib.pyplot as plt
acc_to_plot_val = []
acc_to_plot_train = []
with open("training.out", "r") as prune_file:
    for line in prune_file:
        line = line.rstrip()
        if 'val_accuracy' in line:
            s_idx = line.index('val_accuracy')
            acc = float(line[s_idx+14:s_idx+20])
            acc_to_plot_val.append(acc)
        if 'accuracy' in line:
            s_idx = line.index('accuracy')
            acc = float(line[s_idx+11:s_idx+17])
            acc_to_plot_train.append(acc)


plt.plot(acc_to_plot_val, color='m', label='Validation Accuracy')
plt.plot(acc_to_plot_train, color='b', label='Training Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracy.png')
