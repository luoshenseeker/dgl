import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
# from gan_attack_implement import get_filename

# songTi = matplotlib.font_manager.FontProperties(fname='C:\\Users\\luoshenseeker\\home\\work\\科研\\new/AIJack/simsun.ttc')
# plt.xticks(fontproperties=songTi,fontsize=12)
# plt.yticks(fontproperties=songTi,fontsize=12)
# plt.xlabel('x',fontproperties=songTi,fontsize=14)
# plt.ylabel('y',fontproperties=songTi,fontsize=14)
# plt.legend(prop=songTi,fontsize=12)

matplotlib.rcParams['font.family'] = 'sans-serif'  
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'# 中文设置成宋体，除此之外的字体设置成New Roman  
# matplotlib.rcParams['text.usetex'] = True

plot_format = "png"

def read_pkl_origin(filename, mode, client_num=1):
    np.set_printoptions(threshold=np.inf)   # 解决显示不完全问题

    # filename = "C:\\Users\\luoshenseeker\\home\\work\\科研\\new/Labs-Federated-Learning/data/NIID-Bench-origin/saved_exp_info/acc/" + filename
    # print('!origin!')
    filename = os.path.join(os.getcwd(), f"output/{mode}/") + filename
    print('!old!')

    fr=open(filename,'rb')

    acc_hist = pickle.load(fr)


    # 新版需用
    # server_acc = []
    # # server_loss = np.dot(weights, loss_hist[i + 1])
    # for i in range(len(acc_hist)):
    #     n_samples = np.array([200 for _ in range(client_num)])
    #     weights = n_samples / np.sum(n_samples)   # sample size为聚合权重
    #     if np.dot(weights, acc_hist[i]) != 0.0:
    #         server_acc.append(np.dot(weights, acc_hist[i]))

    return np.array(acc_hist)

def plot_acc_with_order(exp_name: str, pkl_file: list, legends = [], mode="acc", line_type="smooth", save=True, plot_range=(0, 200), client_num=[]):
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    # exp_name = get_exp_name(pkl_file[0])

    if pkl_file[0][:5] == "MNIST" or pkl_file[0][:6] == "FMNIST":
        start = 0
        end = 200
        op = 5
        stp = 5
    elif pkl_file[0][:5] == "CIFAR":
        start = 0
        end = 800
        op = 10
        stp = 10
    else:
        start = 0
        end = 200
        op = 5
        stp = 5
    start = plot_range[0]
    end = plot_range[1]

    y = {}
    meanop = {}
    meanop_5 = {}
    colors = ['#ff851b','#3d6daa','#c04851','#806d9e','#66a9c9',   '#ff851b','#3d6daa','#c04851','#806d9e','#66a9c9']  # 黄 浅蓝 红 紫 深蓝 alg1
    # colors = ['#ff851b','#806d9e','#c04851','#3d6daa','#66a9c9']  # 黄 紫 红 浅蓝 深蓝 md

    acc = pd.DataFrame()

    n = len(pkl_file)
    if not client_num:
        client_num = [2 for _ in range(n)]
    for k in range(n):
        y[k] = read_pkl_origin(pkl_file[k], mode=mode, client_num=client_num[k])
        acc[k] = y[k][start:end]
        print(round(acc[k], 2).tolist())
        if line_type == "smooth":
            last_10 = np.array(round(acc[k][end-9:end+1], 3))
            avg = round(sum(last_10) / len(last_10), 3)
            fluctuation = round((max(last_10) - min(last_10)) / 2, 3)

            print(f"μ{pkl_file[k][-9:-4]}: {avg} ±{fluctuation}")

            # 出图用
            meanop[k]=acc[k].rolling(op).mean()  # 每10个算一个平均数，共200个平均数，前9个为0
            stdop1=acc[k].rolling(op).std()  # 每10个算一个标准差，共200个方差，前9个为0
            meanop_5[k] = [meanop[k][i] for i in range(stp-1,len(meanop[k]),stp)]  # 每10个值标一个点，共20个点
            plt.plot(range(stp, end + 1, stp),
                    meanop_5[k],
                    # color=colors[k]
                    )
            plt.fill_between(range(start + 1, end + 1),
                            meanop[k] - 1.44 * stdop1,
                            meanop[k] + 1.44 * stdop1,
                            # color=colors[k],
                            alpha=0.35)
        else:
            if end > len(acc[k]):
                end = len(acc[k])
            plt.plot(range(start+1, end+1), acc[k])

    if not legends:
        if n == 4:
            plt.legend(labels=["Random", "Importance", "Cluster", "Ours"])
        elif n == 3:
            # plt.legend(loc='lower right', labels=["random_sampling", "cluster_sampling", "ours"])
            plt.legend(loc='lower right',labels=["random_sampling", "importance_sampling", "ours"])
            # plt.legend(loc='lower right',labels=["random_sampling", "importance_sampling", "ours"])
    else:
        plt.legend(labels=legends, fontsize=10.5)

    # filename = exp_name+'_acc'
    plt.xlim([start, end])  #设置x轴显示的范围
    plt.grid()
    plt.xlabel('训练轮数', {'size':10.5})
    if mode == "mape":
        plt.title(exp_name+'MAPE', {'size':10.5})
        # plt.ylabel('全局模型', {'size':10.5})
    elif mode == "r2":
        plt.title(exp_name+'$R^2$', {'size':10.5})
        # plt.ylabel('全局模型', {'size':10.5})
    # if mode == "acc":
    #     plt.ylabel('全局模型准确率', {'size':10.5})
    # elif mode == "loss":
    #     plt.ylabel('全局模型损失', {'size':10.5})
    # plt.title(exp_name, {'size':10.5})
    # # plt.title('MNIST Non-iid p=1', {'size':18})  # title的大小设置为18
    if save:
        plt.savefig(os.path.join(os.getcwd(), f'output/plot_result/{exp_name}_{mode}.{plot_format}', format=plot_format, dpi=600, bbox_inches="tight"))
    # plt.show()

        print("Saved")
    # plt.clf()

if __name__ == "__main__":
    matplotlib.rcParams['font.family'] = 'sans-serif'  
    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'# 中文设置成宋体，除此之外的字体设置成New Roman  
    # plt.figure().set_size_inches(7,3.5)
    # file_list = [
    #     "normal_up0.05_gepoch5_np_noattack.pkl",
    #     "normal_up0.1_gepoch5_np_noattack.pkl",
    #     "normal_up0.5_gepoch5_np_noattack.pkl",
    #     "normal_up1.0_gepoch5_np_noattack.pkl",
    # ]
    # legends = ["$\\theta_u$=0.05", "$\\theta_u$=0.1", "$\\theta_u$=0.5", "$\\theta_u$=1"]
    # exp_name = "MNIST 无GAN攻击"
    # plt.subplot(121)
    # plot_acc_with_order(exp_name, file_list,
    #     legends,
    #     mode="acc",
    #     save=False
    # )
    # plt.subplot(122)
    # plot_acc_with_order(exp_name, file_list,
    #     legends,
    #     mode="loss",
    #     save=False
    # )
    # plt.savefig(f'C:\\Users\\luoshenseeker\\home\\work\\科研\\new/AIJack/output/plot_result/{exp_name}.{plot_format}', format=plot_format, dpi=600, bbox_inches="tight")
    # plt.clf()


    plt.figure().set_size_inches(7,10)
    legends = ["$\\theta_u$=0.01", 
        # "$\\theta_u$=0.05", 
        "$\\theta_u$=0.1", 
        # "$\\theta_u$=0.5", 
        "$\\theta_u$=1"]
    file_name = "hgt_result"
    exp_name = "HGT result $lr=0.1$ "
    file_list = [
        "lr0.1_n200.pkl",
    ]
    plt.subplot(121)
    plot_acc_with_order(exp_name, file_list,
        legends,
        mode="mape",
        save=False
    )
    plt.subplot(122)
    plot_acc_with_order(exp_name, file_list,
        legends,
        mode="r2",
        save=False
    )
    plt.savefig(os.path.join(os.getcwd(), f'output/plot_result/{file_name}.{plot_format}', format=plot_format, dpi=600, bbox_inches="tight"))
    plt.clf()