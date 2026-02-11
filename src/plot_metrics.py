import matplotlib.pyplot as plt
import os 
import csv 
import numpy as np
import cv2
import os
import re
from matplotlib.lines import Line2D
from datetime import datetime
import os
import json
from numpy import linalg as LA
from pathlib import Path


def load_data_from_log(folder):
    log_path = os.path.join(folder, "train.log")
    log_path_eval = os.path.join(folder, "eval.log")

    results = {
        "env_step_rewards": [],
        "env_step_metrics": [],
        "episode_reward": [],
        "grad_norm": [],
        "weight_distance": [],
        "weight_magnitude": [],
        "zgr": [],
        "fzar": [],
        "srank": [],
        "env_step_eval": [],
        "eval_rewards": [],
        "eNTK_rank":[],
        "eNTK_frob":[]
    }

    required_fields = [
        "env_step",
        "episode_reward",
        "grad_norm",
        "weight_distance",
        "weight_magnitude",
        "zgr",
        "fzar",
        "srank",
        "NTK_rank",
        "NTK_frobenius"
    ]

    skipped = 0

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:

            results["env_step_rewards"].append(int(row["env_step"]))
            results["episode_reward"].append(float(row["episode_reward"]))

            if any(row[field] == "" for field in required_fields):
                skipped += 1
                continue

            try:
                results["env_step_metrics"].append(int(row["env_step"]))
                results["grad_norm"].append(float(row["grad_norm"]))
                results["weight_distance"].append(float(row["weight_distance"]))
                results["weight_magnitude"].append(float(row["weight_magnitude"]))
                results["zgr"].append(float(row["zgr"]))
                results["fzar"].append(float(row["fzar"]))
                results["srank"].append(float(row["srank"]))
                results["eNTK_rank"].append(float(row["NTK_rank"]))
                results["eNTK_frob"].append(float(row["NTK_frobenius"]))
            except ValueError:
                skipped += 1
                continue

    with open(log_path_eval, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            results["env_step_eval"].append(float(row["env_step"]))
            results["eval_rewards"].append(float(row["episode_reward"]))

    return results
            
def plot_metrics(folders,name='plot_metrics.png',labels=[]):
    all_results= []
    for folder in folders:
        result = load_data_from_log(folder)
        all_results.append(result)
        
    if len(all_results)==1:
        base_dir = os.path.join(folders[0],'fig/')
        path_save_fig = os.path.join(base_dir)
        timestamp=0
    else :
        base_dir = os.path.join('./comparison/fig/')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path_save_fig = os.path.join(base_dir, timestamp)



    # Création figure 2x2
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()  # pour indexer facilement axs[0], axs[1], ...
    cmap = plt.get_cmap("tab10")  # "tab10" a 10 couleurs distinctes, tu peux aussi utiliser "tab20" ou "viridis"
    colors = [cmap(i) for i in range(len(all_results))]
    min_len = min(len(result['env_step_metrics']) for result in all_results)
    legend_elements = []
    for id,result in enumerate(all_results):
        print("id : ",id)
        # 1️⃣ Weight magnitude & distance
        axs[0].plot(result['env_step_metrics'][:min_len], result['weight_distance'][:min_len], color=colors[id])
        axs[0].set_title('Weights Distance')
        axs[0].set_xlabel('Env steps')
        plt.setp(axs[0].get_xticklabels(), rotation=45, ha='right')  
        axs[0].set_ylabel('Value')
        axs[0].grid(True)

        # 6️⃣ Rewards
        axs[1].plot(result['env_step_metrics'][:min_len], result['weight_magnitude'][:min_len], color=colors[id])
        axs[1].set_title('Weight Magnitude')
        axs[1].set_xlabel('Env steps')
        plt.setp(axs[1].get_xticklabels(), rotation=45, ha='right')  
        axs[1].set_ylabel('Value')
        axs[1].grid(True)
        
        # 2️⃣ ZGR
        axs[2].plot(result['env_step_metrics'][:min_len], np.array(result['zgr'][:min_len]) * 100, color=colors[id])
        axs[2].set_title('Zero Gradient Ratio (ZGR)')
        axs[2].set_xlabel('Env steps')
        plt.setp(axs[2].get_xticklabels(), rotation=45, ha='right')  
        axs[2].set_ylabel('ZGR (%)')
        axs[2].grid(True)

        # 3️⃣ FZAR
        axs[3].plot(result['env_step_metrics'][:min_len], np.array(result['fzar'][:min_len]) * 100, color=colors[id])
        axs[3].set_title('Feature Zero Activation Ratio (FZAR)%')
        axs[3].set_xlabel('Env steps')
        plt.setp(axs[3].get_xticklabels(), rotation=45, ha='right')  
        axs[3].set_ylabel('FZAR (%)')
        axs[3].grid(True)

        # 4️⃣ Feature Rank
        axs[4].scatter(result['env_step_metrics'][:min_len], result['srank'][:min_len], s=7,alpha=0.6,color=colors[id])
        axs[4].set_title('Feature Rank (srank)')
        axs[4].set_xlabel('Env steps')
        plt.setp(axs[4].get_xticklabels(), rotation=45, ha='right')  
        axs[4].set_ylabel('srank')
        axs[4].grid(True)

        # 5️⃣ Gradient norm
        axs[5].plot(result['env_step_metrics'][:min_len], result['grad_norm'][:min_len], color=colors[id])
        axs[5].set_title('Gradient Norm (gn)')
        axs[5].set_xlabel('Env steps')
        plt.setp(axs[5].get_xticklabels(), rotation=45, ha='right')  
        axs[5].set_ylabel('gn')
        axs[5].grid(True)

        legend_elements.append(Line2D([0], [0], color=colors[id], lw=2, label=labels[id]))



    fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02)
    )
    plt.tight_layout()


    os.makedirs(path_save_fig,exist_ok=True)
    metadata = {
        "timestamp": timestamp,
        "figure": name,
        "models": [
            {"path": p, "label": l}
            for p, l in zip(folders, labels)
        ]
    }
    with open(os.path.join(path_save_fig, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    plt.savefig(f'{path_save_fig}/' +name,dpi=300,bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    min_len_eval = min(len(result['env_step_eval']) for result in all_results)

    for id,result in enumerate(all_results):
        ax.plot(result['env_step_rewards'][:min_len], result['episode_reward'][:min_len], color=colors[id])
        ax.plot(result['env_step_eval'][:min_len_eval], result['eval_rewards'][:min_len_eval],  color=colors[id],marker='x')
       
     
    fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02)
    )
    ax.set_title('Training and Evaluation Reward')
    ax.set_xlabel('Env steps')
    ax.set_ylabel('Reward')
    ax.grid(True)
    plt.savefig(f'{path_save_fig}/' +name+'_plot_reward.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    axs = axs.flatten() 
    for id,result in enumerate(all_results):
        axs[0].plot(result['env_step_metrics'][:min_len], result['eNTK_rank'][:min_len], color=colors[id])
        axs[0].set_title('eNTK Rank')
        axs[0].set_xlabel('Env steps')
        plt.setp(axs[0].get_xticklabels(), rotation=45, ha='right')  
        axs[0].set_ylabel('Value')
        axs[0].grid(True)

        # 6️⃣ Rewards
        axs[1].plot(result['env_step_metrics'][:min_len], result['eNTK_frob'][:min_len], color=colors[id])
        axs[1].set_title('eNTK Frobenius Norm')
        axs[1].set_xlabel('Env steps')
        plt.setp(axs[1].get_xticklabels(), rotation=45, ha='right')  
        axs[1].set_ylabel('Value')
        axs[1].grid(True)
    fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02)
    )
    plt.tight_layout()
    plt.savefig(f'{path_save_fig}/' +name+'_eNKT.png',dpi=300,bbox_inches='tight')
    plt.close()




def save_K(K,save_file,step):
    folder_save_K = os.path.join(save_file,'data')
    os.makedirs(folder_save_K,exist_ok=True)
    K_np = K.cpu().numpy()
    path_save_K = f'{folder_save_K}/K_{step}'
    np.save(path_save_K,K_np)
    return folder_save_K

def plot_K(folder_data,save_file,step):
    path_save_K = f'{folder_data}/K_{step}.npy'
    K_np = np.load(path_save_K)
    
    # K_norm = K_np / np.max(np.abs(K_np))
    K_norm = K_np
    print(LA.cond(K_norm))
    plt.figure(figsize=(6,6))
    plt.imshow(K_norm, cmap='coolwarm', aspect='auto',vmin=-1, vmax=1,)
    plt.colorbar(label='Normalized K values')
    plt.title("Heatmap of NTK / Gradient Covariance Matrix")
    plt.xlabel("Input index")
    plt.ylabel("Input index")
    plt.tight_layout()
    path_save_file = os.path.join(save_file,'fig/')
    os.makedirs(path_save_file,exist_ok=True)
    plt.savefig(f'{path_save_file}/K_{step}.png')
    
    
def cond_K(file):
    p = Path(file+'/data')
    steps={}
    for f in p.glob("*.npy"):
        K_np = np.load(f)
        c_K=LA.cond(K_np)
        end = str(f).split("_")[-1]
        step = int(end.split('.')[0])
        steps[step]=c_K
    steps_sorted = dict(sorted(steps.items()))
    xs = sorted(steps_sorted.keys())
    ys = [steps_sorted[x] for x in xs]
    plt.plot(xs, ys)
    plt.xlabel("step")
    plt.yscale('log')
    plt.ylabel("cond(eNTK)")
    path_save_fig = os.path.join(file,'fig/')
    plt.savefig(f'{path_save_fig}cond_eNTK',dpi=300,bbox_inches='tight')
    plt.close()

from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

def create_ntk_video_matplotlib(save_file, output_name='ntk_evolution.mp4', fps=1,repeat_frames=10):
    fig_path = os.path.join(save_file, 'fig/')
    files = sorted([f for f in os.listdir(fig_path) if f.startswith('K_')],
                   key=lambda x: int(re.search(r'K_(\d+)', x).group(1)))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    writer = FFMpegWriter(fps=fps)
    video_path = os.path.join(save_file, output_name)
    
    with writer.saving(fig, video_path, dpi=300):
        for file in files:
            step = int(re.search(r'K_(\d+)', file).group(1))
            img = plt.imread(os.path.join(fig_path, file))
            
            ax.clear()
            ax.imshow(img)
            ax.set_title(f'NTK at step {step}')
            ax.axis('off')
            for _ in range(repeat_frames):
                writer.grab_frame()
    
    plt.close()
    print(f"Vidéo sauvegardée : {video_path}")



    
if __name__ == '__main__' : 
    path_train = 'logs/cartpole-swingup/state/K_BASIC_TDMPC_SimNorm/1' 
    path_train_two = 'logs/cartpole-swingup/state/K_BASIC_tdmpc_bis/1'
    path_train_three = 'logs/cartpole-swingup/state/K_with_LN_enc_simnorm_500/1'
    path_train_tdmpc2 = 'logs/cartpole-swingup/state/K_tdmpc2/1'
    # path_all_train =[path_train_two,path_train,path_train_three,path_train_tdmpc2]
    # labels = ['TD-MPC','TD-MPC + SimNorm','TD-MPC + SimNorm + LN','TD-MPC2']
    
    path_1 = 'logs/pendulum-swingup/state/K_BASIC_TDMPC_SimNorm/1'
    path_2 = 'logs/pendulum-swingup/state/K_BASIC_TDMPC_SimNorm_LN_enc/1'
    paths = [path_1,path_2]
    labels = ['TD-MPC + SimNorm','TD-MPC + SimNorm + LN']
    # path_all_train =[path_train_two,path_train_three,path_train_tdmpc2]
    # labels = ['TD-MPC','TD-MPC + SimNorm + LN','TD-MPC2']
    plot_metrics(paths,name="comparison_TD-MPC_all_and_2",labels=labels)
    
    
    # save_file = 'logs/cartpole-swingup/state/K_BASIC_TDMPC/1'
    # create_ntk_video_matplotlib(save_file, output_name='ntk_evolution.mp4', fps=5)
    # for path in path_all_train:
    #     cond_K(path)