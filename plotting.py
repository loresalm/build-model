import matplotlib.pyplot as plt
import helpers as hp
import pandas as pd
import json



fig = plt.figure(figsize=(22, 11))
gs = fig.add_gridspec(2, 4)
bottom_ax = fig.add_subplot(gs[1, :])
axes1 = []
for i in range(4):
    ax = fig.add_subplot(gs[0, i])
    axes1.append(ax)

path_df_processed = "data/AMZN_2024-01-02-18-11_2023-10-16-06-59_processed.csv"
df = hp.load_adataset(path_df_processed)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(by='time')

title_L = df["title_Length"].values
axes1[0].hist(title_L, bins=30, color='skyblue', edgecolor='black')
axes1[0].set_xlabel('title lenght')
axes1[0].set_ylabel('Frequency')

body_L = df["body_Length"].values
axes1[1].hist(body_L, bins=30, color='skyblue', edgecolor='black')
axes1[1].set_xlabel('body lenght')
axes1[1].set_xlim([0, 1000])

market = df["market_change"].values
axes1[2].hist(market, bins=30, color='skyblue', edgecolor='black')
axes1[2].set_xlabel('market change')
axes1[2].axvline(0, color='black', linestyle='--', linewidth=2)

sign = df["sign"].values
axes1[3].hist(sign, bins=2, color='skyblue', edgecolor='black', align='mid')
axes1[3].set_xticks([0.25, 0.75])
axes1[3].set_xticklabels(['-', '+'])
axes1[3].set_xlabel('market sign')

bottom_ax.plot(df['time'], df['market_change'],
               marker='.', linestyle='-', markersize=5)
bottom_ax.set_xlabel('date')
bottom_ax.set_ylabel('market change')
bottom_ax.axhline(0, color='black', linestyle='--', linewidth=2)


plt.tight_layout()
plt.savefig('plots/dataset_report.png')
plt.close()

fig = plt.figure(figsize=(12, 18))
gs = fig.add_gridspec(6, 4)
top_axPC1 = fig.add_subplot(gs[0, :3])
top_axPC2 = fig.add_subplot(gs[1, :3])

top_axUMAP1 = fig.add_subplot(gs[2, :3])
top_axUMAP2 = fig.add_subplot(gs[3, :3])

axes0 = []
for i in range(4):
    ax = fig.add_subplot(gs[i, 3])
    axes0.append(ax)
axes1 = []
for i in range(4):
    ax = fig.add_subplot(gs[4, i])
    axes1.append(ax)
axes2 = []
for i in range(4):
    ax = fig.add_subplot(gs[5, i])
    axes2.append(ax)

labels_cmap = "plasma"
sign_cmap = "bwr"
lenght_cmap = "viridis"
dot_size = 7

# plot labels over time
top_axPC1.scatter(df['time'], df['market_change'],
                  c=df['cluster_labels_PC_title'],
                  cmap=labels_cmap, s=dot_size+2)
top_axPC1.axhline(200, color='blue', linestyle='--', linewidth=4)
top_axPC1.axhline(-200, color='red', linestyle='--', linewidth=4)
top_axPC1.set_ylabel('market change')
top_axPC1.set_title("labels from title, color=cluster labels")
top_axPC2.scatter(df['time'], df['market_change'],
                  c=df['cluster_labels_PC_body'],
                  cmap=labels_cmap, s=dot_size+2)
top_axPC2.axhline(200, color='blue', linestyle='--', linewidth=4)
top_axPC2.axhline(-200, color='red', linestyle='--', linewidth=4)
top_axPC2.set_ylabel('market change')
top_axPC2.set_title("labels from body, color=cluster labels")

top_axUMAP1.scatter(df['time'], df['market_change'],
                    c=df['cluster_labels_UMAP_title'],
                    cmap=labels_cmap, s=dot_size+2)
top_axUMAP1.axhline(200, color='blue', linestyle='--', linewidth=4)
top_axUMAP1.axhline(-200, color='red', linestyle='--', linewidth=4)
top_axUMAP1.set_ylabel('market change')
top_axUMAP1.set_title("labels from title, color=cluster labels")
top_axUMAP2.scatter(df['time'], df['market_change'],
                    c=df['cluster_labels_UMAP_body'],
                    cmap=labels_cmap, s=dot_size+2)
top_axUMAP2.axhline(200, color='blue', linestyle='--', linewidth=4)
top_axUMAP2.axhline(-200, color='red', linestyle='--', linewidth=4)
top_axUMAP2.set_ylabel('market change')
top_axUMAP2.set_title("labels from body, color=cluster labels")

# sign plot
axes0[0].scatter(df['PC0_title'], df['PC1_title'],
                 cmap=sign_cmap, marker='o',
                 c=df['sign'], s=dot_size)
axes0[0].set_xlabel('PC1')
axes0[0].set_ylabel('PC2')
axes0[0].set_title("PCA on title")

axes0[1].scatter(df['PC0_body'], df['PC1_body'],
                 cmap=sign_cmap, marker='o',
                 c=df['sign'], s=dot_size)
axes0[1].set_xlabel('PC1')
axes0[1].set_ylabel('PC2')
axes0[1].set_title("PCA on title")

axes0[2].scatter(df['UMAP0_title'], df['UMAP1_title'],
                 cmap=sign_cmap, marker='o',
                 c=df['sign'], s=dot_size)
axes0[2].set_xlabel('UMAP1')
axes0[2].set_ylabel('UMAP2')
axes0[2].set_title("UMAP on title")

axes0[3].scatter(df['UMAP0_body'], df['UMAP1_body'],
                 cmap=sign_cmap, marker='o',
                 c=df['sign'], s=dot_size)
axes0[3].set_xlabel('UMAP1')
axes0[3].set_ylabel('UMAP2')
axes0[3].set_title("UMAP on body")

# label over PC
axes1[0].scatter(df['PC0_title'], df['PC1_title'],
                 cmap=labels_cmap, marker='o',
                 c=df['cluster_labels_PC_title'], s=dot_size)
axes1[0].set_xlabel('PC1')
axes1[0].set_ylabel('PC2')
axes1[0].set_title("PCA on title, color=cluster labels")
axes1[1].scatter(df['PC0_body'], df['PC1_body'],
                 cmap=labels_cmap, marker='o',
                 c=df['cluster_labels_PC_body'], s=dot_size)
axes1[1].set_xlabel('PC1')
axes1[1].set_ylabel('PC2')
axes1[1].set_title("PCA on body, color=cluster labels")
axes1[2].scatter(df['UMAP0_title'], df['UMAP1_title'],
                 cmap='plasma', marker='o',
                 c=df['cluster_labels_UMAP_title'], s=dot_size)
axes1[2].set_xlabel('UMAP1')
axes1[2].set_ylabel('UMAP2')
axes1[2].set_title("UMAP on title")
axes1[3].scatter(df['UMAP0_body'], df['UMAP1_body'],
                 cmap=labels_cmap, marker='o',
                 c=df['cluster_labels_UMAP_body'], s=dot_size)
axes1[3].set_xlabel('UMAP1')
axes1[3].set_ylabel('UMAP2')
axes1[3].set_title("UMAP on body, color=cluster labels")

# plot of lenght
axes2[0].scatter(df['PC0_title'], df['PC1_title'],
                 cmap=lenght_cmap, marker='o',
                 c=df['title_Length'], s=dot_size)
axes2[0].set_xlabel('PC1')
axes2[0].set_ylabel('PC2')
axes2[0].set_title("PCA on title, color=length")
axes2[1].scatter(df['PC0_body'], df['PC1_body'],
                 cmap=lenght_cmap, marker='o',
                 c=df['title_Length'], s=dot_size)
axes2[1].set_xlabel('PC1')
axes2[1].set_ylabel('PC2')
axes2[1].set_title("PCA on body, color=length")
axes2[2].scatter(df['UMAP0_title'], df['UMAP1_title'],
                 cmap=lenght_cmap, marker='o',
                 c=df['body_Length'], s=dot_size)
axes2[2].set_xlabel('UMAP1')
axes2[2].set_ylabel('UMAP2')
axes2[2].set_title("UMAP on title, color=length")
axes2[3].scatter(df['UMAP0_body'], df['UMAP1_body'],
                 cmap=lenght_cmap, marker='o',
                 c=df['body_Length'], s=dot_size)
axes2[3].set_xlabel('UMAP1')
axes2[3].set_ylabel('UMAP2')
axes2[3].set_title("UMAP on body, color=length")

plt.tight_layout()
plt.savefig('plots/clustering_and_reduction.png')
plt.close()


inputs_path = 'inputs.json'
with open(inputs_path, 'r') as json_file:
    inputs = json.load(json_file)
model_names = inputs["model_names"]
nb_subplot = len(model_names)*2


def plot_model_label(df, axes, name):

    for i, col in enumerate(['title', 'body']):
        ax = axes[i]
        col_name_title = f"label_{col}_{name}"

        labels = df[col_name_title].unique()

        print(labels)
        label_to_number = dict(zip(labels, list(range(len(labels)))))

        colors = df[col_name_title].map(label_to_number)

        ax[0].scatter(df['time'], df['market_change'], cmap=labels_cmap,
                      marker='o', c=colors, s=dot_size+2)
        ax[0].set_xlabel('date')
        ax[0].set_ylabel('market change')
        ax[0].axhline(0, color='black', linestyle='--', linewidth=2)
        ax[0].set_title(f"{col}, model:{name}")

        ax[1].scatter(df[f'PC0_{col}'], df[f'PC1_{col}'],
                      cmap=labels_cmap, marker='o',
                      c=colors, s=dot_size)
        ax[1].set_xlabel('PC1')
        ax[1].set_ylabel('PC2')
        ax[1].set_title(f"PCA on {col}, color=cluster labels")

        ax[2].scatter(df[f'UMAP0_{col}'], df[f'UMAP1_{col}'],
                      cmap=labels_cmap, marker='o',
                      c=colors, s=dot_size)
        ax[2].set_xlabel('UMAP1')
        ax[2].set_ylabel('UMAP2')
        ax[2].set_title(f"UMAP on {col}, color=cluster labels")


fig = plt.figure(figsize=(20, nb_subplot*5))
gs = fig.add_gridspec(nb_subplot, 4)

model_id = 0

for i in range(0, nb_subplot, 2):

    t_ax = fig.add_subplot(gs[i, :2])
    b_ax = fig.add_subplot(gs[i+1, :2])

    pc_t = fig.add_subplot(gs[i, 2])
    pc_b = fig.add_subplot(gs[i+1, 2])
    umap_t = fig.add_subplot(gs[i, 3])
    umap_b = fig.add_subplot(gs[i+1, 3])

    axes_title = [t_ax, pc_t, umap_t]
    axes_body = [b_ax, pc_b, umap_b]

    name = model_names[model_id]
    model_id += 1

    plot_model_label(df, [axes_title, axes_body], name)

plt.tight_layout()
plt.savefig('plots/labelling.png')
plt.close()




# 6 - Plotting
"""
model_name= "ahmedrachid/FinancialBERT-Sentiment-Analysis"
tk, md = hp.load_model(model_name)

hp.make_prediction(tk, md, dataset, 'title', model_name, ticker)

distance_matrix = squareform(pdist(embs, 'euclidean'))

"""


"""
# Plot each vector in its reduced form on a 2D plot
plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
            c=df['sign'].values, marker='o')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Plot of Reduced Data')

# Save the figure as a PNG file
plt.savefig('reduced_data_plot.png')

df.to_csv(path_dataset, index=False)


path_dataset = "data/AMZN_2024-01-02-18-11__2023-10-16-06-59_.csv"
path_embs = f"{path_dataset[:-4]}_embs.csv"

df = hp.load_adataset(path_dataset)
embs = np.loadtxt(path_embs, delimiter=',')

# k-means over umap

g = sns.jointplot(
    data=df,
    x="UMAP1", y="UMAP2", hue="cluster_labels",
    kind="kde",
)

plt.savefig('plots/umap.png')
plt.close()

# check label 2 and 3

df_lab2 = df[df.cluster_labels == 2].copy()
df_lab3 = df[df.cluster_labels == 3].copy()

g = sns.jointplot(
    data=df_lab2,
    x="UMAP1", y="UMAP2", hue="sign",
    kind="kde",
)
plt.title('label 2 sign')
plt.savefig('plots/umap_red_lab2.png')
plt.close()

g = sns.jointplot(
    data=df_lab3,
    x="UMAP1", y="UMAP2", hue="sign",
    kind="kde",
)
plt.title('label 3 sign')
plt.savefig('plots/umap_red_lab3.png')
plt.close()

# check sign overall

g = sns.jointplot(
    data=df_lab2,
    x="UMAP1", y="UMAP2", hue="sign",
    kind="kde",
)
plt.title('price sign over UMAP')
plt.savefig('plots/umap_sign.png')
plt.close()

plt.scatter(df['UMAP1'], df['UMAP2'], c=df['market_change'],
            cmap='viridis', marker='o')
plt.savefig('plots/umap_price_trend.png')
plt.close()


plt.scatter(df['PC1'], df['PC2'], c=df['market_change'],
            cmap='viridis', marker='o')
plt.savefig('plots/pc_price_trend.png')
plt.close()
"""