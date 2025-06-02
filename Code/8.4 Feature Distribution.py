import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------- SETTINGS ----------------
features = ['strata_race_label','strata_sex_label','geo_strata_poverty',
            'geo_strata_Segregation','geo_strata_region','geo_strata_PopDensity']

data = pd.read_csv("../Data/BigCitiesHealth_Cleaned.csv")
output_dir = "../Output/Stacked_Distributions_Cleaned"
os.makedirs(output_dir, exist_ok=True)

# ---------------- DISTRIBUTIONS (cleaned) ----------------
for main_feature in features:
    other_features = [f for f in features if f != main_feature]

    df_long = pd.DataFrame()

    for other in other_features:
        temp = data[[main_feature, other]].dropna()
        temp.columns = ['Main', 'Hue']
        temp['Feature'] = other
        df_long = pd.concat([df_long, temp], axis=0)

    # Gruppieren & Prozentanteile berechnen
    group = df_long.groupby(['Main', 'Feature', 'Hue']).size().reset_index(name='Count')
    group['Percent'] = group.groupby(['Main', 'Feature'])['Count'].transform(lambda x: x / x.sum())

    # Plot
    g = sns.catplot(
        data=group,
        kind='bar',
        x='Main',
        y='Percent',
        hue='Hue',
        col='Feature',
        palette="tab20",
        legend=True  
    )

    g.set_titles(col_template="{col_name}", size=12)
    g.set_axis_labels(main_feature, "Proportion")
    
    # X-Achsen lesbar machen
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.05)  # Max auf 1 + bisschen Platz

    handles, labels = g.axes.flatten()[0].get_legend_handles_labels()
    g.fig.legend(
        handles, labels, title="Category",
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize="small"
    )



    g.fig.subplots_adjust(top=0.9, bottom=0.15)
    g.fig.suptitle(f"Distribution of Other Features within: {main_feature}", fontsize=16, fontweight="bold")

    # Speichern
    plt.savefig(f"../Output/8.4 {main_feature} Distribution.png", bbox_inches="tight")
    #plt.show()

print("✅ Feature Distribution saved.")

