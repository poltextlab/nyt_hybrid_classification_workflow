
import pandas as pd

#################################################
# define major topic codes
#################################################

# major topic codes for loop (FOR NYT!!!)
majortopic_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 100]

#########################################################################
# create final summary analysis

data = pd.read_csv("./Data_NYT_clean_SPARK_START_sim.csv", sep=';')
data['majortopic'].loc[data['majortopic']>23] = 100
print(data.groupby(['majortopic']).count())

h_dict = {0:"sim1_and_sim2_to_sim3",1:"sim2_and_sim3_to_sim1",2:"sim1_and_sim3_to_sim2"}

for g in range(10):
    g = g+1
    for h in range(3):
        for i in range(g):
            for j in range(g):
                df = pd.read_csv(f"./ML1_workflow_on_NYT_x10_{h_dict[h]}/NYT_round1_sample{i+1}_svm{j}.csv")
                df = df.sort_values(by=['doc_id'])
                df = df.reset_index(drop=True)
                #print(df.head())
                if i == 0 and j == 0:
                    df_idf = df
                else:
                    df_lemma = df_idf.iloc[:,1:].add(df.iloc[:,1:])
                    df_idf = pd.concat([df_idf[['doc_id']], df_lemma], axis=1)
                    #print(df_idf.head())

        for i in majortopic_codes:
            df_idf[["prediction_{i}".format(i=i)]] = df_idf[["prediction_{i}".format(i=i)]].floordiv(i)

        df_idf["max_value"] = df_idf.iloc[:,1:].max(axis = 1, numeric_only = True)
        df_idf[f"how_many_{g*g}votes"] = df_idf.iloc[:,:-1].isin([g*g]).sum(1)

        print(df_idf.shape)
        df_idf = df_idf.loc[df_idf["max_value"]==g*g]
        print(df_idf.shape)
        df_idf = df_idf.loc[df_idf[f"how_many_{g*g}votes"]==1]
        print(df_idf.shape)

        df_idf = df_idf.drop(['max_value', f'how_many_{g*g}votes'], axis=1)

        print(df_idf.head())

        for i in majortopic_codes:
            df_idf[["prediction_{i}".format(i=i)]] = df_idf[["prediction_{i}".format(i=i)]].floordiv(g*g)

        print(df_idf.head())

        for i in majortopic_codes:
            df_idf[["prediction_{i}".format(i=i)]] = df_idf[["prediction_{i}".format(i=i)]]*i


        df_idf["verdict_idf"] = df_idf.iloc[:,1:].sum(1)

        df_idf = df_idf[["doc_id", "verdict_idf"]]
        #print(df_idf)


        # merge all onto data
        df = data.merge(df_idf, how='inner', on='doc_id')
        df = df.fillna(0)
        df[["verdict_idf"]] = df[["verdict_idf"]].astype(int)
        df = df.drop(columns=['text'])

        df.to_csv(f"NYT_round1_results_x{g*g}_{h_dict[h]}.csv", index=False)
