import pandas as pd

df = pd.read_excel('data.xlsx')

index  = 0
# # print(df)
# for i in range(0, 30):
#     data = df.iloc[index:index + 8, :5]

#     ft = data.iloc[:, -2].str.split('-').str[0].astype(int) + data.iloc[:, -1].str.split('-').str[0].astype(int)
#     ht = data.iloc[:, -2].str.split('-').str[1].astype(int) + data.iloc[:, -1].str.split('-').str[1].astype(int)
#     data.rename(columns={data.columns[0]: f'{data.columns[0]}_Day{i + 1}'}, inplace=True)
#     data['score'] = ft.astype(str) + '-' + ht.astype(str)
#     print(data)
#     index += 9

data = df.iloc[index:index + 8, :5]
ft = data.iloc[:, -2].str.split('-').str[0].astype(int) + data.iloc[:, -1].str.split('-').str[0].astype(int)
ht = data.iloc[:, -2].str.split('-').str[1].astype(int) + data.iloc[:, -1].str.split('-').str[1].astype(int)
data.rename(columns={data.columns[0]: f'{data.columns[0]}_Day{0 + 1}'}, inplace=True)
data['score'] = ft.astype(str) + '-' + ht.astype(str)
print(data)

# data = data.drop(['FT-Score', 'HT-Score'], axis=1)

# data['New Column2'] = data['New Column'].astype(str) + '-' + 

