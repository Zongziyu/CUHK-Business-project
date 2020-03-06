from data_pre import *
import numpy as np
from learn import change_columns
from plot_figures import *
np.set_printoptions(suppress=True, threshold=1e4, linewidth=400)
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# matplotlib.style.use('ggplot')
data = data_load_proc('loan.csv')

print(data['addr_state'].value_counts())

change_columns = change_columns + ['profit_or_loss','addr_state'] + ['home_ownership']

data = data[change_columns] ### need to scale


# plotHeatmap(data)

# print(data[change_columns+['profit_or_loss']])

loan_status_dict_2 = {'Fully Paid':1, 'Current':1, 'Charged Off':-1, 'Late (31-120 days)':-1, 'In Grace Period':-1, 'Late (16-30 days)':-1, 'Default':-1}
data['loan_status'] = data['loan_status'].map(loan_status_dict_2)
plot5c(data,change_columns)
data.to_csv("loan_pre__.csv")

# data['own_house'] = data['home_ownership']=='OWN'
# print(data['own_house'])

