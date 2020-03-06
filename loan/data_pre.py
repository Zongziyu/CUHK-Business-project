import pandas as pd


need_col = ['loan_amnt','int_rate','installment','emp_length','annual_inc','dti','delinq_2yrs','mths_since_last_delinq','open_acc','revol_bal','own_house','mortgage_house','total_pymnt_']
addr_map = {'WA':'West','OR':'West','ID':'West','MT':'West','WY':'West','CA':'West','NV':'West','UT':'West','CO':'West','AZ':'West','NM':'West','OK':'South','AR':'South','TX':'South',
            'LA':'South','MS':'South','TN':'South','AL':'South','GA':'South','FL':'South','SC':'South','KY':'South','WV':'South','VA':'South','NC':'South','ME':'Northeast','VT':'Northeast',
            'NY':'Northeast','PA':'Northeast','NH':'Northeast','MA':'Northeast','RI':'Northeast','CT':'Northeast','NJ':'Northeast','DE':'Northeast','MD':'Northeast','ND':'Midwest',
            'SD':'Midwest','NE':'Midwest','KS':'Midwest','MN':'Midwest','IA':'Midwest','MO':'Midwest','WI':'Midwest','IN':'Midwest','OH':'Midwest'}
def todigit(_str):
    # use to make emp_length to a number
    # n/a is considered donnot have been employed
    # regard < 1 years as 1 year
    if (_str == 'n/a'):
        return int(0)
    str_ = ''
    for i in (str)(_str):
        if (i >= '0' and i <= '9'):
            str_ += i
    return int(str_)

def to0_1(str):
    if str is True:
        return 1
    else :
        return 0

def scale_(data, col):
    x_min, x_max = data[col].min(), data[col].max()
    tmp = (data[col] - x_min) / (x_max - x_min)
    return tmp

def data_load_proc(filepath):
    data=pd.read_csv(filepath)
    data = data.drop(['Unnamed: 0', 'funded_amnt'], axis=1)  # drop the first column
    data['mths_since_last_delinq'].fillna(0, inplace=True)  # There is not record so it is 0
    data['emp_length'].fillna(0, inplace=True)
    s = data.count(axis=1)
    data = data.drop(s[s < 20].index)  # drop the row which has lots of nan

    # make string data to digital data
    data['emp_length'] = data['emp_length'].apply(todigit)
    term_addr = lambda str: int(str.strip()[:2])
    data['term'] = data['term'].apply(term_addr)
    int_rate_addr = lambda str: float(str.strip()[:-1])
    data['int_rate'] = data['int_rate'].apply(int_rate_addr)

    new = data['total_rec_prncp'] + data['total_rec_int'] - data['loan_amnt']
    data['profit_or_loss'] = new.values

    data['total_pymnt_'] = data['total_pymnt']
    data['own_house'] = data['home_ownership']=='OWN'
    data['mortgage_house'] = data['home_ownership']=='MORTGAGE'
    data['own_house'] = data['own_house'].apply(to0_1)
    data['mortgage_house']=data['mortgage_house'].apply(to0_1)
    data['addr_state'] = data['addr_state'].map(addr_map)


    for i in need_col:
        data[i] = scale_(data,i)

    data['Condition'] = 0.592*data['revol_bal'] - 0.901*data['loan_amnt'] - 0.895*data['installment']
    data['Capacity'] = 0.626*data['annual_inc'] - 0.521*data['int_rate'] - 0.404*data['dti']
    data['Capital'] = 0.583*data['revol_bal'] - 0.756*data['dti'] - 0.626*data['open_acc']
    data['Character'] = 0.599*data['total_pymnt_'] - 0.504*data['delinq_2yrs'] - 0.391*data['mths_since_last_delinq']
    data['Collateral'] = 0.455*data['own_house'] + 0.332*data['emp_length'] - 0.262*data['mortgage_house']

    return data