import pandas as pd

def data_load_proc(filepath):
    data=pd.read_csv('loan.csv')
    data = data.drop(['Unnamed: 0', 'funded_amnt'], axis=1)  # drop the first column
    data['mths_since_last_delinq'].fillna(0, inplace=True)  # There is not record so it is 0
    s = data.count(axis=1)
    data = data.drop(s[s < 20].index)  # drop the row which has lots of nan

    def todigit(_str):
        # use to make emp_length to a number
        # n/a is considered donnot have been employed
        # regard < 1 years as 1 year
        if (_str == 'n/a'):
            return int(0)
        str_ = ''
        for i in _str:
            if (i >= '0' and i <= '9'):
                str_ += i
        return int(str_)

    # make string data to digital data
    data['emp_length'] = data['emp_length'].apply(todigit)
    term_addr = lambda str: int(str.strip()[:2])
    data['term'] = data['term'].apply(term_addr)
    int_rate_addr = lambda str: float(str.strip()[:-1])
    data['int_rate'] = data['int_rate'].apply(int_rate_addr)

    new = data['total_rec_prncp'] + data['total_rec_int'] - data['loan_amnt']
    data['profit_or_loss'] = new.values
    return data