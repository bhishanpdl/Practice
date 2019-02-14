
Table of Contents
=================
   * [Pandas Caveats and Limitations](#pandas-caveats-and-limitations)
   * [Speed and Memory](#speed-and-memory)
   * [Useful Commands](#useful-commands)
   * [Efficient sub-selection](#efficient-sub-selection)
   * [Pandas <strong>apply</strong>](#pandas-apply)
   * [Pandas <strong>where</strong>](#pandas-where)
   * [Pandas Series <strong>extract</strong>](#pandas-series-extract)
   * [Pandas assign **pop str extract](#pandas-assign-pop-str-extract)
   * [Pandas <strong>map</strong>](#pandas-map)
   * [Pandas <strong>rename</strong>](#pandas-rename)
   * [Pandas <strong>groupby</strong>](#pandas-groupby)
   * [Pandas <strong>cut</strong>](#pandas-cut)
   * [Pandas <strong>qcut</strong> and <strong>pivot_table</strong>](#pandas-qcut-and-pivot_table)
   * [Pandas <strong>stack</strong>](#pandas-stack)
   * [Pandas <strong>unstack</strong>](#pandas-unstack)
   * [Pandas <strong>pivot</strong>](#pandas-pivot)
   * [Pandas <strong>melt</strong>](#pandas-melt)
   * [Pandas <strong>wide_to_long</strong>](#pandas-wide_to_long)
   * [Pandas <strong>read_csv</strong> with lambda usecols](#pandas-read_csv-with-lambda-usecols)
   * [Pandas <strong>MultiIndex</strong>](#pandas-multiindex)
   * [Pandas <strong>eval</strong>](#pandas-eval)
   * [Pandas <strong>query</strong>](#pandas-query)
   * [pandas.io.json.json_normalize](#pandasiojsonjson_normalize)
   * [Pandas string extract, slice](#pandas-string-extract-slice)
   * [Pandas adding date month columns](#pandas-adding-date-month-columns)
   * [Tidy up when single column contains multiple variables](#tidy-up-when-single-column-contains-multiple-variables)
   * [Tidy up when single cell contains multiple values](#tidy-up-when-single-cell-contains-multiple-values)
   * [Mathematics](#mathematics)
   * [Multiple conditions](#multiple-conditions)
   * [First and last percentage increase in values](#first-and-last-percentage-increase-in-values)
   * [New column with numeric manipulations](#new-column-with-numeric-manipulations)
   * [Weighted average after groupby](#weighted-average-after-groupby)
   * [Wrapping aggregation functions](#wrapping-aggregation-functions)
   * [One column has list of lists](#one-column-has-list-of-lists)
   * [pandas and json to get currency conversion](#pandas-and-json-to-get-currency-conversion)
   * [Unnesting column with list](#unnesting-column-with-list)
   * [Plottings](#plottings)
   * [Mixed examples](#mixed-examples)
   * [Basic Python and Numpy](#basic-python-and-numpy)
   * [Useful links](#useful-links)

# Pandas Caveats and Limitations
#==============================================================================
- Pandas operations are slow, but numpy operations are fast.
- Pandas is highly memory inefficient, it takes about 10 times RAM that of loaded data.
- To parallize pandas operation we can install and use modin.pandas

```python
# Libraries dependencies
pd.read_excel ==> needs: xlrd
pd.read_hdf ==> needs: pytables (conda install pytables, dont use pip)
pd.read_parquet ==> needs: fastparquet (conda install -n viz -c conda-forge fastparquet)

# aggregation does not support np.sqrt function
df.groupby('A')['B'].agg(np.sqrt).head()   # ValueError: function does not reduce (but np.mean works)
df.groupby('A')['B'].apply(np.sqrt).head() # works fine

# dataframe has method applymap, but not series
df.applymap(myfunc)  # works fine
ser.applymap(myfunc) # fails

# pd.wide_to_long fails with datetime column, (we need to make it str)
https://github.com/bhishanpdl/Data_Cleaning/blob/master/data_cleaning_examples/pandas_Tap4_Fe_example.ipynb
df.columns = 'Date', 'Tap4.Fe', 'Tap4.Mn', 'Tap4.Fe', 'Tap5.Mn' # here Date is datetime category
pd.wide_to_long(df,'Tap','Date','numCompound').head() # ValueError: can only call with other PeriodIndex-ed objects
df.melt(id_vars='Date') # This does not fail and gives 3 columns: Date, variable, value

# Python caveats
## caveat: a = b = c
a = b = c = [1,2,3] 
b.append(4)
print(a) gives [1,2,3,4]

## caveat: numpy dtype is fixed
a = np.array([1,2,3])
a[0] = 10.5
print(a) gives [10,2,3]  dtypes are not changed

## caveat: a+= b
a += b is not same as a = a + b
a = [1,2,3]
b = 'hello'
a = a + b # TypeError: can only concatenate list (not "str") to list
a += b
print(a) # [1, 2, 3, 'h', 'e', 'l', 'l', 'o']
```

# Speed and Memory
#==============================================================================
```python
# Tips
- Operations at/iat are faster than loc/iloc.
- When using pd.to_datetime, always use format option.
- When melting a dataframe, rename columns first, then melt.
- Numpy operations are faster than pandas operations. (eg. df['c0'].values faster than df['c0'])
- Regex operations are slow, try alternatives, e.g. new column using loc.
- When you see categorical data, always make the column dtype categorical.
- When dealing with timeseries, make the index datetime and sort the index.
- Use loc operation than apply operation. e.g. (df.loc[df['B'] >= 1000, 'B'] -= 1000, make small large values)


#------------------------------------------------------------------------
# Pandas duplicated is faster than drop_duplicates
def duplicated(df): # fastest
    return df[~   df["A"].duplicated(keep="first")  ].reset_index(drop=True)

def drop_duplicates(df): # second
    return df.drop_duplicates(subset="A", keep="first").reset_index(drop=True)

def group_by_drop(df): # last
    return df.groupby(df["A"], as_index=False, sort=False).first()
    
# Pandas apply is slower than transformed
def f_apply(df):
    return df.groupby('name')['value2'].apply(lambda x: (x-x.mean())/x.std())

def f_unwrap(df):
    g = df.groupby('name')['value2']
    v = df['value2']
    return (v-g.transform(np.mean))/g.transform(np.std)

# map uses very less memory than apply
df = pd.DataFrame({'A': [1, 1, 2,2,1, 5]})
df['B'] = df.apply(lambda row: 1 if row['A'] == 1 else 0, axis=1)
df['C'] = df['A'].map({1:1, 2:0}).fillna(value=0).astype(int)


# Numpy is faster than pandas
pandas: ((df['col1'] * df['col2']).sum()) # slower
numpy: (np.sum((df['col1'].values * df['col2'].values)))  # fastest
numpy: (np.nansum((df['col1'].values * df['col2'].values))) # use this if you have nans

# drop_duplicates is faster than groupby
# 1.15 ms
df.drop_duplicates('Group',keep='last').\
           assign(Flag=lambda x : x['string'].str.contains('Search',case=False))

# groupby 1.60 ms
df.groupby("Group")["string"] \
  .apply(lambda x: int("search" in x.values[-1].lower())) \
  .reset_index(name="Flag")

#------------------------------------------------------------------------
# using numba
import numba
@numba.vectorize
def double_every_value_withnumba(x):
    return x*2

%timeit df['col1_doubled'] = df.a*2  # 233 us
%timeit df['col1_doubled'] = double_every_value_withnumba(df.a.values) # 145 us
```

# Useful Commands
#==============================================================================
```python

## Remember
- axis=0:: default is along each columns. eg. np.mean(arr) is mean for each columns.
- axis=1:: is used while doing operations on groupby, dropna etc.
- groupby:: grouped = df.groupby('id')['value']; for name, group in grouped: print(name,group) 
- groupby:: df.groupby('c0')['c1'].apply(list)
- groupby:: df.groupby(['c0','c1'],as_index=False)['c2'].mean()
- groupby/apply:: acts on the whole dataframe
- groupby/agg:: acts on column and gives one value per group
- groupby/transform:: acts on column and gives values for all rows
- get_dummies:: Pandas one hot encoding is called get_dummies
- rename:: we can use .rename(columns={'old': 'new'})
- reset_index:: we can rename before reset and also use reset_index([1,2]) to not reset first multiindex level
- reset_index:: reset_index(name='newname') is not in the documentation but works and gives no warning.
- stack:: stack, rename_axis, reset_index()
- apply:: df[['year', 'month', 'day']].apply( lambda row: '{:4d}-{:02d}-{:02d}'.format(*row),axis=1)
- concat:: pd.concat([s1, df1], axis='columns', ignore_index=True) # keys = ['s1', 'df1'] gives hierarchial frame
- series :: s.values.tolist()  # if s has elements as list
- series.str.contains:: case=True and regex=False are defaults.
  
## Remember some methods
filter: df.filter(regex='e$', axis=1) # all columns ending with letter e
query: df.query('a==1 and  b==4 and not c == 8')
stack: pd.DataFrame({'c0' : [2,3],'c1' : ['A', 'B']}).stack() # gives only one series with multi-index
get_dummies: pd.get_dummies(pd.Series(list('abcaa')), drop_first=True) # only two columns of b and c with 0 and 1s.
cut: pd.cut(df['A'], bins=bins, labels=labels,include_lowest=True,right=False)

** elements of df.make will be index, and elements of df.num_doors will be new columns
crosstab: pd.crosstab(df.make, df.num_doors, margins=True, margins_name="Total",aggfunc='count')

** Columns B and C will be melted and disappeared from columns.
** New column 'variable' will have names only B and C.
** New column 'value' will have values from old columns B and C.
** Column A has only its elements.
melt: pd.melt(df, id_vars=['A'], value_vars=['B','C'],var_name='variable', value_name='value')

** columns A2019, B2020 have stubnames A and B. New column 'Year' will be created with values from stubnamed cols.
wide_to_long: pd.wide_to_long(df, stubnames=['A', 'B'], i=['Untouched', 'Columns', j='Year').reset_index()

** elements of columns A,B will be multiple-index, elements of column C will be column names,
** the columns will have elements values from values of D column.
pivot_table: pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum)
  
## data slicing
df[0] # first COLUMN not row (even if both index and column names are 0's)
df['col_0'] # df.col_0 works but df.0 does not work, so,  always use bracket notation.
df['a': 'c'] # FAILS  (use: df.loc[:, 'a': 'c'])
df.loc[:, df.columns.isin(list('aed'))] # columns are automatically sorted
df[['col1','col2']]
df[df.columns[2:4]]
df.loc[0] # first ROW
df.iloc[:, [1,3]]
df.loc['row_a':'row_p', 'col_a':'col_c']
df.iloc[:, np.r_[0,3,5:8]]
df = df.sort_index()  # its good to sort the index, if they are not sorted

## data slicing more advanced
## NOTE: Use .loc instead of query for data <15k rows.
## https://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-query
df.loc[df['IQ'] > 90] # df.query('IQ > 90')
df.loc[df['country'].isin(['Nepal','Japan']) # df.query(' ["Nepal", "Japan"] in country')
df.loc[df['country'] == 'Nepal'] # df.query(' country == "Nepal" ')
df.loc[df['country'].str.contains('United')] # United States, United Kingdom etc.
df[(df.a == 1) & (df.b == 4) & ~(df.c == 8)]  # both ~ and not are same in query.
df.query(' (a==1) and  (b==4) and (not c == 8)') # both & and and are same in query.
df.query('a==1 and  b==4 and not c == 8')
df.query(" c0 < c1 < c2 and c3 != 500 ")

## use loc to assign values, df[mask] = something gives warning.
df[df['Affiliation'].str.contains('Harvard')]['Affiliation'] = 'Harvard University' # BAD
df.loc[ df['Affiliation'].str.contains('Harvard',case=False), 'Affiliation'] = 'Harvard University' # GOOD

## multiple conditions
df['young_male'] = ((df.Sex == 'male') & (df.Age < 30)).map({True: 1, False:0})

## iat/at are faster than iloc/loc
df.at['index_name']  # faster than df.loc['index_name']
df.iat[index_number] # df.iloc[index_number]
df.iat[0,2] # faster than df.iloc[0,2]

## faster sub-selection
row_num = df.index.get_loc('my_index_name')
col_num = df.columns.get_loc('my_column_name')
df.iat[row_num,col_num]

## select using dtypes or substrings
df.select_dtypes(include=['int']).head()
df.select_dtypes(include=['number']).head()
df.filter(like='facebook').head()  # all columns having substring facebook
df.filter(regex='\d').head() # column names with digits on them

## select using lookup
df.lookup([3.0,4.0,'hello'], ['c0','c1','c2'])

## dropping range of columns
df = df.drop(df.columns.to_series()["1960":"1999"], axis=1) # inclusive last value

## mapping values
s = pd.Series([1,2,3], index=['one', 'two', 'three'])
map1 = {1: 'A', 2: 'B', 3: 'C'}
s.map(map1) # this will change values from 1,2,3 to A,B,C

## number of notnull elements in a column
num_col2_notnull = pd.notnull(df['col2']).sum()

## Change Y to 1 and N to 0
df['yesno'] = df['yesno'].eq('Y').astype(int)

## Number of notnull numbers after 3rd column
df.iloc[:,3:].notnull().sum(1)

## filter examples
note: select num_items, num_sold but not price
df['mean_row'] = df.filter(like='num_').mean(axis=1,skipna=True)
df1 = df.filter(items=['col0','col1'])
df1 = df.filter(regex='e$', axis=1) # col_one, col_three but not col_two

## some statistics
df.describe(include=[np.number]).T
df.describe(include=[np.number],percentiles=[.01, .05, .10, .25, .5, .75, .9, .95, .99]).T
df.describe(include=[np.object, pd.Categorical]).T
df['col_5'].dropna().gt(120).mean()

## some methods
df.nlargest(10, 'col_5').head() # gives all columns, but sorts by col_5
df.nlargest(100, 'imdb_score').nsmallest(5, 'budget')
```

# Efficient sub-selection
#==============================================================================
```python
df = df.set_index('A').sort_index() # always set index and sort them.
df.loc['myrow'] # faster
df[df['A'] == 'myrow'] # slower

### we can also create index from combining multiple columns
df.index = df['A'] + ', ' + college['B']
df = df.sort_index()
df.loc['firstname, lastname']
```


# Avoid **apply** use df.iterrows
#==============================================================================
Reference: https://ys-l.github.io/posts/2015/08/28/how-not-to-use-pandas-apply/
```python
import pandas as pd
import numpy as np
%load_ext memory_profiler

def complex_computation(a):
    # Pretend that there is no way to vectorize this operation.
    return a[0]-a[1], a[0]+a[1], a[0]*a[1]

def func(row):
    v1, v2, v3 = complex_computation(row.values)
    return pd.Series({'NewColumn1': v1,
                      'NewColumn2': v2,
                      'NewColumn3': v3})

def run_apply(df):
    df_result = df.apply(func, axis=1)
    return df_result

def run_loopy(df):
    v1s, v2s, v3s = [], [], []
    for _, row in df.iterrows():
        v1, v2, v3 = complex_computation(row.values)
        v1s.append(v1)
        v2s.append(v2)
        v3s.append(v3)
    df_result = pd.DataFrame({'NewColumn1': v1s,
                              'NewColumn2': v2s,
                              'NewColumn3': v3s})
    return df_result

def make_dataset(N):
    np.random.seed(0)
    df = pd.DataFrame({
            'a': np.random.randint(0, 100, N),
            'b': np.random.randint(0, 100, N)
         })
    return df

def test():
    from pandas.util.testing import assert_frame_equal
    df = make_dataset(100)
    df_res1 = run_loopy(df)
    df_res2 = run_apply(df)
    assert_frame_equal(df_res1, df_res2)
    print 'OK'

# Testing
df = make_dataset(1000000)  
test() # OK
%memit run_loopy(df)  # peak memory: 321.32 MiB, increment: 148.74 MiB
%memit run_apply(df)  # peak memory: 3085.00 MiB, increment: 2833.09 MiB  (10 times more memory)
%timeit run_loopy(df) # 1 loop, best of 3: 41.2 s per loop
%timeit run_apply(df) # 1 loop, best of 3: 4min 12s per loop
```


# Pandas **apply**
#==============================================================================
```python
#  (apply is too slow!)
df = pd.DataFrame({'A': list('abc')*1000000, 'B': [10, 20,200]*1000000,
                  'C': [0.1,0.2,0.3]*1000000})
                  
# fastest (100ms)
for c in df.select_dtypes(include = [np.number]).columns:
    df[c] = np.log10(df[c].values)

# 3 times slower 300ms
log10_df = pd.concat([df.select_dtypes(exclude=np.number),
                      df.select_dtypes(include=np.number).apply(np.log10)],
                      axis=1)
# 6 times slower
log10_df = df.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)

## apply is slow, however, we can we it for aggregations
df = pd.DataFrame({"User": ["a", "b", "b", "c", "b", "a", "c"],
                  "Amount": [10.0, 5.0, 8.0, 10.5, 7.5, 8.0, 9],
                  'Score': [9, 1, 8, 7, 7, 6, 9]})
def my_agg(x):
    mydict = {
        'Amount_mean': x['Amount'].mean(),
        'Amount_std':  x['Amount'].std(),
        'Amount_range': x['Amount'].max() - x['Amount'].min(),
        'Score_Max':  x['Score'].max(),
        'Score_Sum': x['Score'].sum(),
        'Amount_Score_Sum': (x['Amount'] * x['Score']).sum()}

    return pd.Series(mydict, list(mydict.keys()))

df.groupby('User').apply(my_agg) # has columns 'Amount_mean', 'Amount_std', ...

# apply with multiple arguments
s = pd.Series(np.random.randint(0,10, 10))
def between(x, low, high):
    return x >= low and x =< high
s.apply(between, args=(3,6))
s.apply(between, low=3, high=6)
s.apply(between, args=(3,), high=6)


```

# Pandas **where**
#==============================================================================
```python
df.where(mask, -df) # mask = df % 3 == 0  # divisibles of 3 are positive (eg. 0, 3, 6, 9)
df.where(lambda x: x > 4, lambda x: x + 10) # if value is larger than 4, add 10 to it.
```

# Pandas Series **extract**
#==============================================================================
```python
s = pd.Series(['M35 35-39', 'M40 40-44', 'M45 45-49', 'M50 50-54', 'M55 55-59',
       'M60 60-64', 'M65 65-69', 'M70 70-74', 'M75 75-79', 'M80 80+'])
str.extract('(?P<Sex>[MF])\d\d\s(?P<AgeGroup>\d{2}[-+](?:\d{2})?)', expand=True)

# series split expand
df = pd.DataFrame({'c0': [' a b c ', 'd e f', 'g h i ']})
df[['c1','c2','c3']] = df['c0'].str.split(expand=True) # extra whitespaces are removed
df = df.drop('c0',1)
```

# Pandas assign **pop str extract
#==============================================================================
```python
df = pd.DataFrame({'BP': ['100/80'],'Sex': ['M']})

# using str split
df[['sys','dias']] = df['BP'].str.split(pat='/', expand=True)

# using str extract
df[['sys','dias']] = df['BP'].str.extract(r'(\d+)/(\d+)',expand=True)

# using one-liner
df.drop('BP', 1).join(
    df['BP'].str.split('/', expand=True)
            .set_axis(['BPS', 'BPD'], axis=1, inplace=False)
            .astype(float))

# using assign **pop extract
df2 = (df.drop('BP',axis=1)
       .assign(BPS =  lambda x: df.BP.str.extract('(?P<BPS>\d+)/').astype(float))
       .assign(BPD =  lambda x: df.BP.str.extract('/(?P<BPD>\d+)').astype(float)))
# another method
df = df.assign(**df.pop('BP').str.extract(r'(?P<BPS>\d+)/(?P<BPD>\d+)',expand=True).astype(float))
```

# Pandas **map**
#==============================================================================
```python
s = pd.Series([14,1524,2534,3544,65])
age_map = {
    14: '0-14',
    1524: '15-24',
    2534: '25-34',
    3544: '35-44',
    4554: '45-54',
    5564: '55-64',
    65: '65+'
}
s.map(age_map)

# using regex
s = s.astype(str).str.replace(r'14', r'0-14',regex=True)
                 .str.replace(r'65', r'65+',regex=True)
                 .str.replace(r'(\d\d)(\d\d)', r'\1-\2',regex=True))
```

# Pandas **rename**
#==============================================================================
```python
import re

Columns:
['movie_title', 'actor_1_name', 'actor_2_name', 'actor_3_name',
       'actor_1_facebook_likes', 'actor_2_facebook_likes',
       'actor_3_facebook_likes']
    
actor_1_name ==> actor_1  and actor_1_facebook_likes ==> actor_facebook_likes_1
df.rename(columns=lambda x: x.replace('_name','') if '_name' in x 
               else re.sub(r'(actor_)(\d)_(facebook_likes)', r'\1\3_\2',x) if 'facebook' in x 
               else x)
               
# now we can tidy up the dataframe from wide to long
stubs = ['actor', 'actor_facebook_likes']
df = pd.wide_to_long(df, 
                       stubnames=stubs, # columns will become rows
                       i=['movie_title'], # keep this untouched
                       j='actor_num',  # new name
                       sep='_').reset_index()
```

# Pandas **groupby**
#==============================================================================
```python
# groupby attributes
print([attr for attr in dir(pd.core.groupby.groupby.DataFrameGroupBy) if not attr.startswith('_') ])
print([attr for attr in dir(pd.core.groupby.groupby.DataFrameGroupBy) if attr[0].islower() ])

['agg', 'aggregate', 'all', 'any', 'apply', 'backfill', 'bfill', 'boxplot', 'corr', 'corrwith', 'count', 'cov', 'cumcount', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'dtypes', 'expanding', 'ffill', 'fillna', 'filter', 'first', 'get_group', 'groups', 'head', 'hist', 'idxmax', 'idxmin', 'indices', 'last', 'mad', 'max', 'mean', 'median', 'min', 'ndim', 'ngroup', 'ngroups', 'nth', 'nunique', 'ohlc', 'pad', 'pct_change', 'pipe', 'plot', 'prod', 'quantile', 'rank', 'resample', 'rolling', 'sem', 'shift', 'size', 'skew', 'std', 'sum', 'tail', 'take', 'transform', 'tshift', 'var']

## example
#******************************************************************************
df = pd.DataFrame({'A': [1, 1, 1, 2, 2],
                   'B': [1, 1, 2, 2, 1],
                   'C': [10, 20, 30, 40, 50],
                   'D': ['X', 'Y', 'X', 'Y', 'Y']})

# mean, sum, size, count, std, var, describe, first, last, nth, min, max
# agg function: sem (standard error of mean of groups)
df.groupby(‘A’)[‘B’].count() # gives count of non-NaNs (use .size() to count NaNs)
df.groupby(‘A’)[‘B’].sum()   # gives series
df.groupby(‘A’)[[‘B’]].sum() # gives dataframe 
df.groupby(‘A’, as_index=False)[‘B’].sum() # does not set index
df.groupby(‘A’, as_index=False).agg({‘B’: ‘sum’}) # gives columns A and B
df.groupby(‘A’)[‘B’].agg(lambda x: (x - x.mean()) / x.std() ) # zscore
df.groupby(‘D’).get_group(‘X’)                
df.groupby(‘A’).filter(lambda x: x > 1)
df.groupby(‘A’).describe()
df.groupby(‘A’).apply(lambda x: x *2)
df.groupby(‘A’).expanding().sum()
df.groupby('A')['B'].sum() # two rows with sum 6 and 9 (note cumsum() gives 5 rows)
df.groupby('A')['B'].transform('sum') # 5 rows, transform keeps same dimension.
df.transform({'A': np.sum, 'B': [lambda x : x+1, 'sqrt']})

## add prefix
df.groupby('A').mean().add_prefix('mean_') # gives mean_B and mean_C, D is ignored.
df.groupby('A').max().add_suffix('_max')   # note: prefix, suffix accept ONLY ONE string.

(df.groupby('A')
    .agg({'B': 'sum', 'C': 'min'})
    .rename(columns={'B': 'B_sum', 'C': 'C_min'})
    .reset_index()
)

## multiple aggregation
df = pd.DataFrame({'A': [1, 1, 1, 2, 2],
                   'B': [1, 2, 3, 2, 1],
                   'C': [10, 20, 30, 40, 50]})

### using reset to drop one level of multi-index
g = (df.groupby('A') # agg gives here two rows of columns -- C,B and mean,count min,max and index A
    .agg({'B': ['min', 'max'], 'C': ['mean','count']})
    .reset_index()) # now index A, becomes first column with column name A, and its second level column name is empty.
    
### using as_index false to drop one level of multi-index (same as .reset_index())
g = (df.groupby('A',as_index=False)
    .agg({'B': ['min', 'max'], 'C': ['mean','count']}))

#### rename columns
### In above multiple aggregation examples, we get two levels of columns. (after reset or as_index = False)
### To make only one column name, we can use list comprehension.
### note: When we reset and make the index A, as column, it does not have second level column name.
g.columns = ['_'.join(x).strip() if x[1] else x[0] for x in g.columns.ravel()]

## Groupby with custom function
#******************************************************************************
df = pd.DataFrame({'Name': list('ABCAB'),'Score': [20,40,80,70,90]})
def get_stats(group):
    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
bins = [0, 25, 50, 75, 100]
group_names = ['Low', 'Okay', 'Good', 'Great']
df['categories'] = pd.cut(df['Score'], bins, labels=group_names)
df['Score'].groupby(df['categories']).apply(get_stats).unstack()

## Groupby with pd.Grouper
#******************************************************************************
df.groupby([pd.Grouper(freq='1M',key='Date'),'Buyer']).sum()

## time series
#******************************************************************************
df = pd.DataFrame({'date': pd.date_range(start='2016-01-01',periods=4,freq='W'),
                   'group': [1, 1, 2, 2],
                   'val': [5, 6, 7, 8]}).set_index('date') # only 4 rows
df.groupby('group').resample('1D').ffill() # 16 rows
        

## groupby with categorical data
pd.Series([1, 1, 1]).groupby(pd.Categorical(['a', 'a', 'a'],
                        categories=['a', 'b']), observed=False).count()

## pipe and apply
(df.groupby(['A', 'B'])
    .pipe(lambda grp: grp.C.max()) # .apply() gives same result.
    .unstack().round(2))

## using functions
def subtract_and_divide(x, sub, divide=1):
    return (x - sub) / divide

df.iloc[:,:-1].apply(subtract_and_divide, args=(5,), divide=3)

## groupby pipe (pipe is encourased to be used)
f(g(h(df), arg1=1), arg2=2, arg3=3)
(df.pipe(h)
       .pipe(g, arg1=1)
       .pipe(f, arg2=2, arg3=3)
    )
```

# Pandas **cut**
#==============================================================================
```python
df = pd.DataFrame({'A': [40,50,70,80],'B': [400,500,700,800]})
bins = [0, 40, 60, 80, np.inf]
labels=[1, 2, 3, 4] # categories (40 is in group2 not 1 since include_lowest is True)
df['A_cat'] = pd.cut(df['A'], bins=bins, labels=labels,include_lowest=True,right=False)
# 40=2 50=2 70=3 80=4

df = pd.DataFrame({'A': [40,50,70,80],'B': [400,500,700,800]})
bins = [0, 40, 60, 80, np.inf]
labels=['Fail', 'Third', 'Second', 'First']
cuts = pd.cut(df['A'], bins=bins, labels=labels,include_lowest=True,right=False)
df.groupby(cuts)['B'].count().plot.bar()
```


# Pandas **qcut** and **pivot_table**
#==============================================================================
```python
ser = pd.Series(np.random.randn(100))
factor = pd.qcut(ser, [0, .25, .5, .75, 1.])
ser.groupby(factor).mean()
```

# Pandas **stack**
#==============================================================================
```python
## example1
pd.DataFrame({'c0' : [2,3],'c1' : ['A', 'B']}).stack() # gives only one series with multi-index

## example2
Data:
         Apple  Orange  Banana
Texas       12      10      40
Arizona      9       7      12
Florida      0      14     190

This data is not TIDY, there must be variable names and values.
(df.stack()
  .rename_axis(['state','fruit'])
  .reset_index(name='weight'))
  
## example3
religion  <$10k  $10-20k  $20-30k  $30-40k  $40-50k  $50-75k
Agnostic     27       34       60       81       76      137

## using stack *********************
(df.set_index('religion')
  .stack() 
  .rename_axis(['religion','income'])
  .reset_index(name='frequency'))
  
## using melt *********************
(pd.melt(df, id_vars=['religion'], value_vars=df.columns.values[1:],
             var_name='income', value_name='frequency')
  .sort_values(by='religion')
  .head())
```

# Pandas **unstack**
#==============================================================================
```python
employee.groupby(['RACE', 'GENDER'])['BASE_SALARY'].mean().astype(int).unstack('GENDER')

# example 2
flights.groupby(['AIRLINE', 'ORG_AIR'])['CANCELLED'].sum().unstack('ORG_AIR', fill_value=0)
flights.pivot_table(index='AIRLINE', columns='ORG_AIR', values='CANCELLED', aggfunc='sum',fill_value=0).round(2)
```

# Pandas **pivot**
#==============================================================================
```python
# data: https://raw.githubusercontent.com/PacktPublishing/Pandas-Cookbook/master/data/weight_loss.csv
def find_perc_loss(s):
    return (s - s.iloc[0]) / s.iloc[0]
    
# transform data, it will reset each month and gives same number of rows as original
pcnt_loss = weight_loss.groupby(['Name','Month'])['Weight'].transform(find_perc_loss)

# new column
weight_loss['Perc Weight Loss'] = pcnt_loss.round(3)

# get week4 to find winner
week4 = weight_loss.query('Week == "Week 4"')

# make month chronological rather than alphabetical
week4['Month'] = pd.Categorical(week4['Month'],
                                categories=week4['Month'].unique(),
                                ordered=True)
                                
# find winner
winner = week4.pivot(index='Month', columns='Name',
                    values='Perc Weight Loss')

# winner column
winner['Winner'] = np.where(winner['Amy'] < winner['Bob'], 'Amy', 'Bob')

# sytle hightlight
winner.style.highlight_min(axis=1)
```

# Pandas **melt**
#==============================================================================
```python
df = pd.DataFrame({'State' : ['Texas', 'Arizona', 'Florida'],
          'Apple'  : [12, 9, 0],
          'Orange' : [10, 7, 14],
          'Banana' : [40, 12, 190]})

df.melt(id_vars=['State'],
        value_vars=['Apple','Orange','Banana'],
        var_name='Fruit',
        value_name='Weight')
        
## example 2 *********************
  Weight Category  M35 35-39  M40 40-44  M45 45-49  M50 50-54  M80 80+
0              56        137        130        125        115        102   
1              62        152        145        137        127        112

(df.melt(id_vars='Weight Category', var_name='sex_age', value_name='Qual Total')
   .assign(Sex      = lambda x: x['sex_age'].str.extract('([MF])', expand=True))
   .assign(AgeGroup = lambda x: x['sex_age'].str.extract('(\d{2}[-+](?:\d{2})?)', expand=True))
   .drop('sex_age', axis=1)
   ).head(2)
```

# Pandas **wide_to_long**
#==============================================================================
```python
df = pd.DataFrame({"A1970" : list('ab'),"A1980" : list('de'),"B1970" : [2,3],
                   "B1980" : [10,20],"price" : [100,200],"id" : [1,4]})

## make it long
pd.wide_to_long(df, ['A', 'B'], i=['price','id'] , j='year').reset_index()

   price  id  year  A   B
0    100   1  1970  a   2
1    100   1  1980  d  10
2    200   4  1970  b   3
3    200   4  1980  e  20

## example 2 *********************
Data:
year           artist          track  time date.entered  wk1  wk2
2000           Pac             Baby   4:22   2000-02-26   87   82
2000           Adele           Hello    3:15   2000-09-02  91   87

# solution:
# wk1 wk2 means week is 1 and week 2 and they have different ranks for the given song.
df = pd.read_clipboard()
df = pd.DataFrame({'year' : [2000, 2000],
          'artist' : ['Justin', 'Adele'],
          'track' : ['Baby', 'Hello'],
          'time' : ['4:22', '3:15'],
          'date.entered' : ['2000-02-26', '2000-09-02'],
          'wk1' : [87, 91],
          'wk2' : [82, 87]})

# reorder columns and make wk1 wk2 end columns
df = df[['year', 'artist', 'date.entered', 'time', 'track', 'wk1', 'wk2']]
df1 = (pd.wide_to_long(df, 'wk', i=df.columns.values[:5], j='week')
         .reset_index()
         .rename(columns={'date.entered': 'date', 'wk': 'rank'})
         .assign(date = lambda x: pd.to_datetime(x['date']) + 
                                  pd.to_timedelta((x['week'].astype(int) - 1) * 7, 'd'))
         .sort_values(by=['track', 'date'])
)
print(df1)
   year  artist       date  time  track week  rank
0  2000  Justin 2000-02-26  4:22   Baby    1    87
1  2000  Justin 2000-03-04  4:22   Baby    2    82
2  2000   Adele 2000-09-02  3:15  Hello    1    91
3  2000   Adele 2000-09-09  3:15  Hello    2    87
```

# Pandas **read_csv** with lambda usecols
#==============================================================================
```python
usecols_func = lambda x: 'likes_' in x or x == 'A'
df = pd.read_csv('data.csv', index_col='A', usecols=usecols_func)
```


# Pandas **read_json**
#==============================================================================
```python
pd.read_json('[{"A": 1, "B": 2}, {"A": 3, "B": 4}]')
   A  B
0  1  2
1  3  4
```

# Pandas **MultiIndex**
#==============================================================================
```python
##------------------------------
## create multi-index dataframe
index_matrix = [['a','b'],['c','d'],['e','f']]
data_c0 = [10,20]
data_c1 = [100,200]
index = pd.MultiIndex.from_arrays(index_matrix, names=['index_c0', 'index_c1', 'index_c2'])
df = pd.DataFrame({'data_c0': data_c0,'data_c1': data_c1}, index=index)
print(df)
                            data_c0  data_c1
index_c0 index_c1 index_c2                  
a        c        e              10      100
b        d        f              20      200

****reset index****
print(df.reset_index())
  index_c0 index_c1 index_c2  data_c0  data_c1
0        a        c        e       10      100
1        b        d        f       20      200

##------------------------------
## create multi-index dataframe using groupby
df = pd.DataFrame({"User": ["a", "b", "b", "c", "b", "a", "c"],
                  "Amount": [10.0, 5.0, 8.0, 10.5, 7.5, 8.0, 9],
                  'Score': [9, 1, 8, 7, 7, 6, 9]})
x = df.groupby('User').agg({"Score": ["mean",  "std"], "Amount": "mean"}).reset_index(drop=True)
      Score              Amount
       mean       std      mean
0  7.500000  2.121320  9.000000
1  5.333333  3.785939  6.833333
2  8.000000  1.414214  9.750000

****remove multi-index****
x.columns = [i[0]+'_'+i[1] for i in x.columns.ravel()]
   Score_mean  Score_std  Amount_mean
0    7.500000   2.121320     9.000000
1    5.333333   3.785939     6.833333
2    8.000000   1.414214     9.750000
```

# Pandas **eval**
#==============================================================================
https://pandas.pydata.org/pandas-docs/stable/enhancingperf.html#enhancingperf-eval
```python
## WARNING: If dataset is smaller than 15k rows, eval is several order slower than normal methods.
##          DO NOT USE eval and query if you have less than 10k rows.
## pandas eval uses very less memory
## pd.eval used numexpr module, and operations are fast and memory efficient.
pd.eval('df1 + df2 + df3 + df4') # eval is better than plain df1 + df2 + df3 + df4
pd.eval('df.A[0] + df2.iloc[1]')
df.eval('(A + B) / (C - 1)')
df.eval('A + @column_mean') # column_mean = np.mean(df.A)
df.eval('D = (A + B) / C', inplace=True) # creates new column
pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
df.eval('tip * sin(30 * @pi/180)') # pi = np.pi  # sin/cos/log are supported but not pi.
df.eval("""c = a + b
           d = a + b + c
           a = 1""", inplace=False)
           
## Use plain ol' python for dataframe with rows less than 15k
## df is seaborn tips data.
%timeit df.tip + df.tip + df.tip + df.tip # 195 µs
%timeit pd.eval('df.size + df.size + df.size + df.size', engine='python') # 555 µs (3 times slower)
%timeit pd.eval('df.tip + df.tip + df.tip + df.tip') # 1.08 ms (5.5 times slower)

## example2
cols = ['A-B/A+B','A-C/A+C','B-C/B+C']
x = pd.DataFrame([df.eval(col).values for col in cols], columns=cols)
df.assign(**x)
```

# Pandas **query**
#==============================================================================
```python
## NOTE: pd.query uses pd.eval and which uses numexpr library.
##       From official documentation eval is several order magnitude slower if df has <15k rows.
## NOTE: To use query, always rename columns with spaces to underscores
df.columns = df.columns.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)

df = pd.DataFrame({'a': [1,2,3],'b': [4,5,6], 'c': [7,8,9]})
df.query('a==1 and  b==4 and not c == 8')
df.query('a==1 &  b==4 &  ~c == 8')
df.query('a != a.min()')
df.query('a not in b')  # df[df.a.isin(df.b)]
df.query('c != [1, 2]') # df.query('[1, 2] not in c')
df.query('a < b < c and (not mask) or d > 2')
df.query('a == @myvalue')
df.query('a == @mydict['mykey']) 

## exclude all rows if any row value is minimum of that column
df.columns = df.columns.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
q = ' and '.join([f'{i} != {i}.min()' for i in df.columns])
df.query(q)
```
# pandas.io.json.json_normalize
#==============================================================================
```python
data = [{'state': 'Florida',
              'shortname': 'FL',
              'info': {
                   'governor': 'Rick Scott'
               },
              'counties': [{'name': 'Dade', 'population': 12345},
                          {'name': 'Broward', 'population': 40000},
                          {'name': 'Palm Beach', 'population': 60000}]},
             {'state': 'Ohio',
              'shortname': 'OH',
              'info': {
                   'governor': 'John Kasich'
              },
              'counties': [{'name': 'Summit', 'population': 1234},
                           {'name': 'Cuyahoga', 'population': 1337}]
        }]
        
json_normalize(data=data,record_path='counties', meta=['state', 'shortname', ['info', 'governor']])
```

# Pandas string extract, slice
#==============================================================================
```python
## extract letters between two underscores
df = pd.DataFrame.from_dict({'c0': ['T2 0uM_A1_A01.fcs'], 'MFI': [6995], 'Count': [8505]})
df['new'] = df['c0'].str.extract(r'_(.*?)_')  # e.g. A1 is extracted

## slice dataframe column values
df['name'] = df['gdp'].str.slice(0,3) # gdp == 'GDP-2013' to name = GDP
df['year'] = df['gdp'].str.slice(4).astype(int) # gdp == 'GDP-2013' to year = 2013, does not work for nans
df['year'] = df['gdp'].str.slice(4).astype(float) # gdp == 'GDP-2013' to year = 2013, works for nans
df['year'] = df['gdp'].str[4:].astype(float) # gdp == 'GDP-2013' to year = 2013, if there are nans
```

# Pandas adding date month columns
#==============================================================================
```python
df = pd.DataFrame({'year': [2010,2011], 'month': [2,3], 'day': ['d1','d2']})

## best method ************************
df['day'] = df['day'].str[1:]
df['date'] = pd.to_datetime(df[['year','month','day']])


## another method *********************
df['day'] = df['date'].str[1:].astype(int)
df['date'] = df[['year', 'month', 'day']].apply(
    lambda row: '{:4d}-{:02d}-{:02d}'.format(*row),
    axis=1)
df.head()

## another method *********************
dfx['date'] = df['year'].apply(lambda x: '{:4d}'.format(x)) + '-' +\
              df['month'].apply(lambda x: '{:02d}'.format(x)) + '-' +\
              df['day'].str[1:].astype(int).apply(lambda x: '{:02d}'.format(x))
```

# Tidy up when single column contains multiple variables
#==============================================================================
```python
df = pd.DataFrame({'Name': list('ABC'),
                  'Date': np.arange(2010,2013),
                 'Info': ['Height', 'Weight','Size'],
                 'Value': [6, 160, 3]})
                 
(df.set_index(['Name','Date', 'Info'])
          .squeeze()
          .unstack('Info')
          .reset_index()
          .rename_axis(None, axis='columns'))
          
# using pivot_table
(df.pivot_table(index=['Name', 'Date'], 
                        columns='Info', 
                        values='Value', 
                        aggfunc='first')
           .reset_index()
           .rename_axis(None, axis='columns'))
```

# Tidy up when single cell contains multiple values
#==============================================================================
```python

      City             Geolocation
0  Houston  29.7604° N, 95.3698° W
1   Dallas  32.7767° N, 96.7970° W
2   Austin  30.2672° N, 97.7431° W

g1 = '29.7604° N, 95.3698° W'
g2 = '32.7767° N, 96.7970° W'
g3 = '30.2672° N, 97.7431° W'
df = pd.DataFrame({'City': ['Houston','Dallas','Austin'],
                       'Geolocation': [g1,g2,g3]})

regexp = r'(?P<latitude>[0-9.]+).\s* (?P<latitude_dir>N|S), '+ \
         r'(?P<longitude>[0-9.]+).\s* (?P<longitude_dir>E|W)'
df = pd.concat([df['City'], 
          df['Geolocation'].str.extract(regexp, expand=True)
          ], axis='columns')

df[['latitude','longitude']] =df[['latitude','longitude']].astype(np.float)
df['latitude_dir'] =df['latitude_dir'].astype('category')
df['longitude_dir'] =df['longitude_dir'].astype('category')
# note: multiple columns astype is not supported for category for pandas24.0
df.dtypes
```

# Mathematics
#==============================================================================
```python
df = pd.DataFrame({ 'minutes': [55, 70,np.nan]})
df['hour'], df['min'] = np.divmod(df['minutes'].values,60)
```

# Multiple conditions
#==============================================================================
```python
## example 1
pd.cut(df['count'], 
       bins=[-np.inf, 1, 10, np.inf], 
       labels=['unique', 'infrequent', 'frequent'])
       

## example 2 *********************
### numpy where is fast
np.where(c0 > 400, 'high', 
         (np.where(c0 < 200, 'low', 'medium')))

### cut is best, memory efficient
pd.cut(df['c0'], [0, 200, 400, np.inf], labels=['low','medium','high'])

### general method  *********************
def conditions(x):
    if x > 400:
        return "High"
    elif x > 200:
        return "Medium"
    else:
        return "Low"

func = np.vectorize(conditions)
new_col = func(df["c0"])

### np.select is slowest and should not be used  *********************
col         = 'c0'
conditions  = [ df[col] >= 400, (df[col] < 400) & (df[col]> 200), df[col] <= 200 ]
choices     = [ "high", 'medium', 'low' ]
df["c0"] = np.select(conditions, choices, default=np.nan)
```

# First and last percentage increase in values
#==============================================================================
```python
a = df.drop_duplicates('name', keep='last').set_index('name')['value']
b = df.drop_duplicates('name').set_index('name')['value']
df = b.sub(a).div(a).mul(100).round(2).reset_index()

Aliter:
df = (df.groupby('name')['value']
       .agg([('mylast','last'),('value','first')])
       .pct_change(axis=1)['value']
       .mul(100)
       .reset_index())
```

# New column with numeric manipulations
#==============================================================================
```python
# check string is numeric (for decimal remove dot)
is_num = df["units"].astype(str).str.replace('.', '').str.isnumeric()
df.loc[is_num, 'new_col'] = df.loc[is_num, 'units']/1000 # divide by 1000 if isdigit
df.loc[is_num,'new_col']=( 
       df.loc[is_num, "price"].astype(float)* 
       df.loc[is_num, "units"]/1000)
```

# Weighted average after groupby
#==============================================================================
```python
df = pd.DataFrame({'val1': [10,20,30,40],'val2': [100,200,300,400],
                  'id': [1,1,2,2],'wt': [0.1,0.1,0.5,0.5]})
wtavg = lambda x: np.average(x.loc[:, ['val1','val2']], weights = x.wt, axis = 0)
dfwavg = df.groupby('id').apply(wtavg)
# verify
df1 = df1 = df[df.id == 1]
sum(df1.val1 * df1.wt) / df1.wt.sum() # 15.0
# verify
(10 * 0.1 + 20 * 0.1) / 0.2 # 15.0
```

# Wrapping aggregation functions
#==============================================================================
```python
def pct_between(s, low, high):
    return s.between(low, high).mean()


def make_agg_func(func, name, *args, **kwargs):
    def wrapper(x):
        return func(x, *args, **kwargs)
    wrapper.__name__ = name
    return wrapper

pct_1_3k   = make_agg_func(pct_between, 'pct_1_3k', low=1000, high=3000)
pct_10_30k = make_agg_func(pct_between, 'pct_10_30k', 10000, 30000)

# state abbreviation, religious affiliation, undergraduate students
college.groupby(['STABBR', 'RELAFFIL'])['UGDS'].agg(['mean', pct_1_3k, pct_10_30k]).head()
```

# One column has list of lists
```python
df = pd.DataFrame({'team': ['A', 'B', 'C'],
                  'data': [['x',1,30], ['y',2,31], ['z',3,32]]})
(pd.DataFrame(df.data.tolist())
 .rename({0: 'name',1:'year',2:'age'},axis=1)
 .join(df.team))
```

# pandas and json to get currency conversion
```python
import numpy as np
import pandas as pd
import seaborn as sns

import json
from urllib.request import urlopen

tips = sns.load_dataset('tips')
grouped = tips.groupby(['sex', 'smoker'])

conv = 'USD_EUR'
url = f'http://free.currencyconverterapi.com/api/v5/convert?q={conv}&compact=y'

# DO NOT USE: converter = json.loads(urlopen(url).read())
with urlopen(url) as u:
    raw = u.read()
    converter = json.loads(raw)

print(converter) # {'USD_EUR': {'val': 0.87945}}
g = grouped['total_bill', 'tip'].mean().apply(lambda x: x*converter[conv]['val'])
print(g)
```

# Unnesting column with list
#==============================================================================
```python
df=pd.DataFrame({'A':[1,2],'B':[[1,2],[1,2]]})
def unnest(df):
    vals = np.array(df.B.values.tolist())
    a = np.repeat(df.A, vals.shape[1])
    return pd.DataFrame(np.column_stack((a, vals.ravel())), columns=df.columns)
## testing 
unnest(df)
```

# Plottings
#==============================================================================
NOTE: Pandas plot requires multi-index data and the index-names becomes xticklabels
,however, seaborn plots requires flat table with tidy data. Seaborn is easier than
pandas.

```python
## plot by two columns
df.groupby(['Category','Sex'])['Laureate ID'].count().unstack().reset_index()\
.plot.bar(x ='Category',y = ['Female', 'Male'],figsize=(12,8))

## seaborn plot
sns.scatterplot(data=df, x='c0', y='c1', hue='c2', size='c4', style='c5',
                palette=('Blues',7), marker={'c0': 's', 'c1': 'X'})
                
## sns categorical plot
## scatter: kind = strip, swarm
## distribution: kind = box, violin, boxen
## categorical estimate: kind = point, bar, count
g = sns.catplot('c0', data=df, hue='c1', palette='husl', kind='count')
g.set(ylim=(0,1))
g.despine(left=True)

## logistic plot
age_bins = [15, 30, 45, 60]
sns.lmplot("age", "survived", titanic, hue="sex",
           palette='husl', x_bins=age_bins, logistic=True).set(xlim=(0, 80));
```
# Mixed examples
#==============================================================================
```python
## item totals
df['item_totals'] = df.groupby('names')['values'].transform('sum')

## three columns all greater than 3 respective thresholds
df[df.gt(thr).all(axis=1)] # eg. for thr = [1,2,2], row [8,8,3] is selected.

## choose rows only if a has two or more repeated values
note: transform gives matrix of one less dimension after groupby, all row values are same for b and c
df[df.groupby("a").transform(len)["b"] >= 2]

## 2-digit year to 4-digit year
yr = df['year']
df['year'] = np.where(yr <= 20, 2000 + yr, 1900 + yr)

## change all df values greater to 1
df = df.clip_upper(1)
df = pd.DataFrame(np.clip(df.values, a_min=0, a_max=1),index=df.index,columns=df.columns)

## swapping values based on condition
df['a'], df['b'] = np.where(df['a'] > df['b'] , [df['b'], df['a']], [df['a'], df['b']])

## assign values
df.loc[df['c0'].isin(group),'c1'] = 'good' # group = ['a','b'], if c0 has 'a' c1 value will be changed.
df.loc[ (df['A']=='blue') & (df['C']=='square'), 'D'] = 'Yes'

## Read json
s = pd.read_json('countries.json',typ='series')
df = pd.DataFrame(s)

## drop duplicated values in one column and set its value to 'unknown' in another column
df.loc[df['col1'].duplicated(keep = False), 'col2'] = 'unknown'
df = df.drop_duplicates()

## train test split
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)

# copy data not view
msk = np.random.rand(len(df)) < 0.8
train, test = df[msk].copy(deep = True), df[~msk].copy(deep = True)
```


# Basic Python and Numpy
```python
# looping
params = [10,20,30]
for e,i in enumerate(reversed(range(len(params)))):
    print(e,i,params[i])

# numpy axis increase
x = np.array([1,2,3]) # shape (3,) 1d array  (we can flatten(), or ravel() to get 1d array)
x[None,:] # shape 1,3   2d row vector
x[:,None] # shape 3,1   2d column vector

# vandermonde matrix
x = np.array([1,2,3])
one_x_x2 = np.vander(a,3,increasing=True)

# Sort by column 0, then by column 1
lexsorted_index = np.lexsort((arr[:, 1], arr[:, 0])) 
arr[lexsorted_index]

# vectorize
def foo(x):
    return x+1 if x > 10 else x-1
foo_v = np.vectorize(foo, otypes=[float])
print('x = [10, 11, 12] returns ', foo_v([10, 11, 12]))

# apply_along_axis
def max_minus_min(x):
    return np.max(x) - np.min(x)
print('Row wise: ', np.apply_along_axis(max_minus_min, axis=1, arr=a))

# digitize (each elements of array belogs to which group?)
x = np.arange(10)
bins = np.array([0, 3, 6, 9])
print(x)
print(np.digitize(x, bins)) # element 0 of x belongs to first bin 1 from bin and so on

# unique
x = np.array([1,1,1,2,2,2,5,25,1,1])
unique, counts = np.unique(x, return_counts=True)
uniq_counts = np.asarray((unique, counts)).T

# bincount
x = np.array([1,1,2,2,2,4,4,6,6,6,5]) # doesn't need to be sorted
print(x)
print(list(range(x.max())))
print(np.bincount(x)) # [0, 2, 3, 0, 2, 1, 3]  0 occurs 0 times, 1 occurs 2 times

# Histogram example
x = np.array([1,1,2,2,2,4,4,6,6,6,5,5])
counts, bins = np.histogram(x, [0, 2, 4, 6, 8])
print('Counts: ', counts) # number of items in that bin
print('Bins: ', bins) # just the second argument of hist

# Create date sequence
npdates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-10'))
bus = npdates[np.is_busday(npdates)] # Check if its a business day
dates = npdates.tolist() # convert npdate to datetime.datetime 
years = [dt.year for dt in dates]

# timedelta
dt64 = np.datetime64(date64, 'D')  # conversion: np.datetime_as_string(dt64)
tenminutes = np.timedelta64(10, 'm')  # 10 minutes
tenseconds = np.timedelta64(10, 's')  # 10 seconds
tennanoseconds = np.timedelta64(10, 'ns')  # 10 nanoseconds

print('Add 10 days: ', dt64 + 10)
print('Add 10 minutes: ', dt64 + tenminutes)
print('Add 10 seconds: ', dt64 + tenseconds)
print('Add 10 nanoseconds: ', dt64 + tennanoseconds)

# some operations
a = np.array([[1,3,5],[2,4,6]])   # a.*?  will give all attributes on jupyter-notebook
b = np.c_[np.ones(len(a)), a]
c = np.vstack([b,range(4)])
d = np.r_[c,np.array([[10,20,30,40]])] # r_ needs same dimension
a1 = np.sort(np.array([4,8,2]); np.searchsorted(a,2)  # gives the index 0
d1,d2 = np.vsplit(d,2)
diag = np.diag(d) # gives diagonal elements
diag_mat = np.diag([1,3,4]) # gives diagonal matrix
np.eye(3) # diagonal matrix with all 3

# numpy functions
arr attributes: shape, size, ndim, dtype, argmax, argsort, astype, tofile, tostring, tolist,
universal functions ufuncs: add, subtract, multiply, maximum, minimum, etc
statistics: min, max, mean, std, var, cov, 
np.zeros, np.ones, np.zeros_like, np.ones_like
np.vstack, np.hstack, np.c_[], np.r_[], np.reshape, np.ravel, np.clip, np.array_split, 
np.sort(axis=0), np.argsort, np.diag, np.tile, np.vsplit, np.hsplit,

# array_split allows split into unequal part but np.split does not
x = np.arange(8.0)
parts = np.array_split(x, 3)

x = np.arange(2.0, 8.0) # [2., 3., 4., 5., 6., 7.]
idx = [3, 5, 6, 10]
parts = np.split(x, idx) # 234, 56, 7, empty

# isin
np.isin([1,3,7], [1,2,3,4,5]) # [ True,  True, False]

# set methods
np.setdiff1d([1,5,2,3], [2,8,10,11]) # [1, 3, 5]
np.setxor1d([1,5,2,3], [2,8,10,11]) # [ 1,  3,  5,  8, 10, 11]

# union, intersection
np.union1d([1,5,2,3], [2,8,10,11]) # [ 1,  2,  3,  5,  8, 10, 11]
np.intersect1d([1,5,2,3], [2,8,10,11]) # [ 1,  2,  3,  5,  8, 10, 11]

from functools import reduce
reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2])) # [3]

# np.c_ needs 2d array but np.stack() works on 1d array
# add new column of sums
xyz = np.stack((x,y,z),axis=-1)
last = np.sum(xyz,axis=1,keepdims=True)  # or, last = np.sum(xyz,axis=1)
xyzz = np.c_[xyz,last]                   # then, xyzz = np.c_[xyz,last[None:,]]

# numpy printoptions
print(np.array_repr(x, precision=2, suppress_small=True))
with np.printoptions(precision=2,suppress=True):
    print(x)
```

# Useful links
#==============================================================================
- [realpython](https://realpython.com/python-pandas-tricks/) 
- [datacamp pandas merge_asof](https://campus.datacamp.com/courses/merging-dataframes-with-pandas/merging-data?ex=13#skiponboarding)

# Useful Images
![](images/data_pipelines.png)
![](images/pandas_chaining.png)
