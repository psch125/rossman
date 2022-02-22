#!/usr/bin/env python
# coding: utf-8

# In[1]:


def processing_data(sales,store) :
    sales['Date'] = pd.to_datetime(sales['Date'], format="%Y-%m-%d") # 년도, 달, 일 순으로 format
    sales['StateHoliday'] = sales['StateHoliday'].replace({0:"0"})
    sales['StateHoliday'] = sales['StateHoliday'].replace({'0':'d'})
    sales = sales.sort_values('Date')
    
    one_hot = []
    for i in range(0,len(sales)) :
        one_hot.append(0)
        
    name = ['StateHoliday_a','StateHoliday_b','StateHoliday_c','StateHoliday_d']
    values = ['a','b','c','d']
    counts = sales['StateHoliday'].value_counts().sort_index().index

    for i in range(0,len(values)) :
        if values[i] not in counts :
            sales[name[i]] = one_hot        
    sales = pd.get_dummies(data = sales, columns = ['StateHoliday'])
    columns = sales.columns.tolist()
    columns.sort()
    sales = sales[columns]
    
    store = processing_store(store)
    store_copy = store
    store_copy= pd.merge(left=sales,right=store_copy,on='Store')

    store_copy['Year']=pd.DatetimeIndex(store_copy.Date).year
    store_copy['Month']=pd.DatetimeIndex(store_copy.Date).month
    store_copy['Day']=pd.DatetimeIndex(store_copy.Date).day
    if 'Id' in sales :
        store_copy = store_copy.drop(['Date','Store','Id'],axis=1)
    else :
        store_copy = store_copy.drop(['Date','Store','Customers'],axis=1)
        

    
    return make_regression_model_1(store_copy)

def processing_store(store) :
    store_copy = store.copy()
    store_copy['CompetitionDistance'] = store_copy['CompetitionDistance'].fillna(
        store_copy['CompetitionDistance'].mean())

    store_copy_cols = ['CompetitionOpenSinceYear','CompetitionOpenSinceMonth',
                    'Promo2SinceWeek','Promo2SinceYear','PromoInterval']
    for i in store_copy_cols :
        store_copy[i].fillna(0,inplace=True)
    store_copy = pd.get_dummies(data=store_copy,columns=['Assortment','PromoInterval','StoreType'])
    return store_copy

def make_regression_model_1(train_data) :
    target_col = 'Sales'
    input_cols = train_data.columns.drop(target_col)
    train_x,test_x,train_y,test_y = train_test_split(train_data[input_cols],
                                                    train_data[target_col],
                                                    test_size=0.4,random_state=1)
    
    test_x,val_x,test_y,val_y = train_test_split(test_x[input_cols],
                                                test_y,
                                                test_size=0.5)
    
    return train_x,test_x,val_x,train_y,test_y,val_y


# In[ ]:




