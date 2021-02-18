import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep = "\t")
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo["order_id"].count()
    
    def info(self) -> None:
        # TODO
        # print data info.
        #print("hello")
        print(self.chipo.info())
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        for col in self.chipo.columns:
            print(col)
    
    def most_ordered_item(self):
        # TODO
        item_name = self.chipo.groupby('item_name').agg({'quantity': ['sum']}).idxmax().quantity['sum']
        quantity  =  self.chipo.groupby('item_name').agg({'quantity': ['sum']}).loc[item_name].quantity['sum']
        order_id  =   self.chipo.groupby('item_name').agg({'order_id': ['sum']}).loc[item_name].order_id['sum']
        
        return (item_name, order_id, quantity)
        

    def total_item_orders(self) -> int:
       return self.chipo['quantity'].sum()
   
    def total_sales(self) -> float:
        self.chipo['item_price_f'] = self.chipo['item_price'].apply(lambda x: x.replace('$', '')).astype(float)
        self.chipo['item_price_m'] = self.chipo['item_price_f'] * self.chipo['quantity']
        return self.chipo['item_price_m'].sum()
   
    def num_orders(self) -> int:
        #print(len(self.chipo.groupby['oder_id']))
        #print(self.chipo.groupby('order_id').sum())
        #print(self.chipo.groupby('order_id').sum())
        return len(self.chipo.groupby('order_id').sum())
    
    def average_sales_amount_per_order(self) -> float:
        return round(self.total_sales()/self.num_orders(), 2)

    def num_different_items_sold(self) -> int:
        return len(self.chipo.groupby('item_name').sum())
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        #print(letter_counter)
        df = pd.DataFrame(list(letter_counter.items()),columns = ['Items','count'])
        df = df.sort_values(by=['count'], ascending=False).head(x)
        fig = plt.figure(figsize = (10, 5)) 
        plt.bar(df['Items'],df['count'],  
        width = 0.4)
        plt.xlabel('Items', fontsize=18)
        plt.ylabel('Number of Orders', fontsize=18)
        fig.suptitle('Most popular items', fontsize=20) 
        plt.show(block=True)
        
    def scatter_plot_num_items_per_order_price(self) -> None:

        self.chipo['item_price_f'] = self.chipo['item_price'].apply(lambda x: x.replace('$', '')).astype(float)
        price = self.chipo.groupby('order_id').agg({'item_price_f': ['sum']}).values.tolist()
        quantity = self.chipo.groupby('order_id').agg({'quantity': ['sum']}).values.tolist()
        plt.scatter(price, quantity, color='blue')
        plt.suptitle('Numer of items per order price', fontsize=20) 
        plt.xlabel('Order Price', fontsize=18)
        plt.ylabel('Quantity', fontsize=18)
        plt.show(block=True)
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    solution.print_columns()
    solution.most_ordered_item()
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	   
    assert quantity == 761  # changed this as I have aggregated on the quantity
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    solution.average_sales_amount_per_order()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    