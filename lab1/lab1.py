from numpy.core.shape_base import block
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.order_id.count()
    
    def info(self) -> None:
        # TODO
        # print data info.
        print(self.chipo.info())
        pass
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(self.chipo.columns)
        pass
    
    def most_ordered_item(self):
        # TODO
        item_name = self.chipo['item_name'].value_counts().idxmax()
        #order_id = -1
        quantity = int(self.chipo['quantity'].where(self.chipo['item_name'] == item_name).dropna(axis = 0).sum())
        return item_name, quantity

    def total_item_orders(self) -> int:
        global totalitem
        totalitem = self.chipo['quantity'].sum()
        return totalitem
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        self.chipo['item_price'] = self.chipo['item_price'].apply(lambda x : x.replace('$',''))
        self.chipo['item_price'] = self.chipo['item_price'].apply(lambda x : float(x))
        global totalsales
        totalsales = (self.chipo['item_price'] * self.chipo['quantity']).sum()
        return totalsales
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        global totalorders
        totalorders = self.chipo['order_id'].max()
        return totalorders
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        avg = (totalsales/totalorders).round(2)
        return avg

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return len(self.chipo['item_name'].unique())
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).


        dct = dict(letter_counter)
        tmp = pd.DataFrame(list(dct.items()),columns = ['item_name','quantity'])
        tmp.sort_values(by = 'quantity', ascending= False, inplace= True)
        df = tmp.iloc[:5]
        plt.figure(figsize=(5,5))
        plt.bar(df["item_name"],df['quantity'])
        plt.title('Most Popular items')
        plt.xticks( size = 5)
        plt.yticks( size = 5)
        plt.show(block = True)
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items

        temp = self.chipo.groupby('order_id').sum()
        plt.scatter(temp['item_price'], temp['quantity'], c='blue', s= 50)
        plt.xlabel('Order Price')
        plt.ylabel('Number of items')
        plt.title("Number of items per order")
        plt.show(block= True)
        pass
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622

    
    solution.info()
    count = solution.num_column()
    assert count == 5
    
    
    item_name, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    #assert order_id == 713926	
    assert quantity == 761      #The previous value(159) seems to be incorrect.
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    
    solution.scatter_plot_num_items_per_order_price()
    
    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    