import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file , sep = '\t')
        # print(self.chipo)
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo['order_id'].count()
    
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
        for cols in self.chipo.columns: 
            print(cols)
        pass
    
    def most_ordered_item(self):
        # TODO
        # item_name = None
        # order_id = -1
        # quantity = -1
        sum_of_quantity = self.chipo.groupby('item_name').agg({'quantity':['sum']})
        order_id_count = self.chipo.groupby('item_name').agg({'order_id':['sum']})
        item_name = sum_of_quantity.idxmax().quantity['sum']
        order_id = order_id_count.loc[item_name].order_id['sum']
        quantity = sum_of_quantity.loc[item_name].quantity['sum']
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total? 
       return self.chipo['quantity'].sum()
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        self.chipo['float_item_price'] = self.chipo['item_price'].apply(lambda x: x.replace('$', '')).astype(float)
        self.chipo["float_item_price*quantity"] = self.chipo["float_item_price"] * self.chipo["quantity"]
        return self.chipo["float_item_price*quantity"].sum()
        
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        return self.chipo['order_id'].nunique() 
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        total_no_of_order = self.chipo['order_id'].nunique()
        self.chipo['float_item_price'] = self.chipo['item_price'].apply(lambda x: x.replace('$', '')).astype(float)
        self.chipo["float_item_price*quantity"] = self.chipo["float_item_price"] * self.chipo["quantity"]
        total_price_sales = self.chipo["float_item_price*quantity"].sum()
        print(total_price_sales/total_no_of_order)
        return round(total_price_sales/total_no_of_order,2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return self.chipo['item_name'].nunique()
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        data_frame = pd.DataFrame(list(letter_counter.items()),columns = ['Items','count'])
        data_frame = data_frame.sort_values(by=['count'], ascending=False).head(5)
        figure = plt.figure(figsize = (10, 5)) 
        plt.bar(data_frame['Items'],data_frame['count'], color ='blue',width = 0.8)
        figure.suptitle('Top Popular Items', fontsize=20) 
        plt.xlabel('Order Price', fontsize=16)
        plt.ylabel('Number Items', fontsize=16)
        plt.show(block=True)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        self.chipo['float_item_price'] = self.chipo['item_price'].apply(lambda x: x.replace('$', '')).astype(float)
        total_price_list = self.chipo.groupby('order_id').agg({'float_item_price': ['sum']}).values.tolist()
        total_quantity_list = self.chipo.groupby('order_id').agg({'quantity': ['sum']}).values.tolist()
        plt.scatter(total_price_list, total_quantity_list, color='blue')
        plt.suptitle('Numer of Items per Order Price', fontsize=24) 
        plt.xlabel('Order Price', fontsize=20)
        plt.ylabel('Num Items', fontsize=20)
        plt.show(block=True)
        # 2. groupby the orders and sum it.
        # group = self.chipo.groupby('order_id').agg({'float_item_price_list': ['sum']})
        # print(group)
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        pass
    
        

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
    assert quantity == 761 #Answer differs than the given because quantity was also taken in consideration
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
    
    