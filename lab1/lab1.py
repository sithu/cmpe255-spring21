import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'lab1\data\chipotle.tsv'
        self.chipo = pd.read_csv(file,sep='\t')
    
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
        print("Dataframe information\n")
        self.chipo.info()
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print("\nName of All the columns")
        for col in self.chipo:      
            print(col)

    def most_ordered_item(self):
        # TODO
        #item_name = self.chipo['item_name'].evalue_counts().argmax()
        item_name = self.chipo['item_name'].mode().loc[0]
        print("\nMost ordered item - ",item_name)
        order_id = self.chipo.loc[self.chipo['item_name'] == item_name,'order_id'].sum()
        quantity = self.chipo.loc[self.chipo['item_name'] == item_name,'quantity'].sum()
        #quantity = quantity.groupby('quantity')
        print(quantity)
        #quantity = quantity.loc(1).sum()
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       total_item_orders = self.chipo['quantity'].sum()
       print("\nTotal items ordered -",total_item_orders)
       return total_item_orders
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        self.chipo['item_price'] = self.chipo['item_price'].apply(lambda x: x.replace('$', '')).astype(float)
        # 2. Calculate total sales.
        self.chipo['total_sales']= self.chipo['quantity'] * self.chipo['item_price']
        total_sales = self.chipo['total_sales'].sum() 
        print("\nTotal sales = ",total_sales)
        return total_sales
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        num_orders = self.chipo['order_id'].nunique()
        print("Total number of orders placed - ",num_orders)
        return num_orders
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        solution = Solution()
        average_sales_amount = solution.total_sales() / solution.num_orders()
        print("Average sales amount per order - ",average_sales_amount)
        return average_sales_amount.__round__(2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        num_diff_items = self.chipo['item_name'].nunique()
        print("Number of different items sold - ", num_diff_items)
        return num_diff_items
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        data = pd.DataFrame(list(letter_counter.items()),columns =['item_name','popularity_value'])
        # 2. sort the values from the top to the least value and slice the first 5 items
        sorted_data = data.sort_values(by=['popularity_value'],ascending=False)[:5]
        print(sorted_data)
        # 3. create a 'bar' plot from the DataFrame
        ax =sorted_data.plot.bar(x='item_name',y='popularity_value')
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        ax.set_xlabel('Items')
        ax.set_ylabel('Number of Orders')
        ax.set_title('Most popular items')
        # 5. show the plot. Hint: plt.show(block=True).
        plt.show(block=True)
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        list_of_prices = self.chipo['item_price'].to_list()
        #print(list_of_prices)
        # 2. groupby the orders and sum it.
        self.chipo.groupby('order_id').sum()
        print(self.chipo)
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        #x = list_of_prices['item_price']
        #y = list_of_prices['quantity']
        plt.scatter(list_of_prices[:],self.chipo['quantity'],s=50,c='blue')

        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items

        plt.xlabel('Order Price')
        plt.ylabel('Num Items')

        plt.title('Numbe of items per order price')
        plt.show(block=True)
        
    
def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    solution.print_columns()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    assert quantity == 761
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
    
    