import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file,sep = '\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        self.topx = self.chipo.head(count)
        print(self.topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return len(self.chipo.index)
    
    def info(self) -> None:
        # TODO
        # print data info.
        print(self.chipo.info)
        pass
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        for col in self.chipo.columns: 
            print(col)
        pass
    
    def most_ordered_item(self):
        # TODO
        grouped_df = self.chipo.groupby(['item_name']).agg({'quantity':'sum'})
        grouped_df = grouped_df.reset_index()
        most_ordered_item = grouped_df.sort_values('quantity', ascending=False)

        item_name = list(most_ordered_item['item_name'])[0]
        quantity = list(most_ordered_item['quantity'])[0]
        return item_name, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       total_items = self.chipo['quantity'].sum()
       return total_items
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        # dollar_to_float =  lambda x:float(x[1:])
        Dollar_to_float = lambda x: float(x['item_price'].split()[0].replace('$', ''))*x['quantity']
        float_df = self.chipo.apply(Dollar_to_float,axis=1)
        total = '{0:.2f}'.format(sum(float_df))
        return float(total)
   
    def num_orders(self) -> int:
        # TODO
        grouped_df = self.chipo.groupby('order_id')
        # print()
        return len(grouped_df.first())
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        Dollar_to_float = lambda x: float(x['item_price'].split()[0].replace('$', ''))*x['quantity']
        numerator = sum(self.chipo.apply(Dollar_to_float,axis=1))
        denom = float(len(self.chipo['order_id'].unique()))
        total = '{0:.2f}'.format(numerator/denom)
        # print(total)
        return float(total)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        uniq_items = len(self.chipo['item_name'].unique())
        return uniq_items
    
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
        plt_df = pd.DataFrame(list(letter_counter.items()), columns=['ITEM','ITEM_PRICE']).sort_values('ITEM_PRICE', ascending=False).head(5)
        plt.bar(plt_df['ITEM'], plt_df['ITEM_PRICE'])
        plt.title('Most popular items')
        plt.xlabel('Items')
        plt.ylabel('Number of Orders')
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
        copy_df = self.chipo
        Dollar_to_float = lambda x: float(x['item_price'].split()[0].replace('$', ''))*x['quantity']
        copy_df['item_price'] = self.chipo.apply(Dollar_to_float,axis=1)
        # print(copy_df.head(10))
        grouped_df = copy_df.groupby('order_id')
        x_list = list(grouped_df['item_price'].sum())
        y_list = list(grouped_df['quantity'].sum())
        plt.scatter(x_list, y_list, s=50, c='blue')
        plt.title('Numer of items per order price')
        plt.xlabel('Order Price')
        plt.ylabel('Num Items')
        plt.show()

        pass
    
        

def test() -> None:
    solution = Solution()
    # print(solution.chipo)
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, quantity = solution.most_ordered_item()
    # solution.most_ordered_item()
    # print(item_name, quantity)
    assert item_name == 'Chicken Bowl'
    assert quantity == 761
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    # solution.total_sales()
    assert 1834 == solution.num_orders()
    # solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    # solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    