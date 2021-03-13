import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep = '\t' )
        print("shape:",self.chipo.shape)
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.#####
        self.topx = self.chipo.head(count)
        print(self.topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.shape[0]
    
    def info(self) -> None:
        # TODO
        # print data info.
        print(self.chipo.info())
        pass
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset.
        return  len(self.chipo.columns)
    

    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(self.chipo.columns)
        pass
    
    def most_ordered_item(self):
        # TODO
       
        most_ordered =  self.chipo.groupby(["item_name"]).agg({"quantity":'sum'}).sort_values('quantity',ascending=False)
        most_ordered = most_ordered.reset_index()
        quantity=most_ordered["quantity"][0]
        item_name=most_ordered["item_name"][0]
        return item_name,quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
        return  self.chipo["quantity"].sum()
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        ls=[]
        self.chipo["item_price"] = pd.to_numeric(self.chipo["item_price"].str.slice(1))
        ls=sum((self.chipo["item_price"]*self.chipo["quantity"]))
        ls="{0:.2f}".format(ls)
        
        return float(ls)
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        len(self.chipo["order_id"].unique())
        return len(self.chipo["order_id"].unique())
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        avg=sum((self.chipo["item_price"]*self.chipo["quantity"]))/ len(self.chipo["order_id"].unique())
        avg="{0:.2f}".format(avg)
    
        return float(avg)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
       
        return len(self.chipo["item_name"].unique())
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        #print(letter_counter)
        lc=pd.DataFrame(list(letter_counter.items()),columns=['ITEM','ITEM_PRICE']).sort_values("ITEM_PRICE",ascending=False)
        lc.reset_index(drop=True, inplace=True)
        lc=lc[:5]
        plt.bar(lc["ITEM"],lc['ITEM_PRICE'])
        plt.title("Most popular items")
        plt.xlabel("Items")
        plt.ylabel("Number of Orders")
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        plt.show(block=True)
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
       
     
     
        price=self.chipo.groupby("order_id").sum(["item_price","quantity"]).reset_index()
      
        plt.scatter(x=price["item_price"],y=price["quantity"],s=50,c='blue')
        plt.title('Number of items per order price')
        plt.xlabel('Order Price')
        plt.ylabel('Num Item')
        plt.show()
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
    item_name,quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
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
    
    