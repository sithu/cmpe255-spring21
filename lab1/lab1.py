import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Solution:
    def __init__(self) -> None:
        # TODO:
        # Load data from data/chipotle.tsv file using Pandas library and
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep="\t")

    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())

    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.

        return len(self.chipo)

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
        for col in self.chipo.columns:
            print(col)
        pass

    def most_ordered_item(self):
        # TODO
        item_name = self.chipo.groupby("item_name").size().reset_index(
            name='count').sort_values(by=['count'], ascending=False)['item_name'].iloc[0]
        order_id = self.chipo.groupby(['item_name'])[
            'order_id'].sum().sort_values(ascending=False).head(1).iloc[0]
        quantity = self.chipo.groupby(['choice_description'])[
            'quantity'].sum().sort_values(ascending=False).head(1).iloc[0]
       # item_name = None
        # order_id = -1
        # quantity = -1
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
        # TODO How many items were orderd in total?
        return self.chipo['quantity'].sum()

    def total_sales(self) -> float:
        # TODO
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        dk = self.chipo
        dk = dk.assign(item_price=lambda x: pd.to_numeric(
            x['item_price'].str.replace("$", "", regex=False)))
        dv = (dk['item_price']*self.chipo['quantity']).sum()
        return dv

    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        # print()
        return self.chipo['order_id'].max()

    def average_sales_amount_per_order(self) -> float:
        # TODO

        itemPrice = pd.to_numeric(
            self.chipo['item_price'].str.replace("$", "", regex=False))
        avgSalesPerOrder = (
            itemPrice*self.chipo['quantity']).sum()/self.chipo['order_id'].max()

        return round(avgSalesPerOrder, 2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return self.chipo.item_name.nunique()

    def plot_histogram_top_x_popular_items(self, x: int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)

        # TODO
        # 1. convert the dictionary to a DataFrame
        chipoDetails = pd.DataFrame.from_dict(letter_counter, orient='index', columns=[
                                              'total_orders'])
        # 2. sort the values from the top to the least value and slice the first 5 items
        sortedValues = chipoDetails['total_orders'].sort_values(
            ascending=False).head(x)
        sortedValues.plot(kind="bar", title="Most popular items")
        plt.title("Most popular items")
        plt.xlabel("Items")
        plt.ylabel("Number of Orders")
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labedls:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        pass

    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        withoutDollar = self.chipo['item_price'].str.replace(
            "$", "", regex=False)
        prices = withoutDollar.str.strip()
        pricesList = list(prices)
        # 2. groupby the orders and sum it.
        dv = self.chipo
        dv['item_price'] = pd.to_numeric(
            dv['item_price'].str.replace('$', '', regex=False))

        dv['total'] = dv['quantity']*dv['item_price']
        ordersSum = dv.groupby(['order_id'])['total'].sum()

        ordersQuantity = self.chipo.groupby(['order_id'])['quantity'].sum()
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items

        plt.scatter(ordersSum, ordersQuantity, 50, "blue")
        plt.title("Numer of items per order price")
        plt.xlabel("Order Price")
        plt.ylabel("Num Items")
        plt.show()
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
    solution.print_columns()
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    assert quantity == 159
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
