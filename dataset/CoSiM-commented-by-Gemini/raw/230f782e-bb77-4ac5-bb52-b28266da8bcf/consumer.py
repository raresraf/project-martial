# -*- coding: utf-8 -*-
"""
Models a consumer thread that simulates a customer's entire shopping journey.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    A thread-based worker that simulates a consumer's entire shopping journey.

    This class represents a consumer who is assigned a set of shopping lists.
    It acquires a single cart and processes all assigned operations ('add' or
    'remove') across all lists before placing one final order.

    Attributes:
        carts (list): A list of shopping lists, where each list contains
                      operation dictionaries specifying an action, product,
                      and quantity.
        marketplace (Marketplace): The shared marketplace object.
        retry_wait_time (float): The duration in seconds to wait before
                                 retrying a failed 'add' operation.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of shopping sessions (each a list of operations).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """
        The main execution logic for the consumer thread.

        Acquires a single cart and processes all assigned shopping operations.
        If an 'add' action fails, the thread will wait and retry. After all
        operations are processed, it places the final order.
        """
        # Block Logic: The loops iterate through a complex nested list structure
        # representing shopping carts and operations.
        for i in range(len(self.carts)):
            for j in range(len(self.carts[i])):
                for k in range(self.carts[i][j]["quantity"]):
                    # Pre-condition: Check if the operation is 'add'.
                    if self.carts[i][j]['type'] == 'add':
                        product = self.carts[i][j]['product']
                        # Block Logic: Persistently try to add the product, sleeping on failure.
                        while not self.marketplace.add_to_cart(self.cart_id, product):
                            time.sleep(self.retry_wait_time)
                    # Pre-condition: Check if the operation is 'remove'.
                    elif self.carts[i][j]['type'] == 'remove':
                        # Note: This does not check for success and assumes the item can always be removed.
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
        
        # Finalize all shopping by placing a single order for the cart.
        self.marketplace.place_order(self.cart_id)
