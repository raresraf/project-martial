# -*- coding: utf-8 -*-
"""
Models a consumer thread that simulates a customer performing multiple
shopping sessions in a Marketplace.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    A thread-based worker that simulates a consumer's shopping behavior.

    This class models a consumer who processes a series of shopping lists ('carts').
    For each list, it acquires a new shopping cart from the marketplace,
    executes the defined 'add' and 'remove' operations, and then places an order.

    Attributes:
        carts (list): A list of shopping sessions. Each session is a list of
                      operation dictionaries.
        marketplace (Marketplace): The shared marketplace object.
        retry_wait_time (float): The duration in seconds to wait before
                                 retrying a failed 'add' operation.
        name (str): The name of the consumer thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of shopping sessions.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retries.
            **kwargs: Keyword arguments for the Thread constructor, including 'name'.
        """
        Thread.__init__(self, kwargs=kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']


    def run(self):
        """
        The main execution logic for the consumer thread.

        Processes each shopping session in `self.carts`. For each session, it
        gets a new cart, performs all operations, and places the order. It
        then prints the items bought in that specific order.
        """
        # Block Logic: Iterate through each assigned shopping session.
        for curr_cart in self.carts:
            
            # Functional Utility: A new cart is created for each shopping session,
            # ensuring sessions are isolated from each other.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Process each operation within the current session.
            for curr_op in curr_cart:
                quantity_count = 0

                # Pre-condition: Check if the operation is 'add'.
                if curr_op["type"] == "add":
                    # Block Logic: Persistently try to add the desired quantity of a product.
                    # Invariant: Loop continues until the target quantity has been added.
                    while curr_op["quantity"] > quantity_count:
                        # Attempt to add the item to the cart.
                        if self.marketplace.add_to_cart(cart_id, curr_op["product"]):
                            quantity_count += 1
                        else:
                            # Inline: If adding fails (e.g., out of stock), wait and retry.
                            sleep(self.retry_wait_time)

                
                # Pre-condition: Check if the operation is 'remove'.
                if curr_op["type"] == "remove":
                    # Block Logic: Attempt to remove the product 'quantity' times.
                    # Note: This loop does not check for success and may not be robust
                    # if the item is not in the cart.
                    while curr_op["quantity"] > quantity_count:
                        self.marketplace.remove_from_cart(cart_id, curr_op["product"])
                        quantity_count += 1

            
            # Finalize the session by placing the order for the current cart.
            product_list = self.marketplace.place_order(cart_id)

            # Synchronization: Acquire a lock before printing to prevent interleaved
            # output from multiple consumer threads.
            with self.marketplace.lock:
                for curr_prod in product_list:
                    print(self.name, "bought", curr_prod)
