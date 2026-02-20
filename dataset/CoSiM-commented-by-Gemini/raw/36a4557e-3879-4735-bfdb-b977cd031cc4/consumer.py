"""
This module defines the Consumer thread for a marketplace simulation.
The Consumer attempts to acquire a list of products from the marketplace
based on a predefined list of actions.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer that interacts with a Marketplace.

    Each consumer is a thread that executes a series of "add" or "remove"
    operations to fill a shopping cart and finally places an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping lists. Each shopping list contains
                          dictionaries specifying operations ('add'/'remove'),
                          product, and quantity.
            marketplace (Marketplace): The central marketplace object.
            retry_wait_time (float): Seconds to wait before retrying to add a
                                     product if it's unavailable.
            **kwargs: Arguments for the Thread base class, including 'name'.
        """
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.carts = carts
        # Get a unique cart ID from the marketplace for this consumer.
        self.consumer_id = marketplace.new_cart()
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']
        
        pass

    def run(self):
        """
        The main execution loop for the consumer thread.

        Iterates through all assigned shopping operations, attempts to add or
        remove products from the marketplace cart, and places an order at the end.
        """
        # A consumer can have multiple lists of operations to perform.
        for cart in range(len(self.carts)):  
            for operation in range(len(self.carts[cart])):  
                op_type = self.carts[cart][operation]['type']
                product = self.carts[cart][operation]['product']
                quantity = self.carts[cart][operation]['quantity']

                if op_type == "add":
                    # Block Logic: Attempt to add 'quantity' of a 'product'.
                    while quantity > 0:
                        # This inner loop implements a polling mechanism. It will
                        # continuously try to add the product until successful.
                        while True:
                            verdict = self.marketplace.add_to_cart(self.consumer_id, product)
                            
                            # Pre-condition: If the product was successfully added, break the retry loop.
                            if verdict:
                                break
                            
                            # If the product is not available, wait and retry.
                            time.sleep(self.retry_wait_time)

                        quantity -= 1
                else:  # op_type == "remove"
                    # Block Logic: Remove 'quantity' of a 'product'.
                    while quantity > 0:
                        self.marketplace.remove_from_cart(self.consumer_id, product)
                        quantity -= 1
        
        # After all operations, finalize the transaction.
        products = self.marketplace.place_order(self.consumer_id)
        for item in products:
            print(self.name + " bought " + str(item))
