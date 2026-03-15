"""
Defines the Consumer thread for a multi-threaded marketplace simulation.

This module contains the Consumer class, which simulates a customer
interacting with a shared marketplace. Each consumer processes a list of
shopping carts, with each cart containing a series of actions like adding
or removing products.
"""
from threading import Thread
import time

# Constants defining the structure of a shopping cart action.
QUANTITY = "quantity"
PRODUCT = "product"
TYPE = "type"
ADD = "add"
REMOVE = "remove"

class Consumer(Thread):
    """
    A thread that simulates a consumer interacting with the marketplace.

    The consumer is initialized with a list of shopping lists ('carts'). For each
    list, it gets a new cart from the marketplace, processes add/remove actions,
    and finally places the order. It includes a retry mechanism for when actions
    cannot be immediately fulfilled (e.g., product not available).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping lists. Each shopping list is a list
                          of dictionaries, where each dictionary represents an action
                          (e.g., {'product': 'tea', 'quantity': 2, 'type': 'add'}).
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): The time in seconds to wait before retrying a
                                     failed marketplace action.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through its assigned shopping lists, executes the specified
        add/remove actions for each product, and places the order.
        """
        # Process each shopping list assigned to this consumer.
        for cart in self.carts:
            
            # Request a new, unique cart ID from the marketplace for this shopping session.
            id_cart = self.marketplace.new_cart()

            
            # Process each action (add/remove) in the current shopping list.
            for element in cart:
                counter = 0
                
                # Pre-condition: Perform the action 'quantity' number of times.
                while counter < element[QUANTITY]:
                    val = False
                    
                    # Attempt to perform the add or remove operation.
                    if element[TYPE] == ADD:
                        val = self.marketplace.add_to_cart(id_cart, element[PRODUCT])
                    elif element[TYPE] == REMOVE:
                        val = self.marketplace.remove_from_cart(id_cart, element[PRODUCT])
                    
                    
                    # Check if the operation was successful.
                    if val:
                        # If successful, increment the counter to move to the next unit.
                        counter += 1
                    elif not val:
                        # Invariant: If the operation failed (e.g., item out of stock),
                        # wait for a specified time and retry the same operation.
                        time.sleep(self.retry_wait_time)
            
            # After all actions in the list are done, place the final order.
            self.marketplace.place_order(id_cart)