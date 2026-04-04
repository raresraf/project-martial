# -*- coding: utf-8 -*-
"""
Models a consumer thread that simulates a customer shopping in a Marketplace.
"""

import time
from threading import Thread

class Consumer(Thread):
    """
    A thread-based worker that simulates a consumer's shopping behavior.

    This class represents a consumer who receives a list of shopping actions,
    executes them against a shared marketplace, and finally places an order.
    It handles retries for actions that cannot be immediately fulfilled.

    Attributes:
        carts (list): A list of shopping carts, where each cart contains
                      actions to be performed (add/remove products).
        marketplace (Marketplace): The shared marketplace object from which
                                   products are added or removed.
        id_cart (int): The unique identifier for this consumer's shopping cart,
                       obtained from the marketplace.
        retry_wait_time (float): The duration in seconds to wait before
                                 retrying a failed marketplace action.
        name (str): The name of the consumer thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of shopping lists, each containing dictionaries
                          that specify an action ('add'/'remove'), a product,
                          and a quantity.
            marketplace (Marketplace): An instance of the Marketplace class that
                                       manages product inventory and carts.
            retry_wait_time (float): The time in seconds to sleep before retrying
                                     a failed operation.
            **kwargs: Keyword arguments to be passed to the Thread constructor,
                      such as 'name'.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        # Functional Utility: Each consumer gets a unique cart ID from the marketplace
        # upon initialization.
        self.id_cart = self.marketplace.new_cart()
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through all shopping actions defined in `self.carts`,
        attempting to perform each one. If an action fails (e.g., trying
        to add an out-of-stock item), the thread will wait and retry.
        After all actions are processed, it places the final order.
        """
        # Block Logic: Process each shopping list assigned to this consumer.
        for cart in self.carts:
            # Block Logic: Process each individual action within a shopping list.
            for spread in cart:
                # Unpack the action details from the dictionary.
                tip, prod, qty = spread.values()
                i = 0
                # Block Logic: Attempt to perform the action 'qty' times.
                # Invariant: The loop continues until the action has been
                # successfully completed the specified number of times.
                while i < qty:
                    # Pre-condition: Check if the action is 'add' and if the
                    # marketplace can successfully add the item.
                    if tip == "add" and self.marketplace.add_to_cart(self.id_cart, prod):
                        i += 1
                    # Pre-condition: Check if the action is 'remove' and if the
                    # marketplace can successfully remove the item.
                    elif tip == "remove" and self.marketplace.remove_from_cart(self.id_cart, prod):
                        i += 1
                    else:
                        # Inline: If the marketplace action fails, wait for a specified
                        # time before retrying. This implements a simple busy-wait
                        # polling strategy for resource availability.
                        time.sleep(self.retry_wait_time)

        # Finalize the shopping session by placing the order.
        items_bought = self.marketplace.place_order(self.id_cart)

        # Pre-condition: Check if the order placement resulted in any items.
        if items_bought is not None:
            # Block Logic: Announce each product that was successfully purchased.
            for product in items_bought:
                print(str(self.name) + " bought " + str(product))
