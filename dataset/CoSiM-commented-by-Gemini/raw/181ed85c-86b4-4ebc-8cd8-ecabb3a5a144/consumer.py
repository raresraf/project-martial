# -*- coding: utf-8 -*-
"""
Models a consumer thread that simulates a customer performing multiple
shopping sessions in a Marketplace.
"""
from threading import Thread
import time

class Consumer(Thread):
    """
    A thread-based worker that simulates a consumer's shopping behavior.

    This class models a consumer who processes a series of shopping lists.
    For each list, it acquires a new shopping cart, executes all the 'add'
    and 'remove' operations for that list, and then places an order for that cart.

    Attributes:
        carts (list): A list of shopping sessions. Each session is a list of
                      operation dictionaries.
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

    def run(self):
        """
        The main execution logic for the consumer thread.

        Processes each shopping session in `self.carts` independently. For each
        session, it gets a new cart, performs all operations, and places an order.
        """
        # Block Logic: Iterate through each assigned shopping session.
        for cart in self.carts:
            # Functional Utility: A new cart is created for each shopping session,
            # ensuring sessions are isolated from each other.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Process each operation within the current session.
            for action in cart:
                action_type = action["type"]
                product = action["product"]
                quantity = action["quantity"]

                i = 0
                # Block Logic: Attempt to perform the action 'quantity' times.
                while i < quantity:
                    # Pre-condition: Check if the action is 'add'.
                    if action_type == "add":
                        # Block Logic: Busy-wait until the item is successfully added.
                        # This polls the marketplace, retrying after a delay on failure.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    else:
                        # Note: This does not check for success and assumes the item
                        # can always be removed.
                        self.marketplace.remove_from_cart(cart_id, product)
                    i += 1

            # Finalize the current session by placing its order.
            self.marketplace.place_order(cart_id)
