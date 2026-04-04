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

    def run(self):
        """
        The main execution logic for the consumer thread.

        Acquires a single cart and processes all assigned shopping operations.
        If an 'add' action fails, the thread will wait and retry. After all
        operations are processed, it places the final order.
        """
        # Functional Utility: A single cart is created for the consumer's entire
        # lifecycle, consolidating all purchases into one order.
        cart_id = self.marketplace.new_cart()

        # Block Logic: Iterate through each assigned shopping list.
        for cart in self.carts:
            # Block Logic: Process each individual operation within a list.
            for operation in cart:
                op_name = operation.get("type")
                product = operation.get("product")
                quantity = operation.get("quantity")

                # Pre-condition: Check if the operation is 'add'.
                if op_name == "add":
                    times = 0
                    # Block Logic: Persistently try to add the desired quantity of a product.
                    # Invariant: Loop continues until the target quantity has been successfully added.
                    while times < quantity:
                        # The marketplace returns a truthy value on success.
                        is_successful = self.marketplace.add_to_cart(cart_id, product)
                        if is_successful:
                            times += 1
                        else:
                            # Inline: If adding fails, wait for a specified time before retrying.
                            time.sleep(self.retry_wait_time)
                # Pre-condition: Check if the operation is 'remove'.
                elif op_name == "remove":
                    # Block Logic: Attempt to remove the product 'quantity' times.
                    # Note: This does not check for success and assumes the item is in the cart.
                    for times in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

        # Finalize all shopping by placing a single order for the cart.
        self.marketplace.place_order(cart_id)
