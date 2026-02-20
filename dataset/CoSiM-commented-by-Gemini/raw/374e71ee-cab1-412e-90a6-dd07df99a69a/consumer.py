"""
This module defines the Consumer thread for a marketplace simulation.
The Consumer attempts to acquire a list of products from the marketplace
based on a predefined list of actions.
"""

from threading import Thread, Lock
import time

class Consumer(Thread):
    """
    Represents a consumer that interacts with a Marketplace to buy products.

    Each consumer runs in its own thread, processing a list of shopping
    operations. It uses a polling mechanism to acquire items and ensures
    thread-safe output when printing its final purchases.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping lists. Each list contains
                          operation dictionaries ('add'/'remove').
            marketplace (Marketplace): The central marketplace object.
            retry_wait_time (float): Seconds to wait before retrying an
                                     unsuccessful 'add' operation.
            **kwargs: Arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self._carts = carts
        self._market = marketplace
        self.retry_time = retry_wait_time
        self._lock = Lock() # A lock to ensure thread-safe printing.

    def run(self):
        """
        The main execution loop for the consumer thread.

        It gets a cart from the marketplace, processes all assigned operations,
        and finally places the order and prints the results in a thread-safe manner.
        """
        cart_id = self._market.new_cart()
        for op_list in self._carts:

            # Iterate through each operation in the shopping list.
            for op in op_list:

                op_type = op["type"]
                prod = op["product"]
                quantity = op["quantity"]

                if op_type == "add":
                    # Block Logic: Attempt to add 'quantity' of a product.
                    while quantity > 0:
                        # Keep trying to add the product until successful.
                        ret = self._market.add_to_cart(cart_id, prod)

                        if ret == True:
                            quantity -= 1
                        else:
                            # If product is not available, wait and retry.
                            time.sleep(self.retry_time)

                if op_type == "remove":
                    # Block Logic: Remove 'quantity' of a product.
                    while quantity > 0:
                        self._market.remove_from_cart(cart_id, prod)
                        quantity -= 1
            
            # Block Logic: Place order and print results.
            # The 'with' statement ensures that the print block is atomic and
            # output from different consumer threads doesn't get mixed.
            with self._lock:
                products_list = self._market.place_order(cart_id)
                for prod in products_list:
                    print("cons" + str(cart_id) + " bought " + str(prod))
