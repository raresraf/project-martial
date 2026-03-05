"""
This module implements a multi-producer, multi-consumer marketplace simulation.

This version uses a highly centralized architecture where all products from all
producers are stored in a single global list within the `Marketplace`. This
creates a major contention point. The module also features extensive logging.

NOTE: This implementation contains SEVERE concurrency bugs. The locking
discipline, particularly in the `add_to_cart` method, is broken and will
lead to race conditions and unpredictable behavior in a multi-threaded
environment. The practice of modifying a list while iterating over it is also
unsafe.
"""


import sys
import time
import logging
from threading import Thread, Lock, currentThread
from logging.handlers import RotatingFileHandler


class Consumer(Thread):
    """
    Represents a consumer thread that attempts to purchase items from a shopping list.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping lists for this consumer to process.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # A dispatch dictionary to call the correct marketplace method.
        self.operations = {"add": marketplace.add_to_cart,
                           "remove": marketplace.remove_from_cart}

    def run(self):
        """Main execution loop for the consumer."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]

                # This loop will busy-wait on a single product until it can be acquired.
                while quantity > 0:
                    op_type = operation["type"]
                    product = operation["product"]

                    # Attempt the operation (e.g., add_to_cart).
                    if self.operations[op_type](cart_id, product) is not False:
                        # If successful, decrement the desired quantity.
                        quantity -= 1
                    else:
                        # If it fails, wait and then retry the same operation.
                        time.sleep(self.retry_wait_time)

            # Once all operations for the cart are complete, place the order.
            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    The central marketplace, which manages all producers, products, and carts.

    BUG: This class uses a single global list for all products, creating a
    massive bottleneck. Its locking strategy is also fundamentally flawed.
    """

    def __init__(self, queue_size_per_producer):
        self.carts_lock = Lock()
        self.carts = []

        # --- Centralized and Problematic Data Structures ---
        self.producers_lock = Lock()
        self.producers_capacity = queue_size_per_producer
        self.producers_sizes = [] # Tracks current size for each producer.
        # A single list for ALL products from ALL producers. This is a bottleneck.
        self.products = []

        # --- Logging Setup ---
        formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
        formatter.converter = time.gmtime
        file_handler = RotatingFileHandler("marketplace.log", maxBytes=4096, backupCount=0)
        file_handler.setFormatter(formatter)
        logger = logging.getLogger("marketplace")
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        self.logger = logger

    def register_producer(self):
        """Registers a new producer, returning a unique producer_id."""
        self.logger.info("enter register_producer()")
        with self.producers_lock:
            self.producers_sizes.append(0)
            self.logger.info("leave register_producer")
            return len(self.producers_sizes) - 1

    def publish(self, producer_id, product):
        """Publishes a product from a given producer to the global product list."""
        self.logger.info("enter publish(%d, %s)", producer_id, str(product))
        with self.producers_lock:
            if self.producers_sizes[producer_id] == self.producers_capacity:
                self.logger.info("leave publish")
                return False

            self.producers_sizes[producer_id] += 1
            self.products.append((product, producer_id))
            self.logger.info("leave publish")
            return True

    def new_cart(self):
        """Creates a new empty cart and returns its ID."""
        self.logger.info("enter new_cart()")
        with self.carts_lock:
            self.carts.append([])
            self.logger.info("leave new_cart")
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        BUG: This method's locking is broken. It releases the lock mid-operation,
        creating a window for race conditions. It also modifies the list
        `self.products` while iterating over it, which is unsafe.
        """
        self.logger.info("enter add_to_cart(%d, %s)", cart_id, str(product))
        self.producers_lock.acquire()
        try:
            # Iterate over a copy to avoid modification issues, though the logic is still flawed.
            for (prod, prod_id) in list(self.products):
                if prod == product:
                    # Decrement producer's published count and remove from global list.
                    self.producers_sizes[prod_id] -= 1
                    self.products.remove((prod, prod_id))
                    
                    # !! BUG !!: The lock is released here, before the operation is complete.
                    self.producers_lock.release()
                    
                    # The product is added to the cart *after* the lock is released.
                    # This section is not protected.
                    self.carts[cart_id].append((prod, prod_id))
                    self.logger.info("leave add_to_cart")
                    return True
        finally:
            # Ensure lock is not held if an exception occurs or loop finishes.
            if self.producers_lock.locked():
                self.producers_lock.release()

        self.logger.info("leave add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the global product list."""
        self.logger.info("enter remove_from_cart(%d, %s)", cart_id, str(product))
        for (prod, prod_id) in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove((prod, prod_id))
                with self.producers_lock:
                    self.products.append((prod, prod_id))
                    self.producers_sizes[prod_id] += 1
                self.logger.info("leave remove_from_cart")
                return True # Should return True, not None
        return False

    def place_order(self, cart_id):
        """Finalizes an order by printing the contents of the cart."""
        self.logger.info("enter place_order(%d)", cart_id)
        order = ""
        for (product, _) in self.carts[cart_id]:
            order += "{} bought {}\n".format(currentThread().getName(), product)
        sys.stdout.write(order)
        self.logger.info("leave place_order")
        return self.carts[cart_id]


class Producer(Thread):
    """Represents a producer thread that creates and publishes products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """Main execution loop for the producer."""
        while True:
            for (product, quantity, wait_time) in self.products:
                produced_count = 0
                while produced_count < quantity:
                    # Attempt to publish a product.
                    if self.marketplace.publish(self.producer_id, product):
                        produced_count += 1
                        time.sleep(wait_time) # Simulate time to produce next item
                    else:
                        # If inventory is full, wait before retrying.
                        time.sleep(self.republish_wait_time)
