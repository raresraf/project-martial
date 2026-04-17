"""
@e5ac5a84-12cd-40d4-85b3-fe38aa631e07/consumer.py
@brief multi-threaded electronic marketplace with granular synchronization and auditing.
This module implements a coordinated trading environment where Producers supply goods 
and Consumers execute transactions. The system utilizes multiple specialized mutex 
locks to protect different state categories (registration, session ID generation, 
and inventory levels), minimizing global contention. A rotating log system provides 
a durable audit trail for all marketplace operations.

Domain: Concurrent Systems, Granular Locking, Producer-Consumer Simulation.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Consumer entity simulating an automated shopper.
    Functional Utility: Manages multiple shopping sessions (carts) and performs 
    automated transactions using a polling-based retry strategy.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: Nested list of shopping operation batches.
        @param marketplace: Central trading hub.
        @param retry_wait_time: delay between failed acquisition attempts.
        """
        Thread.__init__(self, **kwargs)
        self.retry_wait_time = retry_wait_time
        self.carts = carts
        self.marketplace = marketplace

    def run(self):
        """
        Main execution loop for shopper actions.
        Logic: Orchestrates session creation and sequential fulfillment of 
        'add' and 'remove' tasks for each assigned cart.
        """
        for cart in self.carts:
            # Atomic creation of a new transaction context.
            cart_id = self.marketplace.new_cart()
            operations_number = 0

            for operation in cart:
                # Block Logic: workload fulfillment loop.
                while operations_number < operation["quantity"]:
                    if operation["type"] == "add":
                        add_to_cart = self.marketplace.add_to_cart(cart_id, operation["product"])
                        if not add_to_cart:
                            # Functional Utility: Fixed-interval backoff when out of stock.
                            time.sleep(self.retry_wait_time)
                        else:
                            operations_number = operations_number + 1
                    else:
                        # Transaction reversal: restore item to global supply.
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        operations_number = operations_number + 1
                operations_number = 0

            # Commit: finalize the session and print purchased inventory.
            self.marketplace.place_order(cart_id)

from threading import Lock, currentThread
import logging
from logging.handlers import RotatingFileHandler


class Marketplace:
    """
    Central coordinator managing inventory buffers and shopper sessions.
    Functional Utility: Uses specialized mutexes to isolate different state 
    mutations, ensuring high performance during concurrent access.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace hub.
        @param queue_size_per_producer: Capacity limit per supply line.
        """
        self.producers_ids = []
        self.producers_sizes = []
        self.carts_number = 0
        self.carts = []
        
        # Granular Synchronization primitives.
        self.print_lock = Lock()
        self.max_elements_for_producer = queue_size_per_producer
        self.num_carts_lock = Lock()
        self.register_lock = Lock()
        self.sizes_lock = Lock()
        
        # Persistence mapping: tracks product origins.
        self.product_to_producer = {}
        self.products = []
        
        # Audit Configuration: configured for rotating file logging.
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        log_form = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        rotating_file_handler = RotatingFileHandler('marketplace.log', 'a', 16384)
        rotating_file_handler.setFormatter(log_form)
        self.logger.addHandler(rotating_file_handler)

    def register_producer(self):
        """
        Allocates a new unique supply line index.
        Logic: Uses a dedicated mutex to ensure atomic registration.
        """
        with self.register_lock:
            prod_id = len(self.producers_ids)
            self.producers_ids.append(prod_id)
            self.producers_sizes.append(0)

        self.logger.info("prod_id = %s", str(prod_id))
        return prod_id

    def publish(self, producer_id, product):
        """
        Accepts a product from a producer into the market.
        Logic: Verifies local capacity before atomic insertion into the global pool.
        @return: True if accepted, False if supply line is full.
        """
        self.logger.info("producer_id = %s product = %s", str(producer_id), str(product))
        prod_id = int(producer_id)

        # Block Logic: Capacity verification.
        for i in range(0, len(self.producers_ids)):
            if self.producers_ids[i] == prod_id:
                if self.producers_sizes[i] >= self.max_elements_for_producer:
                    return False
                # Increment producer load.
                self.producers_sizes[i] = self.producers_sizes[i] + 1

        # Global state update.
        self.products.append(product)
        self.product_to_producer[product] = prod_id
        self.logger.info("return_value = %s", "True")

        return True

    def new_cart(self):
        """Creates a new unique consumer session."""
        with self.num_carts_lock:
            self.carts_number = self.carts_number + 1
            cart_id = self.carts_number

        self.carts.append({"id": cart_id, "list": []})
        self.logger.info("cart_id = %s", str(cart_id))
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from the global pool to a specific shopper session.
        Logic: Atomically claims the product and restores producer capacity.
        """
        self.logger.info("cart_id = %s product = %s", str(cart_id), str(product))
        with self.sizes_lock:
            if product in self.products:
                prod_id = self.product_to_producer[product]
                # Update producer occupancy state.
                for i in range(0, len(self.producers_ids)):
                    if self.producers_ids[i] == prod_id:
                        self.producers_sizes[i] = self.producers_sizes[i] - 1
                
                # Transfer from global pool to session list.
                self.products.remove(product)
                cart = [x for x in self.carts if x["id"] == cart_id][0]
                cart["list"].append(product)
                return True
        self.logger.info("return_value = %s", "False")
        return False

    def remove_from_cart(self, cart_id, product):
        """Restores a product from a cart back to the global supply line."""
        self.logger.info("cart_id = %s product = %s", str(cart_id), str(product))
        cart = [x for x in self.carts if x["id"] == cart_id][0]
        cart["list"].remove(product)
        self.products.append(product)

        with self.sizes_lock:
            prod_id = self.product_to_producer[product]
            # Decrement occupancy for the restoring producer.
            for i in range(0, len(self.producers_ids)):
                if self.producers_ids[i] == prod_id:
                    self.producers_sizes[i] = self.producers_sizes[i] - 1

    def place_order(self, cart_id):
        """Finalizes the purchase and returns the manifest of session goods."""
        self.logger.info("cart_id = %s", str(cart_id))
        cart = [x for x in self.carts if x["id"] == cart_id][0]
        self.carts.remove(cart)
        
        # Output Logic: protected printing to prevent console interleaving.
        for product in cart["list"]:
            with self.print_lock:
                print("{} bought {}".format(currentThread().getName(), product))
        
        self.logger.info("cart_items = %s", str(cart["list"]))
        return cart["list"]


from threading import Thread
import time


class Producer(Thread):
    """
    Simulation thread representing a manufacturing unit.
    Functional Utility: Manages continuous product cycles and handles backpressure 
    from the marketplace coordinator.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the producer and secures a supply line ID.
        """
        Thread.__init__(self, **kwargs)
        self.republish_wait_time = republish_wait_time
        self.products = products
        self.marketplace = marketplace
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        Main production cycle.
        Algorithm: Iterative manufacturing with synchronous backoff retries.
        """
        while True:
            for (product, number_products, time_sleep) in self.products:
                for i in range(number_products):
                    if self.marketplace.publish(str(self.prod_id), product):
                        # Simulate manufacturing time.
                        time.sleep(time_sleep)
                    else:
                        # Functional Utility: Poll backoff when hub is full.
                        time.sleep(self.republish_wait_time)
                        # logic note: decrements index to retry the same unit.
                        i -= 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """Core data model for goods."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Specialization."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Specialization."""
    acidity: str
    roast_level: str
