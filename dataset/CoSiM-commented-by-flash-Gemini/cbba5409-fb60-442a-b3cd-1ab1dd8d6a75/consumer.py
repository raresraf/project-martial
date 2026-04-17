"""
@cbba5409-fb60-442a-b3cd-1ab1dd8d6a75/consumer.py
@brief multi-threaded electronic trading hub with granular locking and reverse indexing.
This module implements a coordinated marketplace where Producers supply goods and 
Consumers execute transactions. It features a sophisticated synchronization model 
using specialized locks for registration, session management, and inventory access. 
An efficient reverse-mapping system (inventory) enables fast location of products 
across multiple supply lines.

Domain: Concurrent Systems, Granular Synchronization, Reverse Indexing.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Simulated customer entity performing automated shopping tasks.
    Functional Utility: Manages individual shopping sessions (carts) and 
    interacts with the marketplace hub using a polling-based retry strategy.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: A list of operation batches (add/remove).
        @param marketplace: The central trading hub.
        @param retry_wait_time: delay between failed acquisition attempts.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Main shopper execution loop.
        Logic: Iteratively processes each cart in the assigned workload.
        """
        for cart in self.carts:
            # Atomic creation of a new session context.
            cart_id = self.marketplace.new_cart()

            for opperation in cart:
                if opperation["type"] == "add":
                    gotten_quantity = 0
                    # Block Logic: Fulfillment loop.
                    # Continuously attempts to acquire the required quantity from the hub.
                    while gotten_quantity < opperation["quantity"]:
                        if self.marketplace.add_to_cart(cart_id, opperation["product"]):
                            gotten_quantity += 1
                        else:
                            # Functional Utility: Fixed-interval backoff during stockout.
                            sleep(self.retry_wait_time)

                elif opperation["type"] == "remove":
                    # Restores items to the global pool.
                    for _ in range(opperation["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, opperation["product"])

            # Commit the session and receive the purchased manifest.
            self.marketplace.place_order(cart_id)


from threading import Lock
from threading import currentThread

import logging
from logging.handlers import RotatingFileHandler
import time

# System Audit: Configured for rotating file logging of all marketplace operations.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('marketplace.log', maxBytes=1000000, backupCount=3)
formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(funcName)s: %(message)s', \
    '%Y-%m-%d %H:%M:%S')
formatter.converter = time.gmtime
handler.setFormatter(formatter)
logger.addHandler(handler)


class Marketplace:
    """
    Central hub for coordinating transactions and inventory state.
    Functional Utility: Uses granular mutex locks to protect different state 
    categories, minimizing total system contention during high-frequency operations.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        @param queue_size_per_producer: Capacity limit for individual supply lines.
        """
        logging.info("enter: %s", queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer
        self.unused_producer_id = 0
        # List of lists representing per-producer stock.
        self.products = []
        # specialized mutex for supply line registration.
        self.register_lock = Lock()
        self.unused_cart_id = 0
        # List of lists representing per-consumer carts.
        self.carts = []
        # specialized mutex for session creation.
        self.cart_lock = Lock()
        # Efficiency Logic: Reverse map from product to the list of producers holding it.
        self.inventory = {}
        # specialized mutex for cross-producer inventory access.
        self.products_lock = Lock()
        # specialized mutex to prevent interleaved console output.
        self.consumer_print_lock = Lock()

        logging.info("exit")

    def register_producer(self):
        """
        Allocates a new unique supply line.
        Logic: Atomic increment of producer counter and initialization of storage.
        """
        logging.info("enter")
        with self.register_lock:
            new_id = self.unused_producer_id
            self.unused_producer_id += 1
        self.products.append([])
        logging.info("exit: %s", new_id)
        return new_id

    def publish(self, producer_id, product):
        """
        Receives a product into the supply pool.
        Logic: Verifies local capacity and updates the reverse index for fast lookup.
        @return: True if accepted, False if the producer's line is full.
        """
        logging.info("enter: %s, %s", producer_id, product)
        if len(self.products[producer_id]) < self.queue_size_per_producer:
            self.products[producer_id].append(product)
            # Maintain the reverse index for $O(1)$ lookup.
            self.inventory.setdefault(product, [])
            self.inventory[product].append(producer_id)
            logging.info("exit: True")
            return True

        logging.info("exit: False")
        return False

    def new_cart(self):
        """
        Creates a new shopper session context.
        @return: A unique cart_id.
        """
        logging.info("enter")
        with self.cart_lock:
            new_id = self.unused_cart_id
            self.unused_cart_id += 1
        self.carts.append([])
        logging.info("exit: %s", new_id)
        return new_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers a product from the supply pool to a customer cart.
        Logic: Uses the reverse index to immediately identify a capable producer.
        """
        logging.info("enter: %s, %s", cart_id, product)
        with self.products_lock:
            if product in self.inventory:
                if len(self.inventory[product]) > 0:
                    # Atomic pop from the reverse index.
                    producer_id = self.inventory[product].pop()
                    # update both session state and global supply state.
                    self.carts[cart_id].append((product, producer_id))
                    self.products[producer_id].remove(product)
                    logging.info("exit: True")
                    return True

        logging.info("exit: False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Restores a product from a cart back to the global supply.
        """
        logging.info("enter: %s, %s", cart_id, product)
        # Search for the product-producer association in the current session.
        cart_list = [tup for tup in self.carts[cart_id] if product in tup]

        if len(cart_list) > 0:
            (item_val, producer_id) = cart_list[0]
            # Transaction Reversal: remove from session, restore to global stock and index.
            self.carts[cart_id].remove(cart_list[0])
            self.inventory[product].append(producer_id)
            self.products[producer_id].append(product)

        logging.info("exit")

    def place_order(self, cart_id):
        """
        Finalizes the purchase and prints the receipt manifest.
        """
        logging.info("enter: %s", cart_id)
        with self.consumer_print_lock:
            # Block Logic: Consistent output formatting.
            for item in [product for (product, _) in self.carts[cart_id]]:
                print(f"{currentThread().getName()} bought {item}")

        logging.info("exit")


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    Simulation thread representing a manufacturing entity.
    Functional Utility: Manages the production cycle and handles backpressure 
    from the marketplace hub.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the producer and secures a supply line ID.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        Infinite manufacturing loop.
        Logic: Cycles through its product catalog and attempts to fulfill quotas.
        """
        while True:
            for product in self.products:
                published_quantity = 0
                # Block Logic: Production Throttling.
                # Attempts to publish until the quantity is met, observing wait times.
                while published_quantity < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        published_quantity += 1
                        # Minimum delay between individual unit publications.
                        sleep(self.republish_wait_time)
                    else:
                        # Functional Utility: Extended backoff when supply line is full.
                        sleep(product[2])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """Core data model for marketplace goods."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Beverage specialization."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Beverage specialization."""
    acidity: str
    roast_level: str
