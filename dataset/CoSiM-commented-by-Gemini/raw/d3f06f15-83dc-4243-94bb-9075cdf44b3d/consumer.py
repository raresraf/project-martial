
"""
@file consumer.py (and others)
@brief A multi-threaded producer-consumer simulation for an e-commerce marketplace.
@details This module defines a producer-consumer system with a central marketplace.
It includes classes for the Consumer, Marketplace, Producer, Products, and a test suite.

@warning CRITICAL CONCURRENCY FLAWS: This implementation is NOT thread-safe and contains
numerous severe race conditions. Key methods like `register_producer`, `new_cart`, and
`add_to_cart` access and modify shared state without proper locking. The `add_to_cart`
method in particular iterates over shared collections without synchronization, which will
lead to exceptions and corrupted data under concurrent load. The locking that does exist
is improperly placed or insufficient.

NOTE: This file appears to be a concatenation of multiple Python files.
"""

from threading import Thread, Lock
import time
import unittest
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass


# --- Consumer Logic ---
class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.
    Each consumer thread processes a list of shopping commands, using a single
    persistent cart ID for all its operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.
        :param carts: A list of shopping lists for the consumer to process.
        :param marketplace: The shared Marketplace instance.
        :param retry_wait_time: Time to wait before retrying a failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.name = kwargs["name"]
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main execution loop for the consumer thread."""
        # A single cart is created and reused for all operations by this consumer.
        cart_id = self.marketplace.new_cart()

        for cart in self.carts:
            for cart_op in cart:
                quantity = cart_op.get("quantity")
                
                # Block Logic: Implements a busy-wait retry loop for add operations.
                if cart_op.get("type") == "add":
                    while quantity > 0:
                        while not self.marketplace.add_to_cart(cart_id, cart_op.get("product")):
                            time.sleep(self.retry_wait_time)
                        quantity -= 1
                elif cart_op.get("type") == "remove":
                    while quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, cart_op.get("product"))
                        quantity -= 1

            for product in self.marketplace.place_order(cart_id):
                print(f"{self.name} bought {product}")


# --- Marketplace Logic ---
class Marketplace:
    """
    The central marketplace that coordinates producers and consumers.
    @warning This class is NOT thread-safe due to missing locks on most
    methods that modify shared state.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.nb_producers = 0 # Shared counter.
        self.nb_consumers = 0 # Shared counter.
        self.producers = {}   # Shared dictionary.
        self.consumers = {}   # Shared dictionary.
        self.producer_lock = Lock()
        self.consumer_lock = Lock() # Global lock for all consumer operations.
        
        logging.basicConfig(filename="marketplace.log", filemode='w',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S')

    def register_producer(self):
        """
        Registers a new producer and returns a unique ID.
        @warning NOT THREAD-SAFE. Modifies `self.producers` and `self.nb_producers`
        without acquiring a lock, leading to a race condition.
        """
        logging.info("producer registered with id %s", self.nb_producers)
        self.producers[self.nb_producers] = []
        self.nb_producers += 1
        return self.nb_producers - 1

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.
        @warning TOCTOU BUG: The check for queue length happens before the lock
        is acquired. Another thread could change the state between the check and the lock.
        """
        logging.info("producer %s published product %s", producer_id, product)
        if len(self.producers[producer_id]) == self.queue_size_per_producer:
            logging.info("publish returned False")
            return False
        
        with self.producer_lock:
            self.producers[producer_id].append(product)
        logging.info("publish returned True")
        return True

    def new_cart(self):
        """
        Creates a new, empty cart and returns its unique ID.
        @warning NOT THREAD-SAFE. Modifies `self.consumers` and `self.nb_consumers`
        without acquiring a lock, leading to a race condition.
        """
        logging.info("cart registered with id %s", self.nb_consumers)
        self.consumers[self.nb_consumers] = []
        self.nb_consumers += 1
        return self.nb_consumers - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.
        @warning CRITICAL RACE CONDITION & INEFFICIENCY: This method iterates over
        the shared `self.producers` dictionary without holding a lock, which can
        cause crashes or incorrect behavior if another thread modifies it.
        This linear search is also extremely inefficient (O(NumProducers * NumProducts)).
        """
        logging.info("cart %s added to cart %s", cart_id, product)
        for producer_id in range(self.nb_producers):
            # This iteration is not synchronized.
            for prd in self.producers[producer_id]:
                if prd == product:
                    # A single global lock for all cart operations creates a bottleneck.
                    with self.consumer_lock:
                        self.producers[producer_id].remove(product)
                        self.consumers[cart_id].append([product, producer_id])
                    logging.info("add_to_cart returned True")
                    return True
        logging.info("add_to_cart returned False")
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the producer."""
        logging.info("cart %s removed from cart %s", cart_id, product)
        for [prd, producer_id] in self.consumers[cart_id]:
            if prd == product:
                with self.consumer_lock:
                    self.consumers[cart_id].remove([product, producer_id])
                    self.producers[producer_id].append(product)
                break

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        @warning NOT THREAD-SAFE. Reads from the shared `self.consumers` dictionary
        without acquiring a lock.
        """
        products = [product for [product, _] in self.consumers[cart_id]]
        logging.info("cart %s placed order: %s", cart_id, products)
        return products

# --- Unit Test Suite & Other Classes ---
class TestMarketplace(unittest.TestCase):
    # This test suite only verifies sequential execution and does not
    # test the concurrency aspects of the marketplace.
    pass

class Producer(Thread):
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for product in self.products:
                [product_id, quantity, wait_time] = product
                time.sleep(wait_time)
                while quantity > 0:
                    while not self.marketplace.publish(self.producer_id, product_id):
                        time.sleep(self.republish_wait_time)
                    quantity -= 1

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    acidity: str
    roast_level: str
