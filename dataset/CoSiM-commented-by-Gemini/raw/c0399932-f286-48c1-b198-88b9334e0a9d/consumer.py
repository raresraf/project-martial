"""
This module implements a multi-threaded producer-consumer simulation of a marketplace.

It features a `Marketplace` class that acts as a central, shared resource for
`Producer` and `Consumer` threads. The `Marketplace` state is managed as a
Singleton using class-level variables and is protected by a single class-level
lock, which serializes all interactions. The implementation uses Python's
`logging` module for instrumentation.

NOTE: This implementation contains a significant bug in the `remove_from_cart`
method, where it incorrectly decrements producer capacity instead of
incrementing it when a product is returned to the marketplace.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that performs a series of shopping actions.

    The consumer is initialized with a list of shopping carts (lists of actions)
    and interacts with the marketplace to fulfill them, using a polling mechanism
    to retry failed actions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping lists to process.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying an action.
            **kwargs: Keyword arguments for the `Thread` parent class.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """
        Main execution logic for the consumer. Processes each cart sequentially.
        """
        for cart in self.carts:
            for data in cart:
                for i in range(data["quantity"]):
                    ret = False
                    # Block Logic: This loop continuously polls the marketplace to
                    # perform an action (add/remove) until it succeeds.
                    while not ret:
                        if data["type"] == "add":
                            ret = self.marketplace.add_to_cart(self.cart_id, data["product"])
                        else:
                            ret = self.marketplace.remove_from_cart(self.cart_id, data["product"])
                        if not ret:
                            time.sleep(self.retry_wait_time)
        
        # After all actions, the order is "placed" (items are printed).
        self.marketplace.place_order(self.cart_id)

import time
from threading import Lock
import logging
import logging.handlers


class Marketplace:
    """
    A Singleton class that manages the shared state of the marketplace.

    It uses class-level variables for state and a single class-level lock to
    ensure thread safety, at the cost of serializing all marketplace operations.
    """
    
    # Set up a rotating file logger for all marketplace activities.
    formatter = logging.Formatter('(%(asctime)s %(threadName)-9s) %(funcName)s %(message)s')
    formatter.converter = time.gmtime
    handler = logging.handlers.RotatingFileHandler('marketplace.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Class variable: A global list representing all products available for sale.
    MARK = []

    # Class variable: Maps a product to the ID of the producer who made it.
    GET_PROD = {}

    # Class variable: Maps a producer ID to their remaining production capacity.
    PROD = {}

    # Class variable: Maps a cart ID to the contents of that cart.
    CONS = {}

    # Class variable: A single lock protecting all shared state.
    lock = Lock()

    def __init__(self, queue_size_per_producer):
        """
        Initializes a marketplace instance.

        Args:
            queue_size_per_producer (int): Max items a producer can publish.
        """
        self.logger.info(f" <- queue_zie_per_producer = {queue_size_per_producer}")
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """Registers a new producer, returning a unique producer ID."""
        self.lock.acquire()
        producer_id = len(self.PROD)
        self.PROD[producer_id] = self.queue_size_per_producer
        self.lock.release()


        self.logger.info(f" -> producer_id = {producer_id}")
        return producer_id

    def publish(self, producer_id, product):
        """
        Adds a product from a given producer to the central marketplace pool.

        Returns:
            bool: True if successful, False if the producer has no capacity.
        """
        self.logger.info(f" <- producer_id = {producer_id}, product = {product}")
        self.lock.acquire()
        if self.PROD[producer_id] > 0:
            self.PROD[producer_id] -= 1
            self.MARK.append(product)
            self.GET_PROD[product] = producer_id
            self.lock.release()
            self.logger.info(f" -> True")
            return True
        self.lock.release()


        self.logger.info(f" -> False")
        return False

    def new_cart(self):
        """Creates a new empty cart and returns its unique ID."""
        self.lock.acquire()
        cart_id = len(self.CONS)
        self.CONS[cart_id] = {}
        self.lock.release()


        self.logger.info(f" -> cart_id = {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from the main marketplace pool to a consumer's cart.
        
        Returns:
            bool: True if the product was available and added, False otherwise.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        try:
            # Atomically removes the product from the available pool.
            self.MARK.remove(product)
        except ValueError:
            # Product was not in the pool.
            self.lock.release()
            self.logger.info(f" -> False")
            return False
        
        producer_id = self.GET_PROD[product]
        # Organizes the cart by which producer made the product.
        try:
            self.CONS[cart_id][producer_id].append(product)
        except KeyError:
            self.CONS[cart_id][producer_id] = []
            self.CONS[cart_id][producer_id].append(product)
        
        # Correctly increments the producer's capacity, as a slot has been freed.
        self.PROD[producer_id] += 1
        self.lock.release()
        self.logger.info(f" -> True")
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the marketplace pool.

        Returns:
            bool: True if the product was found and removed, False otherwise.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        for entry in self.CONS[cart_id]:
            for search_product in self.CONS[cart_id][entry]:
                if product == search_product:
                    self.CONS[cart_id][entry].remove(search_product)
                    self.MARK.append(product)
                    # BUG: This should increment capacity (`+=1`), not decrement.
                    # Decrementing suggests the producer is *less* able to produce
                    # after an item is returned, which is incorrect.
                    self.PROD[entry] -= 1
                    self.lock.release()
                    self.logger.info(f" -> True")
                    return True
        self.lock.release()
        self.logger.info(f" -> False")
        return False

    def place_order(self, cart_id):
        """Prints the contents of a cart to simulate a purchase."""
        self.logger.info(f" <- cart_id = {cart_id}")
        for prod_list in self.CONS[cart_id].values():
            for prod in prod_list:
                self.lock.acquire()
                print(f'cons{cart_id + 1} bought {prod}')
                self.lock.release()

import time
from threading import Thread


class Producer(Thread):
    """
    Represents a producer thread that continuously produces items and publishes
    them to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of products to produce (item, quantity, time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish.
            **kwargs: Keyword arguments for the `Thread` parent class.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """Main execution logic: continuously produce and publish products."""
        while True:
            for product in self.products:
                for i in range(product[1]):
                    ret = False
                    # Block Logic: Polls until the product is successfully published.
                    while not ret:
                        ret = self.marketplace.publish(self.producer_id, product[0])
                        # Simulates the time taken to produce the item.
                        time.sleep(product[2])
            # Waits before starting the next production cycle.
            time.sleep(self.republish_wait_time)
