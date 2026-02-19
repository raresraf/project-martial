"""
@file consumer.py
@brief A multi-threaded producer-consumer simulation using a single global lock and extensive logging.
@details This module implements a marketplace simulation where all operations are synchronized
through a single `threading.Lock`, effectively making the `Marketplace` class a monitor.
This approach ensures thread safety at the cost of concurrency. The module also features
detailed logging of marketplace activities to a file.
"""

import time
from threading import Thread, Lock
import logging
import logging.handlers


class Consumer(Thread):
    """
    @brief Represents a consumer thread that buys products from the marketplace.
    @warning This class appears to have a bug in its cart handling. It creates a single cart ID
    in the constructor and uses it for all subsequent operations, which is likely not the
    intended behavior if it is meant to process multiple independent carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts A list of shopping lists, each with 'add'/'remove' commands.
        @param marketplace The shared Marketplace object.
        @param retry_wait_time Time to wait before retrying an action.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Bug: A single cart is created and shared for all operations of this consumer.
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """@brief Main loop for the consumer, processing all assigned commands."""
        for cart in self.carts:
            for data in cart:
                for i in range(data["quantity"]):
                    # Block Logic: Retry an action (add/remove) until it succeeds.
                    ret = False
                    while not ret:
                        if data["type"] == "add":
                            ret = self.marketplace.add_to_cart(self.cart_id, data["product"])
                        else:
                            ret = self.marketplace.remove_from_cart(self.cart_id, data["product"])
                        # If the action failed, wait before retrying.
                        if not ret:
                            time.sleep(self.retry_wait_time)
        # Place the order for the single, shared cart after all commands are processed.
        self.marketplace.place_order(self.cart_id)


class Marketplace:
    """
    @brief The central marketplace, synchronized with a single global lock.
    @details This class manages all products and carts. Every method acquires a global lock,
    ensuring that only one operation can occur in the entire marketplace at any given time.
    While thread-safe, this is a highly contended, non-performant locking strategy.
    """
    
    # Setup for logging all marketplace operations to 'marketplace.log'.
    formatter = logging.Formatter('(%(asctime)s %(threadName)-9s) %(funcName)s %(message)s')
    formatter.converter = time.gmtime
    handler = logging.handlers.RotatingFileHandler('marketplace.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # A global list representing all products currently for sale.
    MARK = []

    # Maps a product to the ID of the producer who published it.
    # Potential Bug: If multiple producers publish a product with the same name (e.g., "milk"),
    # this map will only remember the last producer, which could cause issues.
    GET_PROD = {}

    # Maps a producer_id to their remaining publishing capacity.
    PROD = {}

    # Maps a cart_id to the contents of that cart.
    CONS = {}

    # A single lock to synchronize all marketplace operations.
    lock = Lock()

    def __init__(self, queue_size_per_producer):
        self.logger.info(f" <- queue_zie_per_producer = {queue_size_per_producer}")
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """@brief Registers a new producer, giving them an ID and setting their initial capacity."""
        with self.lock:
            producer_id = len(self.PROD)
            self.PROD[producer_id] = self.queue_size_per_producer
        self.logger.info(f" -> producer_id = {producer_id}")
        return producer_id

    def publish(self, producer_id, product):
        """@brief Adds a product to the central marketplace inventory."""
        self.logger.info(f" <- producer_id = {producer_id}, product = {product}")
        with self.lock:
            if self.PROD[producer_id] > 0:
                self.PROD[producer_id] -= 1
                self.MARK.append(product)
                self.GET_PROD[product] = producer_id
                self.logger.info(f" -> True")
                return True
        self.logger.info(f" -> False")
        return False

    def new_cart(self):
        """@brief Creates a new, empty cart for a consumer."""
        with self.lock:
            cart_id = len(self.CONS)
            self.CONS[cart_id] = {}
        self.logger.info(f" -> cart_id = {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """@brief Moves a product from the marketplace to a consumer's cart."""
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        with self.lock:
            try:
                self.MARK.remove(product)
            except ValueError:
                self.logger.info(f" -> False")
                return False # Product not found in marketplace.
            producer_id = self.GET_PROD[product]
            # Add the product to the cart, organized by its original producer.
            self.CONS[cart_id].setdefault(producer_id, []).append(product)
            self.PROD[producer_id] += 1
        self.logger.info(f" -> True")
        return True

    def remove_from_cart(self, cart_id, product):
        """@brief Removes a product from a cart and returns it to the marketplace."""
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        with self.lock:
            for entry in self.CONS[cart_id]:
                for search_product in self.CONS[cart_id][entry]:
                    if product == search_product:
                        self.CONS[cart_id][entry].remove(search_product)
                        self.MARK.append(product)
                        self.PROD[entry] -= 1
                        self.logger.info(f" -> True")
                        return True
        self.logger.info(f" -> False")
        return False

    def place_order(self, cart_id):
        """@brief Simulates placing an order by printing the contents of the cart."""
        self.logger.info(f" <- cart_id = {cart_id}")
        for prod_list in self.CONS[cart_id].values():
            for prod in prod_list:
                # Inefficiently acquires/releases lock for each print statement.
                with self.lock:
                    print(f'cons{cart_id + 1} bought {prod}')


class Producer(Thread):
    """
    @brief Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """@brief Main loop for the producer, continuously publishing its products."""
        while True:
            for product in self.products:
                for i in range(product[1]):
                    # Block Logic: Retry publishing a product until it succeeds.
                    ret = False
                    while not ret:
                        ret = self.marketplace.publish(self.producer_id, product[0])
                        # This sleep happens even on success, which might not be intended.
                        time.sleep(product[2])
            # After a full cycle of publishing, wait before starting again.
            time.sleep(self.republish_wait_time)
