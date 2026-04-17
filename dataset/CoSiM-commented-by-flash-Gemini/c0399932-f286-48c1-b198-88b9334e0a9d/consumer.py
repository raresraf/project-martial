"""
@c0399932-f286-48c1-b198-88b9334e0a9d/consumer.py
@brief Threaded marketplace simulation with centralized logging and inventory management.
* Algorithm: Distributed task processing with shared state managed via static class-level dictionaries and global locks.
* Functional Utility: Coordinates product flow between independent producer and consumer threads using a marketplace intermediary.
"""

import time
from threading import Thread

class Consumer(Thread):
    """
    @brief Consumer agent that executes a sequence of purchase and return operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer and registers a new cart within the marketplace.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """
        @brief Main transaction loop for the consumer.
        Algorithm: Iterative execution of operations with a retry mechanism for failed inventory interactions.
        """
        for cart in self.carts:
            for data in cart:
                for i in range(data["quantity"]):
                    ret = False
                    # Logic: Busy-wait loop with sleep-based backoff to handle transient inventory unavailability.
                    while not ret:
                        if data["type"] == "add":
                            ret = self.marketplace.add_to_cart(self.cart_id, data["product"])
                        else:
                            ret = self.marketplace.remove_from_cart(self.cart_id, data["product"])
                        
                        if not ret:
                            time.sleep(self.retry_wait_time)
        
        # Post-condition: Finalizes the order, effectively "checking out" from the marketplace.
        self.marketplace.place_order(self.cart_id)

import time
from threading import Lock
import logging
import logging.handlers

class Marketplace:
    """
    @brief Centralized inventory controller with persistent logging and thread-safe operations.
    * Domain: System-wide singleton-like state using class-level variables.
    """
    
    # Block Logic: Configuration for the audit logging system using rotating file handlers.
    formatter = logging.Formatter('(%(asctime)s %(threadName)-9s) %(funcName)s %(message)s')
    formatter.converter = time.gmtime
    handler = logging.handlers.RotatingFileHandler('marketplace.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # State Storage: Uses shared lists and dictionaries for global inventory visibility.
    MARK = []         # Intent: Global list of products currently available for purchase.
    GET_PROD = {}     # Intent: Reverse lookup map from product to its original producer ID.
    PROD = {}         # Intent: Tracks available capacity (quota) for each producer.
    CONS = {}         # Intent: Maps cart IDs to their respective product contents.

    lock = Lock()

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace instance with producer capacity constraints.
        """
        self.logger.info(f" <- queue_zie_per_producer = {queue_size_per_producer}")
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its inventory quota.
        Invariant: Uses global class lock to ensure unique producer ID assignment.
        """
        self.lock.acquire()
        producer_id = len(self.PROD)
        self.PROD[producer_id] = self.queue_size_per_producer
        self.lock.release()

        self.logger.info(f" -> producer_id = {producer_id}")
        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Adds a product to the global market pool if the producer has remaining quota.
        Pre-condition: Target producer must have a quota greater than 0.
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
        """
        @brief Creates a new consumer cart registry.
        """
        self.lock.acquire()
        cart_id = len(self.CONS)
        self.CONS[cart_id] = {}
        self.lock.release()

        self.logger.info(f" -> cart_id = {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to move a product from the market pool to a specific cart.
        Logic: Atomically verifies availability, updates consumer state, and restores producer quota.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        try:
            # Logic: Atomic removal from global pool.
            self.MARK.remove(product)
        except ValueError:
            # Post-condition: Product not available in market.
            self.lock.release()
            self.logger.info(f" -> False")
            return False
        
        producer_id = self.GET_PROD[product]
        try:
            self.CONS[cart_id][producer_id].append(product)
        except KeyError:
            self.CONS[cart_id][producer_id] = [product]
        
        # Logic: Returning items to cart increases the "available production capacity" for that producer.
        self.PROD[producer_id] += 1
        self.lock.release()
        self.logger.info(f" -> True")
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a product from a cart back to the global market pool.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        # Logic: Deep search across all producer-partitioned entries within the cart.
        for entry in self.CONS[cart_id]:
            for search_product in self.CONS[cart_id][entry]:
                if product == search_product:
                    self.CONS[cart_id][entry].remove(search_product)
                    self.MARK.append(product)
                    self.PROD[entry] -= 1
                    self.lock.release()
                    self.logger.info(f" -> True")
                    return True
        self.lock.release()
        self.logger.info(f" -> False")
        return False

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction by outputting all purchased items.
        """
        self.logger.info(f" <- cart_id = {cart_id}")
        for prod_list in self.CONS[cart_id].values():
            for prod in prod_list:
                # Logic: Uses lock to synchronize output to standard streams.
                self.lock.acquire()
                print(f'cons{cart_id + 1} bought {prod}')
                self.lock.release()

import time
from threading import Thread

class Producer(Thread):
    """
    @brief Production agent that periodically generates goods for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer and registers it with the target marketplace.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Continuous production lifecycle.
        Algorithm: Iterative product publishing with fixed production delay per item.
        """
        while True:
            for product in self.products:
                for i in range(product[1]):
                    ret = False
                    while not ret:
                        # Logic: Blocks until the marketplace accepts the new product batch.
                        ret = self.marketplace.publish(self.producer_id, product[0])
                        # Domain: Production Latency.
                        time.sleep(product[2])
            # Functional Utility: Throttles re-publication attempts to avoid market saturation.
            time.sleep(self.republish_wait_time)
