"""
@c0399932-f286-48c1-b198-88b9334e0a9d/consumer.py
@brief multi-threaded producer-consumer marketplace with transaction logging.
This implementation provides a coordinated environment where producers supply products 
and consumers perform transactional operations. It features a centralized state 
management system using class-level primitives and comprehensive logging for 
transaction auditing and debugging.

Domain: Concurrent State Management, Producer-Consumer Simulation, Transaction Logging.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Consumer entity that executes a series of shopping operations.
    Functional Utility: Manages a session (cart) and performs automated add/remove 
    actions with a retry mechanism based on marketplace availability.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread and registers a new cart.
        @param carts: List of operation sequences.
        @param marketplace: Shared coordination hub.
        @param retry_wait_time: Delay between retries for failed transactions.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Functional Utility: secures a unique session identifier upon instantiation.
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """
        Main execution loop for consumer operations.
        Logic: Iteratively processes 'add' and 'remove' requests, waiting for availability 
        if the marketplace is congested or out of stock.
        """
        for cart in self.carts:
            for data in cart:
                for i in range(data["quantity"]):
                    ret = False
                    while not ret:
                        # Functional Utility: conditional dispatch based on operation type.
                        if data["type"] == "add":
                            ret = self.marketplace.add_to_cart(self.cart_id, data["product"])
                        else:
                            ret = self.marketplace.remove_from_cart(self.cart_id, data["product"])
                        
                        # Inline: Implements a polling strategy for resource contention.
                        if not ret:
                            time.sleep(self.retry_wait_time)
        # Finalizes the transaction session.
        self.marketplace.place_order(self.cart_id)

import time
from threading import Lock
import logging
import logging.handlers


class Marketplace:
    """
    Centralized coordinator for producers and consumers.
    Functional Utility: Manages inventory levels and transaction state using 
    shared class-level storage and thread-safe operations.
    """
    
    # Logging Configuration: Set up for rotating file-based auditing.
    formatter = logging.Formatter('(%(asctime)s %(threadName)-9s) %(funcName)s %(message)s')
    formatter.converter = time.gmtime
    handler = logging.handlers.RotatingFileHandler('marketplace.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Static Inventory: Central pool of available products.
    MARK = []

    # Static Registry: Map of products to their originating producers.
    GET_PROD = {}

    # Static Registry: Current capacity and load of producers.
    PROD = {}

    # Static Registry: Active consumer carts and their contents.
    CONS = {}

    # Global synchronization primitive for protecting shared static state.
    lock = Lock()

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace instance.
        @param queue_size_per_producer: Capacity limit per producer.
        """
        self.logger.info(f" <- queue_zie_per_producer = {queue_size_per_producer}")
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        Registers a new producer and allocates capacity.
        Logic: Uses the current count of producers to assign a new ID.
        @return: The newly assigned producer_id.
        """
        self.lock.acquire()
        producer_id = len(self.PROD)
        self.PROD[producer_id] = self.queue_size_per_producer
        self.lock.release()

        self.logger.info(f" -> producer_id = {producer_id}")
        return producer_id

    def publish(self, producer_id, product):
        """
        Accepts a product from a producer into the marketplace.
        Logic: Verifies that the producer has not exceeded their capacity.
        @return: True if successful, False otherwise.
        """
        self.logger.info(f" <- producer_id = {producer_id}, product = {product}")
        self.lock.acquire()
        if self.PROD[producer_id] > 0:
            # Atomic update of supply state.
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
        Allocates a new cart for a consumer session.
        @return: A unique cart_id.
        """
        self.lock.acquire()
        cart_id = len(self.CONS)
        self.CONS[cart_id] = {}
        self.lock.release()

        self.logger.info(f" -> cart_id = {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from the marketplace pool to a specific consumer cart.
        Logic: Implements item-level locking by removing from the global pool.
        @return: True if the item was available and moved, False otherwise.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        try:
            # Functional Utility: atomically claims the product.
            self.MARK.remove(product)
        except ValueError:
            self.lock.release()
            self.logger.info(f" -> False")
            return False
        
        producer_id = self.GET_PROD[product]
        try:
            self.CONS[cart_id][producer_id].append(product)
        except KeyError:
            self.CONS[cart_id][producer_id] = []
            self.CONS[cart_id][producer_id].append(product)
        
        # Increments producer capacity as the item is no longer in the supply queue.
        self.PROD[producer_id] += 1
        self.lock.release()
        self.logger.info(f" -> True")
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Reverses an addition, moving an item from a cart back to the marketplace pool.
        Logic: Searches through the cart and restores producer capacity.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        for entry in self.CONS[cart_id]:
            for search_product in self.CONS[cart_id][entry]:
                if product == search_product:
                    # Atomic reversal of transaction state.
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
        Finalizes the consumer session and prints the purchased inventory.
        """
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
    Manufacturing entity that supplies goods to the marketplace.
    Functional Utility: Runs a production loop for a set of products, 
    observing rate limits and capacity constraints.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the producer and registers with the marketplace.
        @param products: Catalog of products to manufacture.
        @param marketplace: Target distribution hub.
        @param republish_wait_time: Interval for retrying publication when full.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Main production and publication loop.
        Logic: Continuously manufactures items and attempts to publish them 
        to the marketplace, with delays simulating production time.
        """
        while True:
            for product in self.products:
                for i in range(product[1]):
                    ret = False
                    while not ret:
                        ret = self.marketplace.publish(self.producer_id, product[0])
                        # Simulate production overhead.
                        time.sleep(product[2])
            # Backoff before the next production cycle.
            time.sleep(self.republish_wait_time)
