
"""
@file consumer.py
@brief Simulated marketplace environment implementing a concurrent Producer-Consumer system.

Functional Intent: Provides a multi-threaded framework where Producers generate items 
and Consumers purchase them via a central Marketplace. Manages thread synchronization 
to ensure data consistency across carts and inventory.

Domain: Production Systems, Concurrency and Synchronization.
"""

from threading import Thread, Semaphore
import time

class Consumer(Thread):
    """
    @brief Represents a consumer entity that interacts with the marketplace in a separate thread.
    
    Logic: Iterates through assigned carts and performs batch 'add' or 'remove' operations 
    until all requested quantities are met, then finalizes orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the Consumer with specific shopping goals.
        @param carts List of carts containing operation sequences (add/remove).
        @param marketplace Reference to the shared Marketplace instance.
        @param retry_wait_time Duration to wait when an operation fails due to resource contention.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

        # Synchronization: Ensures atomic access to shared consumer-specific resources.
        self.consumer_sem = Semaphore(1)

    def halt_consumer(self, halt_period):
        """
        @brief Suspends the consumer thread for a defined interval.
        """
        time.sleep(halt_period)

    def run(self):
        """
        @brief Core execution loop for the consumer thread.
        
        Algorithm: Sequential cart processing with nested operation retry logic.
        """
        for cart in self.carts:
            
            # Block Logic: Registration of a new session (cart) in the marketplace.
            self.consumer_sem.acquire()
            cart_id = self.marketplace.new_cart()
            self.consumer_sem.release()

            # Block Logic: Execution of individual shopping operations.
            for operation in cart:

                actual_quantity = int(operation['quantity'])
                for _ in range(actual_quantity):

                    selected_operation = operation['type']
                    requested_item = operation['product']

                    # Block Logic: Retry loop for handling inventory availability.
                    # Invariant: Continues attempts until the marketplace succeeds or yields a null response.
                    while True:

                        if selected_operation == 'add':
                            self.consumer_sem.acquire()
                            add_op = self.marketplace.add_to_cart(cart_id, requested_item)
                            self.consumer_sem.release()

                            if add_op:
                                break

                            # Optimization: Back-off strategy when items are out of stock.
                            self.halt_consumer(self.retry_wait_time)

                        elif selected_operation == 'remove':
                            self.consumer_sem.acquire()
                            rem_op = self.marketplace.remove_from_cart(cart_id, requested_item)
                            self.consumer_sem.release()

                            if rem_op is None:
                                break

                            self.halt_consumer(self.retry_wait_time)

            # Finalizes the transaction for the current cart.
            self.marketplace.place_order(cart_id)


from threading import Lock
import threading


import time
import unittest
import logging
from logging.handlers import RotatingFileHandler

# Configuration: Sets up rotating logs for system activity tracking.
logging.basicConfig(
        handlers=[RotatingFileHandler('./marketplace.log', maxBytes=100000, backupCount=10)],
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = time.gmtime

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for validating Marketplace logic and thread safety.
    """
    
    def setUp(self):
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), self.marketplace.producer_id - 1)

    def test_publish_true(self):
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(0, "prod1"))

    def test_publish_false(self):
        self.marketplace.register_producer()

        self.marketplace.items_per_producer[0] = 6
        self.assertFalse(self.marketplace.publish(0, "prod1"))

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), self.marketplace.num_carts - 1)

    def test_add_to_cart_true(self):
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "prod1")

        cart_id = self.marketplace.new_cart()

        self.assertTrue(self.marketplace.add_to_cart(cart_id, "prod1"))

    def test_add_to_cart_false(self):
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "prod1")

        cart_id = self.marketplace.new_cart()

        self.assertFalse(self.marketplace.add_to_cart(cart_id, "prod2"))

    def test_remove_from_cart(self):
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "prod1")

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod1")

        self.marketplace.remove_from_cart(cart_id, "prod1")
        self.assertEqual(self.marketplace.carts_info[cart_id], [])

    def test_place_order(self):
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "prod1")

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod1")
        product_list_test = self.marketplace.place_order(cart_id)
        self.assertEqual(product_list_test, ['prod1'])

class Marketplace:
    """
    @brief Central coordinator for producer registration, item publishing, and consumer transactions.
    
    Functional Utility: Acts as a thread-safe broker between independent Producers 
    and Consumers, managing inventory quotas and cart states.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace with capacity constraints.
        @param queue_size_per_producer Maximum items a single producer can have in the market simultaneously.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.items_per_producer = [] # Logic: Track inventory per registered producer.
        self.available_in_market = [] # Logic: Master list of all items currently for sale.
        self.manufacturer = {} # Logic: Map products to their respective producers.
        self.num_carts = 0
        self.producer_id = 0
        self.carts_info = {} # Logic: State tracking for active consumer shopping sessions.

        # Synchronization: Locks to protect console output and shared cart data structures.
        self.consumer_lock_object = Lock()
        self.cart_lock_object = Lock()

    def register_producer(self):
        """
        @brief Onboards a new producer and assigns a unique identifier.
        """
        producer_id = self.producer_id
        self.producer_id += 1
        self.items_per_producer.append([])

        logging.info(f'Method "register_producer" - output: producer_id = {producer_id}')
        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Exposes a new product to the market if the producer's quota is not exceeded.
        @return True if successfully published, False otherwise.
        """
        logging.info(f'Method "publish" - input: producer_id = {producer_id}, product = {product}')
        current_queue_size = len(self.items_per_producer[producer_id])

        if current_queue_size < self.queue_size_per_producer:
            self.items_per_producer[producer_id].append(product)
            self.manufacturer[product] = producer_id
            self.available_in_market.append(product)

            logging.info(f'Method "publish" - output: True')
            return True

        logging.info(f'Method "publish" - output: False')
        return False

    def new_cart(self):
        """
        @brief Allocates a new cart identifier for a consumer.
        """
        cart_id = self.num_carts
        self.num_carts += 1

        if cart_id not in self.carts_info.keys():
            self.carts_info[cart_id] = []

        logging.info(f'Method "new_cart" - output: cart_id = {cart_id}')
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product from the market inventory to a specific cart.
        
        Logic: Atomically removes the item from the producer's queue and general 
        market list to prevent over-selling.
        """
        logging.info(f'Method "add_to_cart" - input: cart_id = {cart_id}, product = {product}')

        if product not in self.available_in_market:
            logging.info(f'Method "add_to_cart" - output: False')
            return False

        # Synchronization: Critical section for inventory mutation.
        self.cart_lock_object.acquire()
        actual_producer = self.manufacturer[product]

        if product in self.items_per_producer[actual_producer]:
            self.items_per_producer[actual_producer].remove(product)

        if product in self.available_in_market:
            self.available_in_market.remove(product)
        self.cart_lock_object.release()

        self.carts_info[cart_id].append(product)
        logging.info(f'Method "add_to_cart" - output: True')
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns an item from a consumer's cart back to the market inventory.
        """
        logging.info(f'Method "remove_from_cart" - input: cart_id = {cart_id}, product = {product}')
        self.available_in_market.append(product)

        actual_producer = self.manufacturer[product]
        self.items_per_producer[actual_producer].append(product)

        if product in self.carts_info[cart_id]:
            self.carts_info[cart_id].remove(product)

        return

    def place_order(self, cart_id):
        """
        @brief Finalizes the purchase, removes the cart, and prints items to console.
        """
        logging.info(f'Method "place_order" - input: cart_id = {cart_id}')

        products = []
        for cart_key, cart_value in self.carts_info.items():
            if cart_key == cart_id:
                products = cart_value
                del self.carts_info[cart_key]
                break

        # Synchronization: Serializes console output to prevent interleaved print statements.
        self.consumer_lock_object.acquire()
        for product in products:
            print(f'{threading.currentThread().name} bought {product}')
        self.consumer_lock_object.release()

        logging.info(f'Method "place_order" - products = {products}')
        return products


from threading import Thread, Semaphore
import time

class Producer(Thread):
    """
    @brief Represents a producer entity that generates items for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the Producer with a product catalog.
        @param products List of (product_id, quantity, production_time) tuples.
        @param marketplace Shared Marketplace instance.
        @param republish_wait_time Duration to wait when the marketplace is full.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.producer_sem = Semaphore(1)

    def halt_producer(self, halt_period):
        """
        @brief Simulates production time or back-off wait.
        """
        time.sleep(halt_period)

    def run(self):
        """
        @brief Core production loop.
        
        Logic: Registers as a producer, then continuously attempts to publish 
        its product catalog according to specified quantities and production times.
        """
        self.producer_sem.acquire()
        current_prod_id = self.marketplace.register_producer()
        self.producer_sem.release()

        while True:
            for product in self.products:
                product_obj = product[0]
                quantity = product[1]
                wait_time = product[2]
                idx = 0

                while idx < int(quantity):
                    if self.marketplace.publish(current_prod_id, product_obj) is False:
                        # Back-off if market quota is reached.
                        self.halt_producer(self.republish_wait_time)
                    else:
                        # Simulate manufacturing time before the next item.
                        self.halt_producer(wait_time)
                        idx += 1
