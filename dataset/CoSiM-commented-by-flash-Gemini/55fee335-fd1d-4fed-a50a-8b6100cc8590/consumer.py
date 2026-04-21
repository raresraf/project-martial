
"""
@file consumer.py
@brief Concurrent marketplace simulation implementing the Producer-Consumer paradigm.

Functional Intent: Provides a thread-safe platform where multiple Producers can 
publish goods and Consumers can acquire them via virtual carts. Leverages 
fine-grained locking (per-producer mutexes) and robust retry logic to manage 
high-concurrency resource access while maintaining inventory integrity.

Domain: Production Systems, Concurrency and Synchronization.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    @brief Represents a consumer entity that interacts with the marketplace in its own thread.
    
    Logic: Sequentially iterates through shopping carts and executes batch add/remove 
    operations until all target quantities are achieved.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with shopping goals and retry policies.
        @param carts List of carts containing operation sequences (add/remove).
        @param marketplace Reference to the central Marketplace instance.
        @param retry_wait_time Polling interval when items are out of stock.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Core execution loop for the consumer thread.
        
        Algorithm: Iterative task processing with persistent polling.
        """
        # Block Logic: Session initialization.
        cart_id = self.marketplace.new_cart()

        for i in self.carts:
            for j in i:
                quantity = j['quantity']
                product = j['product']
                action = j['type']

                # Block Logic: Batch operation loop.
                for _ in range(0, quantity):
                    if action == 'add':
                        added = self.marketplace.add_to_cart(cart_id, product)

                        # Invariant: Continues polling until the marketplace grants the item reservation.
                        while not added:
                            time.sleep(self.retry_wait_time)
                            added = self.marketplace.add_to_cart(cart_id, product)

                    elif action == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)

            # Functional Intent: Commits the cart contents and logs the transaction.
            self.marketplace.place_order(cart_id)

import logging
import time
from logging.handlers import RotatingFileHandler


class Logger:
    """
    @brief Utility class for configuring thread-safe diagnostic logging.
    """
    
    MAX_BYTE_COUNT = 1000000
    BACKUP_COUNT = 5

    @staticmethod
    def create_logger(name, log_file):
        """
        @brief Factory method for creating a rotating file logger with ISO-like formatting.
        """
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - '
                                      '%(funcName)s - %(message)s')
        formatter.converter = time.gmtime

        handler = RotatingFileHandler(log_file,
                                      maxBytes=Logger.MAX_BYTE_COUNT,
                                      backupCount=Logger.BACKUP_COUNT)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        return logger

import threading
from threading import Lock
import unittest
from tema.product import Tea
from tema.logger import Logger


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for validating Marketplace thread-safety and logic.
    """

    def setUp(self) -> None:
        self.marketplace = Marketplace(1)

    def test_register_producer(self):
        self.assertGreaterEqual(int(self.marketplace.register_producer()), 0)

    def test_publish(self):
        self.assertEqual(self.marketplace.publish(123, None), False)

        producer_id = self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(producer_id, None), False)

        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.publish(producer_id, product), True)

        self.assertEqual(self.marketplace.publish(producer_id, product), False)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.publish(producer_id, product), True)

    def test_new_cart(self):
        self.assertGreaterEqual(int(self.marketplace.new_cart()), 0)

    def test_add_to_cart(self):
        self.assertEqual(self.marketplace.publish(12345, None), False)

        cart_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(cart_id, None), False)

        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), False)

        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, product)


        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)
        self.marketplace.remove_from_cart(cart_id, product)

        snd_cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(snd_cart_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), False)

        self.marketplace.remove_from_cart(snd_cart_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)

        self.marketplace.place_order(cart_id)
        self.marketplace.publish(producer_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)
        self.assertEqual(len(self.marketplace.place_order(cart_id)), 1)

    def test_remove_from_cart(self):
        self.assertEqual(self.marketplace.remove_from_cart(123, None), False)

        cart_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, None), False)

        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), False)

        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), True)

        self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), False)

    def test_place_order(self):
        self.assertEqual(self.marketplace.place_order(1234), None)

        producer_id = self.marketplace.register_producer()
        product = Tea("1", 2, "3")
        self.marketplace.publish(producer_id, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        result = self.marketplace.place_order(cart_id)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 1)

        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        result = self.marketplace.place_order(cart_id)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 1)

        self.assertEqual(self.marketplace.place_order(cart_id), [])


class Marketplace:
    """
    @brief Central coordinator for thread-safe item publishing and purchase fulfillment.
    
    Functional Utility: Manages inventory quotas per producer and handles atomic 
    transfers between producers and consumer carts. Utilizes multiple levels 
    of locking to prevent race conditions during registration, discovery, and checkout.
    """
    
    LOG_FILE = 'marketplace.log'
    MAX_BYTE_COUNT = 1000000
    BACKUP_COUNT = 5

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace with capacity and synchronization primitives.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producers = {} # Logic: Map of producer IDs to [Lock, InventoryList, Count].
        self.carts = {} # Logic: Map of cart IDs to their respective reserved items.
        
        # Synchronization: Global locks for protecting shared index incrementers and cart maps.
        self.register_lock = Lock()
        self.cart_lock = Lock()
        self.order_lock = Lock()
        
        self.crt_assignable_producer_idx = 0
        self.crt_assignable_cart_idx = 0

        self.logger = Logger.create_logger(__name__, Marketplace.LOG_FILE)

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its dedicated synchronization context.
        """
        with self.register_lock:
            producer_id = str(self.crt_assignable_producer_idx)
            self.crt_assignable_producer_idx += 1

        # Logic: Each producer gets its own lock to allow concurrent publishing across different producers.
        self.producers[producer_id] = [Lock(), [], 0]

        self.logger.info("Registered a new producer with ID: %s.", producer_id)

        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Exposes a product to the marketplace if the producer's buffer is not full.
        """
        if producer_id not in self.producers:
            self.logger.error("Unregistered producer ID: %s.", producer_id)
            return False

        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Producer: %s is trying to publish: %s", producer_id, product)

        # Block Logic: Quota validation.
        if self.producers[producer_id][2] == self.queue_size_per_producer:
            self.logger.info("Producer: %s failed to publish: %s. List is full.", producer_id, product)
            return False

        self.producers[producer_id][1].append(product)

        # Synchronization: Critical section for producer inventory counter update.
        with self.producers[producer_id][0]:
            self.producers[producer_id][2] += 1

        self.logger.info("Producer: %s successfully published: %s.", producer_id, product)

        return True

    def new_cart(self):
        """
        @brief Spawns a new shopping session for a consumer.
        """
        with self.cart_lock:
            cart_id = self.crt_assignable_cart_idx
            self.crt_assignable_cart_idx += 1

        self.carts[cart_id] = []

        self.logger.info("Generated a new cart with ID: %s", cart_id)

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers an item from any producer's pool to a specific cart.
        
        Algorithm: Exhaustive linear search across producer pools.
        Logic: Inspects each producer's buffer under their respective lock. 
        If found, the item is removed (claimed) to prevent duplicate sales.
        """
        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return False

        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Consumer is trying to add product: %s to cart: %s.", product, cart_id)

        for key, value in self.producers.items():
            try:
                # Synchronization: Locks the specific producer's pool during the pop operation.
                with value[0]:
                    idx = value[1].index(product)
                    found_product = value[1].pop(idx)
            except ValueError:
                # Logic: Product not present in this producer's pool, continue search.
                continue

            # Invariant: Once popped from the producer pool, the item must be stored in the cart.
            self.carts[cart_id].append((key, found_product))

            self.logger.info("Consumer has successfully added product: %s to cart: %s", product, cart_id)

            return True

        self.logger.info("Consumer failed to add product: %s to cart: %s. Not found.", product, cart_id)

        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a reserved item from a cart back to its originating producer's pool.
        """
        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return False

        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Consumer is trying to remove product: %s from cart: %s.", product, cart_id)

        for i in self.carts[cart_id]:
            if i[1] == product:
                # Logic: Restores the item to the specific producer's buffer.
                self.producers[i[0]][1].append(i[1])
                self.carts[cart_id].remove(i)

                self.logger.info("Consumer has successfully removed product: %s from cart: %s.", product, cart_id)

                return True

        self.logger.info("Consumer failed to remove product: %s from cart: %s. Not in cart.", product, cart_id)

        return False

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction, updating global quotas and clearing the cart session.
        """
        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return None

        self.logger.info("Consumer is trying to place order for contents of cart: %s", cart_id)

        result = []
        # Block Logic: Final checkout pass.
        for i in self.carts[cart_id]:
            # Synchronization: Decrements the producer's active quota count.
            with self.producers[i[0]][0]:
                self.producers[i[0]][2] -= 1

            # Synchronization: Prevents interleaved console output from concurrent consumers.
            with self.order_lock:
                print(str(threading.current_thread().name) + " bought " + str(i[1]))

            result.append(i[1])

        self.carts[cart_id] = [] # Logic: Reset cart after successful checkout.

        self.logger.info("Consumer is has successfully placed order for contents of cart: %s", cart_id)

        return result

from copy import copy
import time
from threading import Thread


class Producer(Thread):
    """
    @brief Represents a producer entity that generates items for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with a product template and production metadata.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Core production loop.
        
        Logic: Registers itself, then continuously iterates through its catalog, 
        cloning templates and publishing them with appropriate production delays.
        """
        producer_id = self.marketplace.register_producer()

        while True:
            for i in self.products:
                
                product_template = i[0]
                quantity = i[1]
                waiting_time = i[2]

                for _ in range(0, quantity):
                    # Logic: Creates a distinct instance from the template for individual tracking.
                    product = copy(product_template)

                    is_published = self.marketplace.publish(producer_id, product)

                    # Invariant: Continues polling until space is available in the producer pool.
                    while not is_published:
                        time.sleep(self.republish_wait_time)
                        is_published = self.marketplace.publish(producer_id, product)

                    # Optimization: Simulate manufacturing time after a successful publish.
                    time.sleep(waiting_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base immutable representation of a marketplace item.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product type for tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product type for coffee.
    """
    acidity: str
    roast_level: str
