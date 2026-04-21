
"""
@file consumer.py
@brief Concurrent marketplace simulation implementing the Producer-Consumer pattern.

Functional Intent: Provides a thread-safe environment for multiple Producers to 
publish products and Consumers to acquire them via virtual shopping carts. 
Uses re-entrant locking for synchronization and implements automated retry logic 
for handling resource contention.

Domain: Production Systems, Concurrency and Synchronization.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    @brief Represents a consumer entity that operates in its own execution thread.
    
    Logic: Sequentially processes assigned carts, attempting to add or remove 
    products until all target quantities are met.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with specific shopping tasks.
        @param carts List of carts containing product operation requests.
        @param marketplace Shared Marketplace instance.
        @param retry_wait_time Interval to wait when the marketplace is temporarily depleted.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        @brief Core execution loop for the consumer thread.
        
        Algorithm: Iterative cart processing with nested operation retries.
        """
        for cart in self.carts:
            # Block Logic: Session initialization.
            consumer_id = self.marketplace.new_cart()
            
            for product in cart:
                size = product["quantity"]
                if product["type"] == "add":
                    # Block Logic: Persistent acquisition loop.
                    # Invariant: Continues polling until the requested quantity is successfully reserved.
                    while size > 0:
                        if self.marketplace.add_to_cart(consumer_id, product["product"]) is True:
                            size -= 1
                        else:
                            # Optimization: Exponential or fixed back-off during contention.
                            time.sleep(self.retry_wait_time)

                else:
                    # Block Logic: Batch removal pass.
                    while size > 0:
                        self.marketplace.remove_from_cart(
                            consumer_id, product["product"])
                        size -= 1

            # Functional Intent: Finalize the transaction and log the results.
            final_products = self.marketplace.place_order(consumer_id)
            for product in final_products:
                print(self.kwargs['name'], "bought", product, flush=True)


import time
from threading import RLock
import unittest
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import sys
sys.path.insert(1, './tema/')
from product import *


class Marketplace:
    """
    @brief Central broker for thread-safe product management and transaction handling.
    
    Functional Utility: Maintains global inventory and per-consumer cart states. 
    Synchronizes access to shared data structures using a re-entrant lock (RLock).
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace with capacity limits.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_size = 0
        self.consumer_size = 0
        
        # Synchronization: Re-entrant lock for protecting nested internal state updates.
        self.lock = RLock()
        
        self.carts = [] # Logic: List of active consumer carts (lists of products).
        self.shop_items = [] # Logic: Available inventory pool.
        self.products_from_producer = [] # Logic: Quota tracking per producer ID.
        
        # Configuration: Diagnostic logging with rotation.
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            'marketplace.log', maxBytes=2000, backupCount=10)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)8s: %(message)s')
        handler.setFormatter(formatter)
        logging.Formatter.converter = time.gmtime
        self.logger.addHandler(handler)



    def register_producer(self):
        """
        @brief Registers a new producer and returns a unique identifier.
        """
        self.logger.info("Entering register_producer function")
        # Synchronization: Critical section for producer count increment.
        self.lock.acquire()
        var = self.producer_size
        self.producer_size += 1
        self.products_from_producer.append(0)
        self.lock.release()
        self.logger.info(
            "Leaving register_producer function with result %d", var)
        return var

    def publish(self, producer_id, product):
        """
        @brief Adds a product to the shop if the producer's quota allows.
        @return True if successful, False if quota is exceeded.
        """
        self.logger.info(
            "Entering publish function with producer_id=%d and product=%s", producer_id, product)
        
        if self.products_from_producer[producer_id] == self.queue_size_per_producer:
            self.logger.info("Leaving publish function with result %r", False)
            return False
        else:
            # Logic: Encapsulate product with its origin for reverse-tracking during checkout.
            prod = {}
            prod["id"] = producer_id
            prod["product"] = product

            self.products_from_producer[producer_id] += 1
            self.shop_items.append(prod)
            self.logger.info("Leaving publish function with result %r", True)
            return True



    def new_cart(self):
        """
        @brief Allocates a new shopping cart for a consumer.
        """
        self.logger.info("Entering new_cart function")
        self.lock.acquire()
        var = self.consumer_size
        self.consumer_size += 1
        self.carts.append([])
        self.lock.release()
        self.logger.info("Leaving new_cart function with result %d", var)
        return var

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically moves a product from the shop inventory to a specific cart.
        """
        self.logger.info(
            "Entering add_to_cart function with cart_id =%d and product=%s", cart_id, product)

        done = 0
        self.lock.acquire()
        for prod in self.shop_items:
            if prod["product"] == product:
                self.carts[cart_id].append(prod)
                done = 1
                self.shop_items.remove(prod)
                break
        self.lock.release()
        
        self.logger.info(
            "Leaving add_to_cart function with result %r", bool(done))

        return bool(done)

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns an item from a cart back to the general shop inventory.
        """
        self.logger.info(
            "Entering remove_from_cart function with cart_id =%d and product=%s", cart_id, product)
        for prod in self.carts[cart_id]:
            if prod["product"] == product:
                self.shop_items.append(prod)
                self.carts[cart_id].remove(prod)
                break
        self.logger.info("Leaving add_to_cart function")

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction, releasing quotas and returning the final product list.
        """
        self.logger.info(
            "Entering place_order function with cart_id =%d", cart_id)
        final_list = []
        
        # Logic: Reconcile producer inventory counters upon sale.
        for prod in self.carts[cart_id]:
            self.products_from_producer[prod["id"]] -= 1
            final_list.append(prod["product"])
            
        self.logger.info(
            "Leaving place_order function with list: %s", final_list)
        return final_list


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for verifying thread-safe marketplace operations.
    """
    def setUp(self):
        self.marketplace = Marketplace(3)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.register_producer(), 3)
        self.assertEqual(self.marketplace.register_producer(), 4)
        self.assertEqual(self.marketplace.register_producer(), 5)

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.new_cart(), 3)
        self.assertEqual(self.marketplace.new_cart(), 4)
        self.assertEqual(self.marketplace.new_cart(), 5)

    def test_publish(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))

    def test_add_to_cart(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertFalse(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))

    def test_remove_from_cart(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertEqual(len(self.marketplace.carts[0]), 2)
        self.marketplace.remove_from_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM'))
        self.assertEqual(len(self.marketplace.carts[0]), 1)

    def test_place_order(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertEqual(len(self.marketplace.carts[0]), 2)
        self.marketplace.remove_from_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM'))
        self.assertEqual(len(self.marketplace.carts[0]), 1)
        self.assertTrue(self.marketplace.add_to_cart(
            0, Tea(name='White Peach', price=5, type='White')))
        list = [Tea(name='English Breakfast', price=2, type='Black'),
                Tea(name='White Peach', price=5, type='White')]
        self.assertEqual(self.marketplace.place_order(0), list)
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))


from threading import Thread
import time


class Producer(Thread):
    """
    @brief Represents a producer entity that generates and publishes products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with a product catalog and scheduling info.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        @brief Core production loop.
        
        Logic: Registers itself, then continuously iterates through its product list, 
        publishing items according to their individual manufacturing times.
        """
        producer_id = self.marketplace.register_producer()
        while self.kwargs['daemon'] is True:
            for product in self.products:
                count_product = product[1]
                while count_product > 0:
                    if self.marketplace.publish(producer_id, product[0]) is True:
                        count_product -= 1
                        # Simulate production delay.
                        time.sleep(product[2])
                    else:
                        # Optimization: Wait before retrying if the market is full.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base immutable representation of a market product.
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
