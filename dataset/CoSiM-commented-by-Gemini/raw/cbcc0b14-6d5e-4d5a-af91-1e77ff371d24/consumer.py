"""
Module providing a multi-threaded simulation of a retail marketplace.
Uses specialized worker threads (Producers and Consumers) coordinated by a 
central Marketplace entity that manages inventory state and transaction fulfillment.
"""

from threading import Thread, Lock
import time


class Consumer(Thread):
    """
    Represents an automated shopper thread that executes a predefined shopping list.
    Orchestrates the acquisition and removal of products across multiple shopping sessions (carts).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer with shopping tasks.
        
        Args:
            carts (list): List of operations per cart (e.g., [{'product': p, 'type': 'add', 'quantity': q}]).
            marketplace (Marketplace): The central broker for transactions.
            retry_wait_time (float): Seconds to wait if a desired product is out of stock.
            **kwargs: Thread configuration options.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = 0

    def run(self):
        """
        Executes the shopping lifecycle for each assigned cart.
        Logic: Sequential processing of operations (add/remove) with busy-wait retries for stock availability.
        """
        # Block Logic: Iterates through each independent shopping session.
        for cart in self.carts:
            # Atomic setup for thread-local cart identification.
            lock = Lock()
            lock.acquire()
            self.cart_id = self.marketplace.new_cart()
            lock.release()

            # Block Logic: Processes individual product operations within the current cart.
            for ops in cart:
                type_operation = ops['type']
                product = ops['product']
                quantity = ops['quantity']
                i = 0

                if type_operation == "add":
                    # Block Logic: Indefinite retry loop for product acquisition.
                    # Invariant: Consumer waits for the marketplace to fulfill the request before moving to the next item.
                    while i < quantity:
                        status = self.marketplace.add_to_cart(self.cart_id, product)
                        if not status:
                            # Backoff: wait for producers to replenish stock.
                            time.sleep(self.retry_wait_time)
                        else:
                            i += 1
                else:
                    # Action: Return reserved products back to the marketplace.
                    while i < quantity:
                        self.marketplace.remove_from_cart(self.cart_id, product)
                        i += 1

            # Finalization: Converts the cart contents into a finalized transaction log.
            placed_order_cart = self.marketplace.place_order(self.cart_id)

            # Functional Utility: Synchronized console output for simulation tracking.
            lock = Lock()
            for product_bought in placed_order_cart:
                lock.acquire()
                print("{} bought {}".format(self.name, product_bought))
                lock.release()

import logging
from logging.handlers import RotatingFileHandler
import time
import unittest
from dataclasses import dataclass

class Marketplace:
    """
    Acts as a mediator between Producers and Consumers.
    Manages global product availability, individual producer quotas, and consumer cart state.
    Note: State transitions are implemented via list manipulation across shared inventories.
    """

    def __init__(self, queue_size_per_producer):
        """
        Args:
            queue_size_per_producer (int): Buffer limit for each producer's public listing.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.count_producers = 0  
        self.carts = [] # cart_id -> [product, ...]
        self.producer_products = [] # producer_id -> [available_product, ...]
        self.reserved_products = [] # producer_id -> [reserved_product, ...]
        
        # Telemetry Configuration: Rotating logs for system auditing.
        logger = logging.getLogger('my_logger') 
        logger.setLevel(logging.INFO) 
        handler = RotatingFileHandler('marketplace.log', maxBytes=2000, backupCount=10)
        formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
        handler.setFormatter(formatter)
        logging.Formatter.converter = time.gmtime
        logger.addHandler(handler)

        logger.info("Marketplace created")

    def register_producer(self):
        """
        Registers a new supply source in the system and initializes its inventory buffers.
        """
        logger = logging.getLogger('my_logger')
        logger.info("Producer registration started")

        self.producer_products.append([])
        self.reserved_products.append([])
        self.count_producers = self.count_producers + 1

        logger.info("Producer registration finished")
        return self.count_producers - 1

    def publish(self, producer_id, product):
        """
        Lists a product in the marketplace, respecting the producer's queue limit.
        
        Returns:
            bool: True if item was listed, False if quota exceeded.
        """
        logger = logging.getLogger('my_logger')
        logger.info("Product publishing started")

        # Quota Enforcement: Prevents producer overflow.
        if len(self.producer_products[producer_id]) < self.queue_size_per_producer:
            self.producer_products[producer_id].append(product)

            logger.info("Product publishing finished successfully")
            return True

        logger.info("Product publishing: Caller should wait")
        return False

    def new_cart(self):
        """
        Allocates a new transaction context for a consumer.
        """
        logger = logging.getLogger('my_logger')
        logger.info("Cart creation started")

        self.carts.append([])
        logger.info("Cart creation finished")
        return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from a producer's available stock to a consumer's cart.
        Logic: Linear scan through producers to find the first instance of the requested product.
        
        Returns:
            bool: Success status based on item availability.
        """
        logger = logging.getLogger('my_logger')
        logger.info("Product adding in cart started")

        # Block Logic: Scans distributed producer inventories.
        for i in range(self.count_producers):
            if product in self.producer_products[i]:
                # State Transition: Available -> Reserved/In-Cart.
                self.carts[cart_id].append(product)
                self.reserved_products[i].append(product)
                self.producer_products[i].remove(product)
                return True

        logger.info("Product added in cart successfully")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Returns an item from a cart to its source producer's available stock.
        """
        logger = logging.getLogger('my_logger')
        logger.info("Product removing started")

        self.carts[cart_id].remove(product)

        # Block Logic: Locates the reservation record to identify the source producer.
        for i in range(self.count_producers):
            if product in self.reserved_products[i]:
                # State Transition: Reserved -> Available.
                self.reserved_products[i].remove(product)
                self.producer_products[i].append(product)
                return True

        logger.info("Product removing finished")
        return False

    def place_order(self, cart_id):
        """
        Finalizes the transaction for the specified cart.
        """
        logger = logging.getLogger('my_logger')
        logger.info("Order placing finished successfully")
        return self.carts[cart_id]


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base data model for tradable items.
    """
    name: str
    price: int

class TestMarketplace(unittest.TestCase):
    """
    Unit test suite for validating Marketplace operations.
    """
    
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.marketplace = Marketplace(10)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), 0,
                         'wrong producer id')
        self.assertEqual(len(self.marketplace.producer_products), 1,
                         'wrong producer products size')

    def test_publish(self):
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.assertTrue(self.marketplace.publish(0, product),
                        'product not published')
        self.assertEqual(len(self.marketplace.producer_products[0]), 1,
                         'wrong producer products size')

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 0,
                         'wrong cart id')
        self.assertEqual(len(self.marketplace.carts), 1,
                         'wrong carts size')

    def test_add_to_cart(self):
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product),
                        'product not added to cart')
        self.assertEqual(len(self.marketplace.carts[0]), 1,
                         'wrong cart size')
        self.assertEqual(len(self.marketplace.producer_products[0]), 0,
                         'wrong producer products size')

    def test_remove_from_cart(self):
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product),
                        'product not removed from cart')
        self.assertEqual(len(self.marketplace.carts[0]), 0,
                         'wrong cart size')
        self.assertEqual(len(self.marketplace.producer_products[0]), 1,
                         'wrong producer products size')

    def test_place_order(self):
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.place_order(cart_id), [product],
                         'wrong order')


from threading import Thread, Lock
import time


class Producer(Thread):
    """
    Represents a manufacturing source thread that supplies products to the marketplace.
    Operates in a continuous cycle based on its production capacity and item-specific costs.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Args:
            products (list): Collection of (product, quantity, production_time) tuples.
            marketplace (Marketplace): Target for product listings.
            republish_wait_time (float): Back-pressure delay if marketplace is full.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = 0

    def run(self):
        """
        Continuous production loop.
        Logic: Sequential manufacturing of assigned product batches with congestion-aware backoff.
        """
        # Atomic registration.
        lock = Lock()
        lock.acquire()
        self.producer_id = self.marketplace.register_producer()
        lock.release()

        while True:
            # Block Logic: Iterates through the production catalog.
            for product in self.products:
                product_id = product[0]
                quantity = product[1]
                production_time = product[2]
                i = 0

                while i < quantity:
                    # Action: Attempt to list the item.
                    status = self.marketplace.publish(self.producer_id, product_id)
                    if not status:
                        # Backoff: Marketplace buffer full, wait and retry.
                        time.sleep(self.republish_wait_time)
                    else:
                        # Logic: Production time represents the physical cost of generating the item.
                        i += 1
                        time.sleep(production_time)
