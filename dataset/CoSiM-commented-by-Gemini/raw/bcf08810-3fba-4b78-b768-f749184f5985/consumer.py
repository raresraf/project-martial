"""
Module providing a multi-threaded simulation of a retail marketplace.
Uses specialized worker threads (Producers and Consumers) coordinated by a 
central Marketplace entity that ensures data integrity through explicit locking mechanisms.
"""

from time import sleep
from threading import Thread


class Consumer(Thread):
    """
    Represents an automated shopper thread that executes a series of shopping sessions.
    Orchestrates the acquisition and removal of products across multiple carts, 
    waiting for stock availability when necessary.
    """
    
    carts = []
    marketplace = None
    retry_wait_time = -1
    name = None

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer with shopping tasks.
        
        Args:
            carts (list): List of operations per cart.
            marketplace (Marketplace): The central broker for transactions.
            retry_wait_time (float): Seconds to wait if a desired product is out of stock.
            **kwargs: Thread configuration options (requires 'name').
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']
        

    def run(self):
        """
        Executes the shopping lifecycle for each assigned cart.
        Logic: Sequential processing of operations (add/remove) with busy-wait retries for stock availability.
        """
        # Block Logic: Iterates through each independent shopping session.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            # Block Logic: Processes individual product operations within the current cart.
            for cmd in cart:
                cmd_type = cmd['type']
                product = cmd['product']
                quantity = cmd['quantity']

                if cmd_type == 'add':
                    i = 0
                    # Block Logic: Indefinite retry loop for product acquisition.
                    # Invariant: Consumer waits for the marketplace to fulfill the request before moving to the next item.
                    while i < quantity:
                        product_added = self.marketplace.add_to_cart(cart_id, product)
                        
                        if product_added:
                            i += 1
                        else:
                            # Backoff: wait for producers to replenish global inventory.
                            sleep(self.retry_wait_time)
                elif cmd_type == 'remove':
                    # Action: Return reserved products back to the marketplace.
                    for i in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

            # Finalization: Converts the cart contents into a finalized transaction log.
            products = self.marketplace.place_order(cart_id)
            for i in products:
                # Functional Utility: Console output for simulation tracking.
                print(self.name + ' bought ' + str(i))

import time
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import unittest
from random import randint
from tema.product import Product, Tea, Coffee


class TestMarketplace(unittest.TestCase):
    """
    Unit test suite for validating Marketplace operations.
    """

    def setUp(self):
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        old_id = -1
        new_id = -1
        for _ in range(randint(3, 100)):
            old_id = self.marketplace.producers_ids
            new_id = self.marketplace.register_producer()
        self.assertEqual(old_id + 1, new_id)

    def test_new_cart(self):
        old_cart_id = -1
        new_cart_id = -1
        for _ in range(randint(3, 100)):
            old_cart_id = self.marketplace.carts_ids
            new_cart_id = self.marketplace.new_cart()
        self.assertEqual(old_cart_id + 1, new_cart_id)

    def test_publish_true(self):
        max_len = self.marketplace.queue_size_per_producer
        id_prod = self.marketplace.register_producer()
        for _ in range(randint(0, max_len - 2)):
            published = self.marketplace.publish(id_prod, Tea('test_tea', 10, 'test_type'))
            self.assertTrue(published)

    def test_publish_false(self):
        published = False
        max_len = self.marketplace.queue_size_per_producer
        id_prod = self.marketplace.register_producer()
        for _ in range(randint(max_len + 1, 2 * max_len)):
            published = self.marketplace.publish(id_prod, Tea('test_tea', 10, 'test_type'))
        self.assertFalse(published)

    def test_add_to_cart_true(self):
        cart = self.marketplace.new_cart()
        id1 = self.marketplace.register_producer()
        id2 = self.marketplace.register_producer()
        product = Tea('test_tea', 10, 'test_type')
        self.marketplace.publish(id1, product)
        self.marketplace.publish(id2, product)
        found = self.marketplace.add_to_cart(cart, product)
        self.assertTrue(found)

    def test_add_to_cart_false(self):
        cart = self.marketplace.new_cart()
        id1 = self.marketplace.register_producer()
        id2 = self.marketplace.register_producer()
        product1 = Tea('test_tea', 10, 'test_type')
        product2 = Coffee('test_coffee', 20, 'test', 'test')
        self.marketplace.publish(id1, product1)
        self.marketplace.publish(id2, product1)
        found = self.marketplace.add_to_cart(cart, product2)
        self.assertFalse(found)

    def test_remove_from_cart(self):
        cart = self.marketplace.new_cart()
        id1 = self.marketplace.register_producer()
        product = Tea('test_tea', 10, 'test_type')
        self.marketplace.publish(id1, product)
        found = self.marketplace.add_to_cart(cart, product)
        self.assertTrue(found)
        dim_before = len(self.marketplace.carts[cart])
        self.marketplace.remove_from_cart(cart, product)
        dim_after = len(self.marketplace.carts[cart])
        self.assertTrue(dim_before > dim_after)

    def test_place_order(self):
        c_1 = self.marketplace.new_cart()
        c_2 = self.marketplace.new_cart()
        id_1 = self.marketplace.register_producer()
        id_2 = self.marketplace.register_producer()
        p_1 = Tea('test_tea', 10, 'test_type')
        p_2 = Coffee('test_coffee', 20, 'test', 'test')
        self.marketplace.publish(id_1, p_2)
        self.marketplace.publish(id_2, p_1)
        self.marketplace.publish(id_2, p_2)
        self.marketplace.publish(id_1, p_1)
        self.marketplace.add_to_cart(c_2, p_2)
        self.marketplace.add_to_cart(c_1, p_1)
        self.marketplace.add_to_cart(c_1, p_2)
        self.marketplace.add_to_cart(c_1, p_1)
        prod = self.marketplace.place_order(c_1)
        self.assertEqual(len(prod), 3)
        self.assertEqual(prod[0], p_1)
        self.assertEqual(prod[1], p_2)
        self.assertEqual(prod[2], p_1)


class Marketplace:
    """
    Acts as a thread-safe coordinator for producers and consumers.
    Manages global product availability, individual producer quotas, and consumer cart state.
    Utilizes explicit locking to ensure atomic updates to shared registries.
    """
    
    queue_size_per_producer = -1
    producers_ids = -1
    carts_ids = -1
    
    # State: Storage for producer inventory and consumer sessions.
    producers_queues = {} # producer_id -> [[product, owner_cart_id], ...]
    carts = {} # cart_id -> [(product, source_producer_id), ...]
    
    # Synchronization: Global lock for registry modifications.
    lock = Lock()

    # Telemetry: Configures rotating logs for audit trails.
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        handlers=[RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=10)],
        level=logging.INFO,
        format="[%(asctime)s] - [%(levelname)s] : %(funcName)s:%(lineno)d -> %(message)s",
        datefmt='%Y-%m-%d  %H:%M:%S'
    )

    def __init__(self, queue_size_per_producer):
        """
        Args:
            queue_size_per_producer (int): Buffer limit for each producer.
        """
        self.queue_size_per_producer = queue_size_per_producer
        

    def register_producer(self):
        """
        Registers a new supply source.
        
        Returns:
            int: Unique identifier for the producer.
        """
        logging.info('ENTER')
        with self.lock:
            self.producers_ids += 1
            new_id = self.producers_ids
        self.producers_queues[new_id] = []
        logging.info('EXIT')
        return new_id

    def publish(self, producer_id, product):
        """
        Adds a product to the marketplace, respecting producer limits.
        
        Args:
            producer_id (int): source identifier.
            product (Product): item to be listed.
            
        Returns:
            bool: True if listing succeeded, False if queue is full.
        """
        logging.info('ENTER\n %s %s', str(producer_id), str(product))
        if len(self.producers_queues[producer_id]) < self.queue_size_per_producer:
            # Logic: items are initialized with owner -1 (available).
            item = [product, -1]
            self.producers_queues[producer_id].append(item)
            logging.info('EXIT')
            return True
        logging.info('EXIT')
        return False

    def new_cart(self):
        """
        Allocates a new transaction context for a consumer.
        
        Returns:
            int: Unique cart identifier.
        """
        logging.info('ENTER')
        with self.lock:
            self.carts_ids += 1
            new_cart = self.carts_ids
        self.carts[new_cart] = []
        logging.info('EXIT')
        return new_cart

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from global availability to a consumer's cart.
        Logic: Scans all producers for the first available instance of the product.
        
        Returns:
            bool: True if item was found and reserved, False otherwise.
        """
        logging.info('ENTER\n %s %s', str(cart_id), str(product))
        # Block Logic: Scans distributed producer inventories.
        for key, value in self.producers_queues.items():
            for product_tuple in value:
                with self.lock:
                    # Atomic Reservation: Mark available item (-1) with the requester's cart_id.
                    if product_tuple[0] == product and product_tuple[1] == -1:
                        product_tuple[1] = cart_id
                        self.carts[cart_id].append((product, key))
                        logging.info('EXIT')
                        return True
        logging.info('EXIT')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Returns a reserved item back to the global marketplace.
        """
        logging.info('ENTER\n %s %s', str(cart_id), str(product))
        # Block Logic: Locate the specific instance in the consumer's cart.
        for product_tuple in self.carts.get(cart_id):
            if product_tuple[0] == product:
                producer = product_tuple[1]
                # Restoration: Release ownership back to available state (-1).
                for item in self.producers_queues[producer]:
                    with self.lock:
                        if item[0] == product and item[1] == cart_id:
                            item[1] = -1
                            break
                self.carts[cart_id].remove(product_tuple)
                break
        logging.info('EXIT')

    def place_order(self, cart_id):
        """
        Finalizes the purchase process.
        Logic: removes the reserved items from the producers' inventories permanently.
        
        Returns:
            list: The collection of purchased products.
        """
        logging.info('ENTER\n %s', str(cart_id))
        products = []
        for product_tuple in self.carts[cart_id]:
            product = product_tuple[0]
            producer_id = product_tuple[1]
            products.append(product)
            # Cleanup: Physical removal from the supply source.
            for item in self.producers_queues[producer_id]:
                with self.lock:
                    if item[0] == product and item[1] == cart_id:
                        self.producers_queues[producer_id].remove(item)
                        break
        logging.info('EXIT')
        return products

from time import sleep
from threading import Thread


class Producer(Thread):
    """
    Represents a manufacturing source thread that supplies products to the marketplace.
    Operates in a continuous cycle based on its production capacity and item-specific costs.
    """
    
    p_id = -1
    products = []
    marketplace = None
    republish_wait_time = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Args:
            products (list): Collection of (product, quantity, production_time) tuples.
            marketplace (Marketplace): Target for product listings.
            republish_wait_time (float): Back-pressure delay if marketplace is full.
        """
        Thread.__init__(self, **kwargs)
        self.p_id = marketplace.register_producer()
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        

    def run(self):
        """
        Continuous production loop.
        Logic: Sequential manufacturing of assigned product batches with congestion-aware backoff.
        """
        while True:
            # Block Logic: Iterates through the manufacturing schedule.
            for product in self.products:
                product_type = product[0]
                quantity = product[1]
                production_time = product[2]

                i = 0
                while i < quantity:
                    # Action: Attempt to list the item.
                    published = self.marketplace.publish(self.p_id, product_type)
                    if published:
                        # Logic: Production time represents the physical cost of generating the item.
                        sleep(production_time)
                        i += 1
                    else:
                        # Backoff: Marketplace buffer full, wait and retry.
                        sleep(self.republish_wait_time)
