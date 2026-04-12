"""
Module providing a multi-threaded simulation of a retail marketplace.
Uses specialized worker threads (Producers and Consumers) coordinated by a 
central Marketplace entity that ensures data integrity through explicit locking mechanisms.
"""

from threading import Thread
from time import sleep


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
        self.wait_time = retry_wait_time
        self.cart_ids = []

    def run(self):
        """
        Executes the shopping lifecycle for each assigned cart.
        Logic: Sequential processing of operations (add/remove) with busy-wait retries for stock availability.
        """
        index = 0
        # Block Logic: Iterates through each independent shopping session.
        for cart in self.carts:
            self.cart_ids.append(self.marketplace.new_cart())
            # Block Logic: Processes individual product operations within the current cart.
            for op_cart in cart:
                product = op_cart['product']
                quantity = op_cart['quantity']
                op_type = op_cart['type']
                
                if op_type == "add":
                    i = 0
                    # Block Logic: Indefinite retry loop for product acquisition.
                    # Invariant: Consumer waits for the marketplace to fulfill the request before moving to the next item.
                    while i < quantity:
                        status = self.marketplace.add_to_cart(self.cart_ids[index], product)
                        if status:
                            i += 1
                        else:
                            # Backoff: Wait for producers to replenish stock.
                            sleep(self.wait_time)
                elif op_type == "remove":
                    for i in range(0, quantity):
                        self.marketplace.remove_from_cart(self.cart_ids[index], product)
            
            # Finalization: Converts the cart contents into a finalized transaction log.
            self.marketplace.place_order(self.cart_ids[index])
            index += 1

import threading
import time
from threading import Lock
import logging.handlers
import unittest

from tema.product import Tea


class TestMarketplace(unittest.TestCase):
    """
    Unit test suite for validating Marketplace operations.
    """

    def setUp(self):
        self.marketplace = Marketplace(5)
        self.product1 = Tea('Test_tea1', 0, 'Test_tea2')
        self.product2 = Tea('Test_tea3', 0, 'Test_tea4')

    def test_register_producer(self):
        prod_id = self.marketplace.last_producer_id
        self.assertEqual(self.marketplace.register_producer(),
                         prod_id)

    def test_publish(self):
        id_prod = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(id_prod,
                                                 self.product1))
        self.assertEqual(len(self.marketplace.producers[0]), 1)

    def test_new_cart(self):
        cart_id = self.marketplace.last_cart_id
        self.assertEqual(self.marketplace.new_cart(), cart_id)

    def test_add_to_cart(self):
        cart_id = self.marketplace.new_cart()
        cart_len = len(self.marketplace.carts[cart_id])
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.product1))
        self.assertGreater(len(self.marketplace.carts[cart_id]), cart_len)

    def test_remove_from_cart(self):
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.product1)
        cart_len = len(self.marketplace.carts[cart_id])
        self.marketplace.remove_from_cart(cart_id, self.product1)
        self.assertLess(len(self.marketplace.carts[cart_id]), cart_len)

    def test_place_order(self):
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.product1)
        self.marketplace.add_to_cart(cart_id, self.product2)
        cart_len = len(self.marketplace.carts[cart_id])
        self.assertEqual(len(self.marketplace.place_order(cart_id)), cart_len)


class Marketplace:
    """
    Acts as a thread-safe coordinator for producers and consumers.
    Manages global product availability, individual producer quotas, and consumer cart state.
    Utilizes dual locks to minimize contention between producer and consumer registries.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace with capacity limits and synchronization primitives.
        """
        self.queue_max_size = queue_size_per_producer
        
        # State: Registry of producers and their current inventory.
        self.producers = []
        
        # State: Active shopping sessions.
        self.carts = []
        self.last_producer_id = 0
        self.last_cart_id = 0
        
        # Synchronization: Fine-grained locks for different shared domains.
        self.lock1 = Lock() # Protects producer registry.
        self.lock2 = Lock() # Protects cart registry.
        
        # Telemetry: Configures rotating logs for audit trails.
        logging.basicConfig(handlers=[logging.handlers.RotatingFileHandler("marketplace.log",
                                                                           mode='a',
                                                                           maxBytes=5000,
                                                                           backupCount=5)],
                            level=logging.INFO,
                            format=
                            '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')
        logging.Formatter.converter = time.gmtime

    def register_producer(self):
        """
        Registers a new supply source in the system.
        """
        logging.info("Entered register_producer")
        self.producers.append([])
        with self.lock1:
            self.last_producer_id += 1
            id_prod = self.last_producer_id - 1

        logging.info("New producer id: " + str(id_prod))
        return id_prod

    def publish(self, producer_id, product):
        """
        Adds a product to the marketplace inventory, respecting the producer's quota.
        
        Args:
            producer_id (int): Source of the product.
            product (Product): Item to be listed.
            
        Returns:
            bool: True if item was listed, False if quota exceeded.
        """
        logging.info("Entered publish with producer id " + str(producer_id)
                     + " and product " + str(product))
        
        # Pre-condition: Check if producer has reached their listing capacity.
        if len(self.producers[producer_id]) == self.queue_max_size:
            logging.info("Return value: False")
            return False
            
        with self.lock1:
            self.producers[producer_id].append((product, 1))
        logging.info("Return value: True")
        return True

    def new_cart(self):
        """
        Allocates a new transaction context for a consumer.
        """
        logging.info("Entered new_cart")
        with self.lock2:
            self.last_cart_id += 1
            self.carts.append([])
            cart_id = self.last_cart_id - 1

        logging.info("Returned cart id: " + str(cart_id))
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from the global market to a consumer's cart.
        Logic: Scans producers for the product and marks it as unavailable (state 0).
        
        Returns:
            bool: Success status based on item availability.
        """
        logging.info("Entered add_to_cart with cart id " + str(cart_id)
                     + " and product " + str(product))
        
        # Block Logic: Scans all producer inventories for the requested item.
        for tmp_list in self.producers:
            for prod_tuple in tmp_list:
                tmp = list(prod_tuple)
                if tmp[0] == product:
                    if tmp[1] == 0:
                        # Invariant: Item is currently held in another cart or sold.
                        logging.info("Return value: False")
                        return False
                    
                    # Atomic state transition: Mark as reserved.
                    with self.lock1:
                        tmp[1] = 0
                        prod_tuple = tuple(tmp)

        # Registry update: Append to consumer session.
        with self.lock2:
            self.carts[cart_id].append(product)
        logging.info("Return value: True")
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Returns an item from a cart to the global inventory.
        """
        logging.info("Entered remove_from_cart with cart id " +
                     str(cart_id) + " and product " + str(product))
        with self.lock2:
            self.carts[cart_id].remove(product)

        # Restoration: Locate the source and mark the item as available again (state 1).
        for tmp_list in self.producers:
            for prod_tuple in tmp_list:
                tmp = list(prod_tuple)
                if tmp[0] == product:
                    with self.lock1:
                        tmp[1] = 1
                        prod_tuple = tuple(tmp)

    def place_order(self, cart_id):
        """
        Finalizes the order by logging the collection of products.
        """
        logging.info("Entered place_order with cart id " + str(cart_id))
        prod_list = self.carts[cart_id]
        
        # Cleanup: Remove finalized items from the producers' tracking lists.
        for prod_iter in prod_list:
            for list_prod in self.producers:
                if prod_iter in list_prod:
                    with self.lock1:
                        list_prod.remove(prod_iter)
        
        self.carts[cart_id] = []
        for prod_iter in prod_list:
            # Functional Utility: Console output for simulation tracking.
            print(threading.current_thread().name + " bought " + str(prod_iter))
            logging.info("Buyer " + threading.current_thread().name
                         + " bought " + str(prod_iter))
        return prod_list


from threading import Thread
from time import sleep


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
        self.wait_time = republish_wait_time
        self.id_prod = -1

    def run(self):
        """
        Continuous production loop.
        Logic: Sequential manufacturing of assigned product batches with congestion-aware backoff.
        """
        self.id_prod = self.marketplace.register_producer()
        index = 0
        while True:
            # Cycle Logic: Continuous iteration through the product catalog.
            if index == len(self.products):
                index = 0
            
            i = 0
            # Block Logic: Sequential manufacturing of item quantity.
            while i < self.products[index][1]:
                status = self.marketplace.publish(self.id_prod, self.products[index][0])
                if not status:
                    # Backoff: Marketplace buffer full, wait and retry.
                    sleep(self.wait_time)
                else:
                    # Logic: Production time represents the physical cost of generating the item.
                    sleep(self.products[index][2])
                    i += 1

            index += 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base data model for any tradable item in the simulation.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Specialized product type representing tea variations.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Specialized product type representing coffee variations.
    """
    acidity: str
    roast_level: str
