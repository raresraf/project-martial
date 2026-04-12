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
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        Executes the shopping lifecycle for each assigned cart.
        Logic: Sequential processing of operations (add/remove) with busy-wait retries for stock availability.
        """
        # Block Logic: Iterates through each independent shopping session.
        for cart in self.carts:
            id_cart = self.marketplace.new_cart()
            # Block Logic: Processes individual product operations within the current cart.
            for curr_op in cart:
                for _ in range(curr_op["quantity"]):
                    if curr_op["type"] == "add":
                        # Block Logic: Indefinite retry loop for product acquisition.
                        # Invariant: Consumer waits for the marketplace to fulfill the request before moving to the next item.
                        while True:
                            check = self.marketplace.add_to_cart(id_cart, curr_op["product"])
                            if not check:
                                sleep(self.retry_wait_time)
                            else:
                                break
                    elif curr_op["type"] == "remove":
                        self.marketplace.remove_from_cart(id_cart, curr_op["product"])
            
            # Finalization: Converts the cart contents into a finalized transaction log.
            self.marketplace.place_order(id_cart)

import threading
import unittest
import logging
import logging.handlers
from tema.product import Product

LOG_FILE_NAME = 'marketplace.log'
LOGGING_LEVEL = logging.DEBUG

class Marketplace:
    """
    Acts as a thread-safe coordinator for producers and consumers.
    Manages global product availability, individual producer quotas, and consumer cart state.
    Utilizes multiple fine-grained locks to minimize contention during concurrent operations.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace with capacity limits and synchronization primitives.
        """
        self.allowed_quantity = queue_size_per_producer
        self.producers = {} # Map of producer_id to current stock count.
        self.last_prod_id = 0
        self.last_cart_id = 0
        self.consumers = {} # Map of cart_id to list of products.
        self.market = {}    # Reverse lookup: product -> producer_id.
        self.all_products = [] # Global available inventory.

        # Synchronization: Specific locks for different marketplace domains.
        self.reg_p_lock = threading.Lock()
        self.publish_lock = threading.Lock()
        self.new_cart_lock = threading.Lock()
        self.add_to_cart_lock = threading.Lock()
        
        # Telemetry: Configures timed rotating logs for system auditing.
        self.logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE_NAME,
                                                            interval=30, backupCount=10)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(LOGGING_LEVEL)

    def register_producer(self):
        """
        Registers a new supply source in the system.
        
        Returns:
            int: Unique identifier for the producer.
        """
        self.reg_p_lock.acquire()
        self.logger.info("-- Intrare in metoda register_producer")
        self.last_prod_id += 1
        self.producers[self.last_prod_id] = 0
        self.reg_p_lock.release()
        self.logger.info("-- Iesire din metoda register_producer")
        return self.last_prod_id

    def publish(self, producer_id, product):
        """
        Adds a product to the marketplace inventory, respecting the producer's quota.
        
        Args:
            producer_id (int): Source of the product.
            product (Product): Item to be listed.
            
        Returns:
            bool: True if item was listed, False if quota exceeded.
        """
        self.publish_lock.acquire()
        self.logger.info("-- Intrare in metoda publish cu param producer_id = %s si product = %s",
                         str(producer_id), str(product))
        
        # Update logic: Tracks which producer provided each item.
        self.market[product] = producer_id
        self.all_products.append(product)
        old_val = self.producers.get(producer_id)
        self.producers[producer_id] = old_val + 1
        self.publish_lock.release()

        # Quota Enforcement: Roll back if listing exceeds the allowed buffer size.
        if self.producers[producer_id] > self.allowed_quantity:
            self.producers[producer_id] = old_val
            self.logger.info("-- Iesire din metoda publish cu rezultatul False")
            return False

        self.logger.info("-- Iesire din metoda publish cu rezultatul True")
        return True

    def new_cart(self):
        """
        Allocates a new transaction context for a consumer.
        
        Returns:
            int: Unique cart identifier.
        """
        self.new_cart_lock.acquire()
        self.logger.info("-- Intrare in metoda new_cart")
        self.last_cart_id += 1
        self.consumers[self.last_cart_id] = []
        self.logger.info("-- Iesire din metoda new_cart")
        self.new_cart_lock.release()
        return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from the global market to a consumer's cart.
        
        Returns:
            bool: Success status based on item availability.
        """
        self.add_to_cart_lock.acquire()
        self.logger.info("-- Intrare in metoda add_to_cart cu param cart_id = %s si product = %s",
                         str(cart_id), str(product))
        
        # Pre-condition: Item must be present in the global inventory.
        if product not in self.market or product not in self.all_products:
            self.logger.info("-- Iesire din metoda add_to_cart -> Nu exista produsul")
            self.add_to_cart_lock.release()
            return False

        # Atomic Transfer logic.
        self.consumers[cart_id].append(product)
        prod = self.market[product]
        self.all_products.remove(product)
        self.producers[prod] -= 1

        self.logger.info("-- Iesire triumfatoare din add_to_cart")
        self.add_to_cart_lock.release()
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Returns an item from a cart to the global inventory.
        """
        self.logger.info("-- Intrare in metoda remove_from_cart cu param cart_if = %s si %s",
                         str(cart_id), str(product))
        self.consumers[cart_id].remove(product)
        prod = self.market.get(product)
        self.all_products.append(product)
        self.producers[prod] += 1
        self.logger.info("-- Iesire din metoda remove_from_cart")

    def place_order(self, cart_id):
        """
        Finalizes the order by logging the collection of products.
        """
        self.logger.info("-- Intrare in metoda place_order cu param cart_id = %s ", str(cart_id))
        for prod in self.consumers[cart_id]:
            # Functional Utility: Console output for simulation tracking.
            print(threading.currentThread().getName() + " bought " + str(prod))
        self.logger.info("-- Iesire din metoda place_order")
        return self.consumers[cart_id]

class TestMarketPlace(unittest.TestCase):
    """
    Unit test suite for validating Marketplace operations.
    """
    
    def test_register_producer(self):
        market = Marketplace(12)
        self.assertEqual(1, market.register_producer())

    def test_publish(self):
        market = Marketplace(2)
        market.register_producer()
        self.assertEqual(True, market.publish(1, Product("TeaName", 23.0)))

    def test_new_cart(self):
        market = Marketplace(2)
        first_cart = market.new_cart()
        second_cart = market.new_cart()
        self.assertEqual(1, first_cart)
        self.assertEqual(2, second_cart)

    def test_add_to_cart(self):
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutz", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutz", 5.0))
        self.assertEqual(0, len(market.all_products))

    def test_remove_from_cart(self):
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.remove_from_cart(cart_id, Product("Cafelutzaaa", 5.0))
        self.assertEqual(1, len(market.all_products))

    def test_place_order(self):
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.publish(1, Product("Ceiut", 3.0))
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.publish(1, Product("Ceiut", 3.0))
        market.publish(1, Product("Altceva", 3.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Ceiut", 3.0))
        self.assertEqual(3, len(market.place_order(cart_id)))


from threading import Thread
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
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.published_products = 0
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        Continuous production loop.
        Logic: Sequential manufacturing of assigned product batches with congestion-aware backoff.
        """
        id_prod = self.marketplace.register_producer()

        while 1:
            # Block Logic: Iterates through the production catalog.
            for (product, quantity, production_time) in self.products:
                for i in range(quantity):
                    # Action: Attempt to list the item.
                    check = self.marketplace.publish(id_prod, product)
                    if check:
                        # Logic: Production time represents the physical cost of generating the item.
                        time.sleep(production_time)
                    else:
                        # Backoff: Marketplace buffer full, wait and retry.
                        time.sleep(self.republish_wait_time)
                        i += 1
                    i -= 1


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
