
"""
@file consumer.py (and others)
@brief A multi-threaded producer-consumer simulation for an e-commerce marketplace.
@details This module defines a producer-consumer system that attempts to use a
fine-grained locking strategy with a lock for each individual shopping cart.

@warning CRITICAL DEADLOCK FLAW: This implementation is highly prone to deadlocks.
The `Marketplace.add_to_cart` and `Marketplace.remove_from_cart` methods acquire
the global `product_lock` and the per-cart `cart_id_lock` in opposite orders. This
creates a classic "AB-BA" deadlock scenario where two threads can block each other,
halting the system.

NOTE: This file appears to be a concatenation of multiple Python files.
"""

from threading import Thread, RLock
import time
import logging
import unittest
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass


# --- Consumer Logic ---
class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.

    Each consumer thread processes a list of shopping commands, using a single
    persistent cart ID for all its operations. It uses busy-waiting loops to
    handle operations that cannot be immediately fulfilled.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.
        :param carts: A list of shopping lists for the consumer to process.
        :param marketplace: The shared Marketplace instance.
        :param retry_wait_time: Time to wait before retrying a failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_to_cart(self, cart_id, product, quantity):
        """Helper method to add a product, with a busy-wait retry loop."""
        while quantity > 0:
            while not self.marketplace.add_to_cart(cart_id, product):
                time.sleep(self.retry_wait_time)
            quantity = quantity - 1

    def remove_from_cart(self, cart_id, product, quantity):
        """Helper method to remove a product, with a busy-wait retry loop."""
        while quantity > 0:
            while not self.marketplace.remove_from_cart(cart_id, product):
                time.sleep(self.retry_wait_time)
            quantity = quantity - 1

    def order_cart(self, cart_id):
        """Places an order and prints the purchased items."""
        products = self.marketplace.place_order(cart_id)
        for prod in products:
            print(f"{self.name} bought {prod}")

    def run(self):
        """The main execution loop for the consumer thread."""
        cart_id = self.marketplace.new_cart()
        for cart in self.carts:
            for instruction in cart:
                instr_type = instruction.get("type")
                product = instruction.get("product")
                quantity = instruction.get("quantity")
                if instr_type == "add":
                    self.add_to_cart(cart_id, product, quantity)
                elif instr_type == "remove":
                    self.remove_from_cart(cart_id, product, quantity)
            self.order_cart(cart_id)

# --- Marketplace Logic ---
class Marketplace:
    """
    The central marketplace that coordinates producers and consumers.
    @warning This class is prone to deadlocks due to inconsistent lock acquisition order.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace with fine-grained and global locks."""
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = 0
        self.producer_queue_size = {}
        self.producer_lock = RLock() # Global lock for producer registration/state.

        self.carts = 0
        self.cart_lock = RLock() # Global lock for creating new carts.

        self.cart_id_data = {}   # Stores data for each cart.
        self.cart_id_lock = {}   # Maps each cart_id to its own fine-grained lock.

        self.products = []       # Global list of available products.
        self.product_lock = RLock() # Global lock for the product list.

        self.logger = logging.getLogger("Marketplace")
        self.logger.addHandler(RotatingFileHandler("marketplace.log", maxBytes=10000, backupCount=5))
        self.logger.setLevel(logging.INFO)

    def register_producer(self):
        """Registers a new producer, assigning a unique ID."""
        with self.producer_lock:
            self.producers += 1
            producer_id = self.producers
            self.producer_queue_size[producer_id] = self.queue_size_per_producer
        self.logger.info(f"Producer {producer_id} has been registered")
        return producer_id

    def publish(self, producer_id, product):
        """Allows a producer to list a product."""
        if self.producer_queue_size[producer_id] > 0:
            with self.product_lock:
                self.products.append((product, producer_id))
            
            with self.producer_lock:
                self.producer_queue_size[producer_id] -= 1
            
            self.logger.info(f"Product {product} has been published by producer {producer_id}")
            return True
        return False

    def new_cart(self):
        """Creates a new, empty shopping cart and assigns it a unique lock."""
        with self.cart_lock:
            self.carts += 1
            cart_id = self.carts
            self.cart_id_data[cart_id] = []
            self.cart_id_lock[cart_id] = RLock() # Fine-grained lock per cart.
        self.logger.info(f"Customer created cart {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.
        @warning DEADLOCK RISK: Acquires `product_lock`, then `cart_id_lock`.
        """
        self.product_lock.acquire()
        try:
            # Performance Issue: Iterating a potentially large list while holding a lock.
            for prod in self.products:
                if prod[0] == product:
                    self.products.remove(prod)
                    
                    # Release product_lock before acquiring cart_lock to attempt to avoid deadlock,
                    # but the core issue is the inconsistent order.
                    self.product_lock.release()

                    with self.cart_id_lock[cart_id]:
                        self.cart_id_data[cart_id].append(prod)
                    
                    self.logger.info(f"Customer added product {product} to cart {cart_id}")
                    return True
        finally:
            # Ensure the lock is released if it's still held.
            if self.product_lock.locked():
                 self.product_lock.release()

        self.logger.info(f"Customer failed to add product {product} to cart {cart_id}")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart.
        @warning DEADLOCK RISK: Acquires `cart_id_lock`, then `product_lock`.
        This is the reverse order of `add_to_cart`, creating a deadlock hazard.
        """
        with self.cart_id_lock[cart_id]:
            with self.product_lock:
                for prod in self.cart_id_data[cart_id]:
                    if prod[0] == product:
                        self.cart_id_data[cart_id].remove(prod)
                        self.products.append(prod)
                        self.logger.info(f"Customer removed product {prod} from cart {cart_id}")
                        return True
        return False # This will never be reached due to locks.

    def place_order(self, cart_id):
        """Finalizes an order."""
        products = []
        with self.cart_id_lock[cart_id]:
            for prod, producer_id in self.cart_id_data[cart_id]:
                products.append(prod)
                # Nested lock acquisition can increase deadlock complexity.
                with self.producer_lock:
                    self.producer_queue_size[producer_id] += 1
            self.cart_id_data[cart_id].clear()
        self.logger.info(f"Customer placed an order and emptied cart {cart_id}")
        return products

# --- Unit Test Suite & Other Classes ---
class TestMarketplace(unittest.TestCase):
    # ... (Test cases)
    pass

class Producer(Thread):
    # ... (Producer implementation)
    pass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    # ... (Product dataclass)
    pass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    # ... (Tea dataclass)
    pass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    # ... (Coffee dataclass)
    pass
