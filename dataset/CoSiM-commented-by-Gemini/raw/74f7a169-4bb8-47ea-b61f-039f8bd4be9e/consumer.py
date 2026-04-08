"""
This module simulates a multi-threaded producer-consumer marketplace.

It contains classes for `Consumer`, `Producer`, and a central `Marketplace`.
It also includes unit tests and product data classes.

Note: This file appears to be a concatenation of several separate modules.
The Marketplace implementation has significant thread-safety issues due to a
lack of locking in critical methods.
"""
from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a shopping process.

    Each consumer is initialized with a list of shopping requests and processes
    them by creating carts, adding/removing items, and placing orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping action lists.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed action.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)


        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def add_product(self, cart_id, product, quantity):
        """
        Helper method to add a specified quantity of a product to a cart.

        Retries with a delay if the product cannot be added immediately.
        """
        for _ in range(quantity):
            while not self.marketplace.add_to_cart(cart_id, product):
                sleep(self.retry_wait_time)

    def remove_product(self, cart_id, product, quantity):
        """Helper method to remove a specified quantity of a product from a cart."""
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """The main execution logic for the consumer thread."""
        # Invariant: Process each list of requests as a separate shopping session.
        for cart in self.carts:
            # A new cart is created for each session.
            cart_id = self.marketplace.new_cart()

            # Invariant: Process each request in the current shopping session.
            for request in cart:
                command = request["type"]
                product = request["product"]
                quantity = request["quantity"]

                # Block Logic: Execute add or remove actions based on the request.
                if command == "add":
                    self.add_product(cart_id, product, quantity)
                elif command == "remove":
                    self.remove_product(cart_id, product, quantity)

            # Finalize the transaction for the current cart.
            order = self.marketplace.place_order(cart_id)

            # Print the contents of the finalized order.
            self.marketplace.print_order(order, self.name)

# --- Start of concatenated Marketplace, Testing, and Producer modules ---

import logging
import time
import unittest
from logging.handlers import RotatingFileHandler
from threading import Lock, current_thread


class Marketplace:
    """
    Manages producers, products, and carts in a simulated marketplace.

    @warning: This class is NOT thread-safe. While it uses a mutex for some
    operations (`register_producer`, `new_cart`), critical methods like
    `publish`, `add_to_cart`, and `remove_from_cart` do not use locks and
    are susceptible to race conditions.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): Max products a producer can have in stock.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0  
        self.cart_id = 0  
        self.producers = {}  # Maps producer ID to a list of their products.
        self.carts = {}      # Maps cart ID to a list of products in that cart.
        self.products = {}   # Maps a product to its producer ID.
        self.mutex = Lock()  

        
        self.logger = logging.getLogger("Logger")
        self.handler = RotatingFileHandler("marketplace.log", maxBytes=25000, backupCount=10)
        self.handler.setLevel(logging.INFO)
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s: %(message)s"))
        logging.Formatter.converter = time.gmtime
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.

        Returns:
            str: The unique ID assigned to the new producer.
        """
        with self.mutex:
            self.producers[self.producer_id] = []
            self.producer_id += 1
            self.logger.info("Thread %s has producer_id=%d", current_thread().name,
                             self.producer_id - 1)
            return str(self.producer_id - 1)

    def publish(self, producer_id, product):
        """
        Adds a product to a producer's inventory.

        Note: This method is not thread-safe.

        Returns:
            bool: True if published, False if producer's stock is full.
        """
        self.logger.info("Thread %s has producer_id=%s, product=%s", current_thread().name,
                         producer_id, product)
        producer_index = int(producer_id)

        
        if len(self.producers[producer_index]) == self.queue_size_per_producer:
            return False

        
        self.producers[producer_index].append(product)
        self.products[product] = producer_index

        return True

    def new_cart(self):
        """
        Creates a new, empty cart for a consumer.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.mutex:
            self.carts[self.cart_id] = []
            self.cart_id += 1
            self.logger.info("Thread %s has cart_id=%s", current_thread().name, self.cart_id - 1)
            return self.cart_id - 1

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from a producer's stock to a consumer's cart.

        Note: This method is not thread-safe.

        Returns:
            bool: True if the product was found and moved, False otherwise.
        """
        self.logger.info("Thread %s has cart_id=%d, product=%s", current_thread().name,
                         cart_id, product)

        for list_of_products in self.producers.values():
            # This check-then-act sequence is a classic race condition.
            if product in list_of_products:
                list_of_products.remove(product)
                self.carts[cart_id].append(product)
                return True

        return False

    def remove_from_cart(self, cart_id, product):
        """
        Moves a product from a consumer's cart back to the original producer's stock.

        Note: This method is not thread-safe.

        Returns:
            bool: True if the product was found and returned to stock, False otherwise.
        """
        self.logger.info("Thread %s has cart_id=%d, product=%s", current_thread().name,
                         cart_id, product)

        
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            # Returns the product to the producer it originally came from.
            self.producers[self.products[product]].append(product)
            return True

        return False

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the cart's contents and deleting the cart.

        Note: This is a destructive read. The cart is removed after the order is placed.
        This method is not thread-safe.

        Returns:
            list: The list of products that were in the cart.
        """
        self.logger.info("Thread %s has cart_id=%d", current_thread().name, cart_id)

        
        cart_content = self.carts[cart_id]
        self.carts.pop(cart_id)

        self.logger.info("Thread %s has cart_content=%s", current_thread().name, cart_content)
        return cart_content

    def print_order(self, order, name):
        """Thread-safely prints the contents of a finalized order."""
        with self.mutex:
            self.logger.info("Thread %s has order=%s, name=%s", current_thread().name,
                             order, name)
            for product in order:
                print(f"{name} bought {product}")


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace logic."""
    def setUp(self):
        """Initializes a marketplace for each test."""
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        """Tests that producer registration provides sequential string-based IDs."""
        for producer_id in range(100):
            self.assertEqual(self.marketplace.register_producer(), str(producer_id), "wrong id")

    def test_publish(self):
        """Tests that publishing respects the producer's queue size limit."""
        prod_id = self.marketplace.register_producer()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(product1 in self.marketplace.producers[int(prod_id)],
                        "product is not on the marketplace")
        self.assertEqual(self.marketplace.products[product1], int(prod_id),
                         "don't recognize the product")

        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")
        self.assertTrue(product2 in self.marketplace.producers[int(prod_id)],
                        "product is not on the marketplace")
        self.assertEqual(self.marketplace.products[product2], int(prod_id),
                         "don't recognize the product")

        self.assertFalse(self.marketplace.publish(prod_id, product3), "failed not to publish")

    def test_new_cart(self):
        """Tests that new cart creation provides sequential IDs."""
        for cart_id in range(100):
            self.assertEqual(self.marketplace.new_cart(), cart_id, "wrong id")

    def test_add_to_cart(self):
        """Tests that products can be successfully added to and removed from stock."""
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")
        self.assertFalse(self.marketplace.publish(prod_id, product3), "failed not to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(product1 in self.marketplace.carts[cart_id], "is not in the cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")
        self.assertTrue(product2 in self.marketplace.carts[cart_id], "is not in the cart")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product3),
                         "product should not be in the market")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product1),
                         "product should be already in the cart")

    def test_remove_from_cart(self):
        """Tests that removing a product returns it to the producer's stock."""
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")

        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product1), "not in the cart")
        self.assertTrue(product1 not in self.marketplace.carts[cart_id], "is in cart")
        self.assertTrue(product1 in self.marketplace.producers[int(prod_id)],
                        "not in producer's list")
        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product2), "not in the cart")
        self.assertTrue(product2 not in self.marketplace.carts[cart_id], "is in cart")
        self.assertTrue(product2 in self.marketplace.producers[int(prod_id)],
                        "not in producer's list")
        self.assertFalse(self.marketplace.remove_from_cart(cart_id, product3), "not in the cart")

    def test_place_order(self):
        """Tests that placing an order consumes the cart and returns its contents."""
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")

        self.assertEqual(self.marketplace.place_order(cart_id), ["coffee", "tea"],
                         "not the same order")


class Producer(Thread):
    """
    Represents a producer thread that continuously adds products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread.

        Args:
            products (list): A list of (product, quantity, sleep_time) tuples.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish.
            **kwargs: Additional keyword arguments for the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main execution loop for the producer."""
        while True:
            # Invariant: Continuously iterate through the assigned product list.
            for product in self.products:
                
                product_id = product[0]
                quantity = product[1]
                production_time = product[2]
                
                # Block Logic: Publish the specified quantity of the current product.
                for _ in range(quantity):
                    
                    # Pre-condition: If publishing fails (queue is full), wait and retry.
                    while not self.marketplace.publish(self.producer_id, product_id):
                        sleep(self.republish_wait_time)

                    sleep(production_time)

# --- Start of concatenated Product dataclasses ---
from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple dataclass representing a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product and adding a 'type'."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, adding acidity and roast level."""
    acidity: str
    roast_level: str
