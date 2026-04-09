"""
This module contains a full simulation of a producer-consumer marketplace.

@note This file appears to be a concatenation of multiple logical files,
including `consumer.py`, `marketplace.py`, `producer.py`, and `product.py`.
The documentation will treat it as a single module as presented.

The simulation includes:
- The `Marketplace` class: The central, thread-safe hub for all transactions.
- The `Producer` class: A thread that creates and publishes products.
- The `Consumer` class: A thread that simulates users buying products.
- Unit tests and dataclass definitions for the products.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing products.

    Each consumer is initialized with a set of shopping lists (carts) and
    executes the actions (add/remove) in each list before placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists for the consumer to process.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an "add"
                                     operation if a product is unavailable.
            **kwargs: Keyword arguments for the parent Thread class, including 'name'.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the consumer thread.

        For each assigned shopping list, it creates a cart, performs the add/remove
        operations, places the order, and prints the purchased items.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()


            # Invariant: Process all add/remove operations for the current cart.
            for operation in cart:
                
                if operation["type"] == "add":
                    count = 0
                    
                    # Pre-condition: Add the specified quantity of the product.
                    while count < operation["quantity"]:
                        
                        # If adding to cart is successful, increment count.
                        if self.marketplace.add_to_cart(cart_id, operation["product"]):


                            count += 1
                        else:
                            # If product is not available, wait and retry.
                            sleep(self.retry_wait_time)
                
                elif operation["type"] == "remove":
                    count = 0
                    
                    # Pre-condition: Remove the specified quantity of the product.
                    while count < operation["quantity"]:
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        
                        count += 1
             
            products_bought = self.marketplace.place_order(cart_id)
            # Post-condition: Print all products bought in the finalized order.
            for product in products_bought:
                print(self.kwargs["name"], "bought", product, flush=True)
                >>>> file: marketplace.py


from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
from distutils.log import INFO
import time
import unittest
import sys
sys.path.insert(1, './tema')
from product import Tea, Coffee

class Marketplace:
    """
    A thread-safe marketplace for producers to publish and consumers to buy products.

    This class serves as the central coordinator, managing all inventories and
    transactions. It uses a single coarse-grained lock for most operations to
    ensure thread safety, which simplifies logic but can increase contention.
    """
    
    # Class-level logger setup
    myLogger = logging.getLogger('marketplace.log')
    myLogger.setLevel(INFO)
    file_handler = RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=5)
    file_handler.setLevel(INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    formatter.converter = time.gmtime
    file_handler.setFormatter(formatter)
    myLogger.addHandler(file_handler)

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items any
                                           single producer can list at a time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        
        
        self.available_products = []
        self.cart_id = -1
        
        
        
        self.carts = []
        
        
        self.queue_size = []

        self.lock = Lock()

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        This is a thread-safe operation that assigns a new producer ID and
        initializes their inventory tracking.

        Returns:
            int: The unique ID for the new producer.
        """
        self.lock.acquire()
        self.myLogger.info("Entered method register_producer")
        
        self.producer_id += 1
        
        self.available_products.append([])
        self.available_products[self.producer_id] = []
        self.queue_size.append(0)
        
        self.myLogger.info("Exited method register_producer with producer_id=%s", self.producer_id)
        self.lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a specific producer.

        Args:
            producer_id (int): The ID of the publishing producer.
            product: The product to be published.

        Returns:
            bool: True if the product was published, False if the producer's
                  queue is full.
        """
        self.myLogger.info("Entered method publish with producer_id=%s, product=%s",
                           producer_id, product)
        id_producer = int(producer_id)
        
        # Pre-condition: Check if the producer has space in their queue.
        if self.queue_size[id_producer] < self.queue_size_per_producer:
            
            self.available_products[id_producer].append(product[0])
            self.queue_size[id_producer] += 1
            self.myLogger.info("Exited method publish with True")
            return True
        self.myLogger.info("Exited method publish with False")
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart.

        Returns:
            int: The unique ID for the newly created cart.
        """
        self.lock.acquire()
        self.myLogger.info("Entered method new_cart")
        self.cart_id += 1
        
        self.carts.append([])
        self.carts[self.cart_id] = []
        self.myLogger.info("Exited method new_cart with cart_id=%s", self.cart_id)
        self.lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product from the marketplace to a user's cart.

        This method searches through all producer inventories to find an
        available product to add.

        Args:
            cart_id (int): The ID of the target cart.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        self.lock.acquire()
        self.myLogger.info("Entered method add_to_cart with cart_id=%s, product=%s",
                           cart_id, product)
        ids = 0
        
        # Invariant: Search all producers until the product is found.
        while ids <= self.producer_id:
            
            if product in self.available_products[ids]:
                
                self.carts[cart_id].append([product, ids])
                
                self.available_products[ids].remove(product)
                self.myLogger.info("Exited method add_to_cart with True")
                self.lock.release()
                return True
            
            ids += 1
        self.myLogger.info("Exited method add_to_cart with False")
        self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the producer's inventory.

        Args:
            cart_id (int): The ID of the cart.
            product: The product to remove.
        """
        self.myLogger.info("Entered method remove_from_cart with cart_id=%s, product=%s",
                           cart_id, product)
        
        # Invariant: The product must be in the cart to be removed.
        for produs in self.carts[cart_id]:
            if produs[0] == product:
                
                self.carts[cart_id].remove([product, produs[1]])
                self.available_products[produs[1]].append(product)
                self.myLogger.info("Exited method remove_from_cart")
                return

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This method removes the items from the producer's queue size count,
        effectively consuming them and freeing up slots for the producer.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of products that were in the cart.
        """
        self.myLogger.info("Entered place_order with cart_id=%s", cart_id)
        cart_products = []
        
        for products in self.carts[cart_id]:
            cart_products.append(products[0])
            
            # This correctly models consumption by freeing up a producer's slot.
            self.lock.acquire()
            self.queue_size[products[1]] -= 1
            self.lock.release()
        self.myLogger.info("Exited place_order.")
        return cart_products

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Sets up the test fixture before each test method."""
        
        self.marketplace = Marketplace(2)
        
        self.tea = Tea(name='Linden', price=9, type='Herbal')
        self.coffee = Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()

    def test_register_producer(self):
        """Tests sequential producer ID assignment."""
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.register_producer(), 3)

    def test_publish(self):
        """Tests the product publishing logic and queue limits."""
        
        self.assertTrue(self.marketplace.publish(0, self.tea))
        self.assertListEqual(self.marketplace.available_products[0], [self.tea])
        self.assertEqual(self.marketplace.queue_size[0], 1)
        
        self.assertTrue(self.marketplace.publish(0, self.coffee))
        self.assertListEqual(self.marketplace.available_products[0], [self.tea, self.coffee])
        self.assertEqual(self.marketplace.queue_size[0], 2)
        
        # Should fail as the queue size per producer is 2.
        self.assertFalse(self.marketplace.publish(0, self.tea))

    def test_new_cart(self):
        """Tests sequential cart ID assignment."""
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.new_cart(), 3)

    def test_add_to_cart(self):
        """Tests adding products to a cart."""
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        # Should fail as the coffee has not been published.
        self.assertFalse(self.marketplace.add_to_cart(0, self.coffee))

    def test_remove_from_cart(self):
        """Tests removing a product from a cart and returning it to inventory."""
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        self.marketplace.remove_from_cart(0, self.tea)
        
        self.assertListEqual(self.marketplace.carts[0], [])
        
        self.assertListEqual(self.marketplace.available_products[0], [self.tea])

    def test_place_order(self):
        """Tests the full order cycle and its effect on producer queue sizes."""
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        self.assertEqual(self.marketplace.queue_size[0], 1)
        
        cart_list = self.marketplace.place_order(0)
        self.assertListEqual(cart_list, [self.tea])
        
        # Queue size should be decremented after the order is placed.
        self.assertEqual(self.marketplace.queue_size[0], 0)


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.

    The producer runs in an infinite loop, attempting to publish its assigned
    products according to the specified quantities and timings.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list where each element is a tuple of
                             (product, quantity_to_produce, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish
                                         if the producer's queue is full.
            **kwargs: Keyword arguments for the parent Thread class.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and enters an infinite loop to publish
        its products.
        """

        producer_id = self.marketplace.register_producer()
        # Invariant: This producer will run indefinitely, trying to publish products.
        while True:
            for product in self.products:
                count = 0
                
                # Invariant: Publish the specified quantity of the current product.
                while count < product[1]:
                    
                    if self.marketplace.publish(producer_id, product):
                        
                        sleep(product[2])
                        
                        count += 1
                    else:
                        # If the producer's queue is full, wait and retry.
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
