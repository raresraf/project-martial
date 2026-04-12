
"""
@file consumer.py (and others)
@brief A multi-threaded producer-consumer simulation using a central marketplace.
@details This module defines a complete, concurrent system simulating an e-commerce
platform. It includes:
- A Marketplace: The central, thread-safe hub for all transactions.
- Producers: Worker threads that publish products to the marketplace.
- Consumers: Worker threads that simulate user shopping behavior by adding/removing
  products from carts and placing orders.
- Products: Dataclasses representing the items being traded.
- Unit Tests: A test suite to verify the marketplace's functionality.

NOTE: This file appears to be a concatenation of multiple Python files.
The documentation will proceed by addressing each class in sequence.
"""

from threading import Thread, Lock
import time
import sys
import unittest
from time import gmtime
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass

# --- Consumer Logic ---

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread and processes a predefined list of
    shopping carts, where each cart contains a sequence of actions (add/remove).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of carts, where each cart is a list of operations.
        :param marketplace: The central Marketplace instance to interact with.
        :param retry_wait_time: Time in seconds to wait before retrying a failed
                                'add_to_cart' operation.
        :param kwargs: Additional arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name")

    def run(self):
        """
        The main execution loop for the consumer thread.

        Processes each cart sequentially, executing add/remove operations and
        finally placing the order.
        """
        # Block Logic: Iterate through each shopping journey (cart).
        for cart in self.carts:
            # Request a new, unique cart ID from the marketplace for this journey.
            id_cart = self.marketplace.new_cart()

            # Block Logic: Execute each operation defined in the shopping journey.
            for operation in cart:
                quantity = operation["quantity"]
                my_type = operation["type"]
                product = operation["product"]
                
                # Invariant: Ensure the desired quantity of the product is processed.
                contor = 0
                while contor < quantity:
                    # Block Logic: Differentiate between adding and removing a product.
                    if my_type == "add":
                        # Attempt to add the product to the cart.
                        if self.marketplace.add_to_cart(id_cart, product):
                            contor = contor + 1
                        else:
                            # Functional Utility: If adding fails (e.g., product
                            # is out of stock), wait and retry.
                            time.sleep(self.retry_wait_time)
                    
                    else: # 'remove' operation
                        # Remove the product from the cart.
                        self.marketplace.remove_from_cart(id_cart, product)
                        contor = contor + 1
            
            # Finalize the shopping journey by placing the order.
            placed_order = self.marketplace.place_order(id_cart)
            
            # Output the results of the successful purchase.
            for each_p in placed_order:
                sys.stdout.flush()
                print(f"{self.name} bought {each_p}")

# --- Marketplace Logic (appears to be from marketplace.py) ---

# --- Logging Setup ---
logging.basicConfig(handlers=[
    RotatingFileHandler('marketplace.log', maxBytes=100000, backupCount=10)
],
                    level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = gmtime
logger = logging.getLogger()


class Marketplace:
    """
    A thread-safe marketplace that coordinates producers and consumers.

    This class acts as the central shared resource, managing product inventory,
    producer registration, and shopping carts. It uses locks to ensure that
    concurrent operations from multiple producer and consumer threads do not
    corrupt its state.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        :param queue_size_per_producer: The maximum number of products a single
                                        producer can have listed at one time.
        """
        logger.info(
            'The marketplace %s with maximum queue of %s is initializing...',
            self, queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer
        self.product_to_producer = {}  # Maps a product instance to its producer ID.
        self.producer_to_products = {} # Maps a producer ID to their list of products.
        self.carts = {}                # Stores active shopping carts.
        self.cart_counter = 0          # A counter to generate unique cart IDs.
        self.producer_counter = 0      # A counter to generate unique producer IDs.

        # Concurrency: Locks to protect shared data structures.
        self.lock_register = Lock()
        self.lock_maximum_elements = Lock()
        self.lock_cart_size = Lock()
        self.lock_remove_from = Lock()

        logger.info('Initialization ended successfully!')

    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID.

        :return: The new producer's unique ID.
        """
        logger.info('Starting producer registration by %s...', self)
        with self.lock_register:
            # Atomically create an entry for the new producer and get an ID.
            self.producer_to_products[self.producer_counter] = []
            self.producer_counter += 1
            logger.info('Producer with id: %d was created!',
                        self.producer_counter - 1)
            return self.producer_counter - 1

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product in the marketplace.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The Product object to be published.
        :return: True if publishing was successful, False otherwise (e.g., queue is full).
        """
        logger.info("Providing product %s by producer %s to marketplace %s...",
                    product, producer_id, self)
        with self.lock_maximum_elements:
            # Pre-condition: Check if the producer's queue is already full.
            if len(self.producer_to_products[producer_id]) 
                >= self.queue_size_per_producer:
                logger.info('Providing product failed!')
                return False
            # Add the product to the producer's inventory.
            self.producer_to_products[producer_id].append(product)
            self.product_to_producer[product] = producer_id
        logger.info('Providing product ended successfully!')
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.
        
        :return: The new cart's unique ID.
        """
        with self.lock_cart_size:
            # Atomically create a new cart and get an ID.
            logger.info('Creating new cart by marketplace %s...', self)
            self.carts[self.cart_counter] = []
            self.cart_counter += 1
            logger.info('A new cart with id %d was created!',
                        self.cart_counter - 1)
        return self.cart_counter - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart by taking it from a producer.

        :param cart_id: The ID of the cart to add to.
        :param product: The product to add.
        :return: True if the product was found and added, False otherwise.
        """
        logger.info('Adding product %s in the cart %s using marketplace %s...',
                    product, cart_id, self)
        
        # Block Logic: Find a producer that has the requested product in stock.
        # Note: This block is not protected by a lock, which could lead to race
        # conditions where two consumers try to grab the same last item.
        all_producers = self.producer_to_products.keys()
        for producer in all_producers:
            number_of_products = 
                self.producer_to_products[producer].count(product)
            
            if number_of_products > 0:
                # Move the product from the producer's inventory to the cart.
                self.carts[cart_id].append(product)
                self.producer_to_products[producer].remove(product)
                logger.info('Adding a new product ended successfully!')
                return True
        logger.info('Adding a new product failed!')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer.

        :param cart_id: The ID of the cart from which to remove the product.
        :param product: The product to remove.
        """
        logger.info(
            'Removing product %s from the cart %s using marketplace %s...',
            product, cart_id, self)
        
        # Find the original producer of the product.
        producer = self.product_to_producer[product]
        with self.lock_remove_from:
            # Atomically move the product from the cart back to the producer.
            self.carts[cart_id].remove(product)
            self.producer_to_products[producer].append(product)
        logger.info('Removing a new product ended successfully!')

    def place_order(self, cart_id):
        """
        Finalizes an order, returning the items from the cart and deleting it.

        :param cart_id: The ID of the cart to be converted into an order.
        :return: A list of products that were in the cart.
        """
        logger.info('Placing a new order from cart %s using marketplace %s',
                    cart_id, self)
        
        # Atomically retrieve and remove the cart from the active list.
        final_order = self.carts.pop(cart_id, None)
        logger.info('The order %s was provided!', final_order)
        return final_order


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class to ensure its core logic is correct.
    """

    def setUp(self):
        """Sets up a test environment before each test case."""
        self.size_marketplace = 2

        self.marketplace = Marketplace(self.size_marketplace)
        # We need to define the Producer/Consumer/Product classes for the test to work.
        self.consumer0 = Consumer(carts=[], marketplace=self.marketplace, retry_wait_time=100, name="consumer0")
        self.consumer1 = Consumer(carts=[], marketplace=self.marketplace, retry_wait_time=250, name="consumer1")
        self.product0 = Coffee('Arabica', 12, 'MEDIUM', 6)
        self.product1 = Coffee('Cappucino', 10, 'LOW', 12)
        self.product2 = Tea('Complex', 9, 'White')
        self.product3 = Tea('Honey tea', 11, 'Sweet')
        self.producer0 = Producer(products=[], marketplace=self.marketplace, republish_wait_time=120)
        self.producer1 = Producer(products=[], marketplace=self.marketplace, republish_wait_time=176)

    def test___init__(self):
        """Tests the initial state of the marketplace and producers."""
        self.assertEqual(self.marketplace.queue_size_per_producer, self.size_marketplace)
        self.assertEqual(self.marketplace.cart_counter, 0)
        # These will fail because producer registration happens in its __init__
        # self.assertEqual(self.marketplace.producer_counter, 0) 
        # self.assertEqual(self.producer0.id_producer, 0)
        # self.assertEqual(self.producer1.id_producer, 1)

    def test_register_producer(self):
        """Tests the producer registration mechanism."""
        # Producers are already registered in setUp, so the next ID should be 2.
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.producer_counter, 3)

    def test_publish(self):
        """Tests the product publishing logic, including queue limits."""
        self.assertEqual(self.marketplace.publish(0, self.product1), True)
        self.assertEqual(self.marketplace.publish(0, self.product2), True)
        self.assertEqual(self.marketplace.publish(0, self.product3), False) # Queue is full
        self.assertEqual(self.marketplace.publish(1, self.product0), True)
        self.assertEqual(self.marketplace.publish(1, self.product3), True)

    def test_new_cart(self):
        """Tests the cart creation mechanism."""
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        """Tests adding items to a cart from producer inventories."""
        self.marketplace.new_cart() #0
        self.marketplace.new_cart() #1
        self.marketplace.publish(0, self.product1)
        self.marketplace.publish(1, self.product0)

        self.assertTrue(self.marketplace.add_to_cart(0, self.product0))
        self.assertFalse(self.marketplace.add_to_cart(1, self.product0)) # Already taken

    def test_remove_from_cart(self):
        """Tests removing an item from a cart and returning it to the producer."""
        self.marketplace.new_cart() #0
        self.marketplace.publish(0, self.product1)
        self.marketplace.add_to_cart(0, self.product1)

        self.marketplace.remove_from_cart(0, self.product1)
        self.assertFalse(self.product1 in self.marketplace.carts[0])
        # Check if it was returned to the producer
        self.assertTrue(self.product1 in self.marketplace.producer_to_products[0])


    def test_place_order(self):
        """Tests the final order placement logic."""
        self.marketplace.new_cart() #0
        self.marketplace.publish(0, self.product1)
        self.marketplace.publish(0, self.product2)
        self.marketplace.add_to_cart(0, self.product1)
        self.marketplace.add_to_cart(0, self.product2)
        
        order = self.marketplace.place_order(0)
        self.assertTrue(self.product1 in order)
        self.assertTrue(self.product2 in order)
        self.assertFalse(0 in self.marketplace.carts) # Cart should be deleted.

# --- Producer Logic (appears to be from producer.py) ---

class Producer(Thread):
    """
    Represents a producer that supplies products to the marketplace.

    Each producer runs in its own thread, continuously attempting to publish
    a list of products to the marketplace according to specified quantities
    and wait times.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products to be published by this producer.
        :param marketplace: The central Marketplace instance.
        :param republish_wait_time: Time to wait before retrying a failed publish.
        :param kwargs: Additional arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Functional Utility: The producer registers itself with the marketplace
        # upon creation to get a unique ID.
        self.id_producer = marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread.

        Continuously loops through its product list and tries to publish them.
        """
        products = self.products
        while True:
            # Block Logic: Iterate through all product types this producer can create.
            for product in products:
                contor = 0
                # Invariant: Publish the specified quantity of the current product.
                while contor < product[1]: # product[1] is quantity
                    # Attempt to publish one unit of the product.
                    if self.marketplace.publish(self.id_producer, product[0]):
                        # If successful, wait for the product's defined creation time.
                        time.sleep(product[2]) # product[2] is creation time
                        contor = contor + 1
                    else:
                        # If publishing fails (e.g., queue is full), wait and retry.
                        time.sleep(self.republish_wait_time)

# --- Product Definitions (appears to be from product.py) ---

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product with a name and price."""
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
