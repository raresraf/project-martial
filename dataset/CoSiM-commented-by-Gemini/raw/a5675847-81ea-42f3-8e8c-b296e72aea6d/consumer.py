"""
This module implements a producer-consumer simulation of a marketplace.

It defines the core components of the simulation:
- Marketplace: A class that manages producers, products, and carts. It uses
  a coarse-grained locking strategy to handle concurrency, though some methods
  may have race conditions.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places an order.
- TestMarketplace: A suite of unit tests to verify the marketplace functionality.
"""


from threading import Thread, Lock
import time
import unittest
import logging
from logging.handlers import RotatingFileHandler

# Placeholders for product classes as the original module is not available.
class Coffee:
    def __init__(self, name, price, acidity, roast): pass
class Tea:
    def __init__(self, name, price, type): pass


class Consumer(Thread):
    """
    Represents a consumer thread that shops in the marketplace.

    The consumer processes a list of desired items, persistently trying to
    add them to its cart until successful.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists for the consumer to process.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): Time to wait before retrying to add a product.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_to_cart(self, quantity, cart_id, product):
        """Helper method to add a specified quantity of a product to a cart."""
        i = 0
        # Invariant: Loop until the desired quantity has been successfully added.
        while i < quantity:
            added_ok = self.marketplace.add_to_cart(cart_id, product)
            if added_ok:
                i = i + 1
            else:
                # If the product is not available, wait and retry.
                time.sleep(self.retry_wait_time)

    def remove_from_cart(self, quantity, cart_id, product):
        """Helper method to remove a specified quantity of a product from a cart."""
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """The main execution logic for the consumer thread."""
        cart_id = self.marketplace.new_cart()
        for cart_list in self.carts:
            for cart_event in cart_list:
                if cart_event["type"] == "add":
                    self.add_to_cart(cart_event["quantity"], cart_id, cart_event["product"])
                else:
                    self.remove_from_cart(cart_event["quantity"], cart_id, cart_event["product"])
        
        # After all operations, place the order and print the items bought.
        for product in self.marketplace.place_order(cart_id):
            print(self.name, "bought", product)


class Marketplace:
    """
    Manages inventory and transactions between producers and consumers.

    This implementation uses simple lists to represent producer inventories and
    shopping carts. It employs a global dictionary `product_in_cart` to track
    the availability of each product instance, which is a potential bottleneck.
    The locking strategy is coarse-grained and may be insufficient to prevent
    all race conditions.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): Max products per producer.
        """
        self.queue_size_per_peroducer = queue_size_per_producer
        self.products = []  # A list where each index is a producer_id and the value is their product list.
        self.carts = []     # A list where each index is a cart_id and the value is the cart's contents.
        
        # A dictionary mapping each product instance to a boolean indicating if it's in a cart.
        self.product_in_cart = {}
        
        # Coarse-grained locks for different operations.
        self.lock_cart = Lock()
        self.lock_producer = Lock()
        
        # Set up logging.
        self.logger = logging.getLogger('marketplace')
        handler = RotatingFileHandler('marketplace.log', maxBytes=4096, backupCount=10)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        logging.Formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel("INFO")

    def register_producer(self):
        """
        Registers a new producer.

        Returns:
            int: The new producer's unique ID (an index in the `products` list).
        """
        self.logger.info("Method register_producer started")
        with self.lock_producer:
            self.products.append([])
            ret = len(self.products) - 1
        self.logger.info("Method register_producer returned " + str(ret))
        return ret

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Returns:
            bool: True if successful, False if the producer's inventory is full.
        """
        self.logger.info("Method publish started")
        with self.lock_producer:
            # Pre-condition: Check if the producer's queue has space.
            if len(self.products[producer_id]) < self.queue_size_per_peroducer:
                self.products[producer_id].append(product)
                self.product_in_cart[product] = False  # Mark the new product as available.
                self.logger.info("New product published to marketplace")
                return True
        self.logger.info("Method publish returned False")
        return False

    def new_cart(self):
        """Creates a new, empty cart and returns its unique ID (an index)."""
        self.logger.info("Method new_cart started")
        with self.lock_cart:
            self.carts.append([])
            ret = len(self.carts) - 1
        self.logger.info("Method new_cart returned " + str(ret))
        return ret

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart if it is available.

        Note: This method is not fully thread-safe. While the check and update
        of `product_in_cart` are atomic, the dictionary itself is not locked,
        leading to potential race conditions if other operations modify it.
        """
        self.logger.info("Method add_to_cart started")
        # Pre-condition: Check if the product exists and is not already in a cart.
        if product in self.product_in_cart.keys() and not self.product_in_cart[product]:
            self.carts[cart_id].append(product)
            self.product_in_cart[product] = True # Mark as "in a cart".
            self.logger.info("New product added to cart " + str(cart_id))
            return True
        self.logger.info("Method add_to_cart returned False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart, making it available again.

        Note: This method is not thread-safe and can lead to race conditions.
        """
        self.logger.info("Method remove_from_cart started")
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            self.product_in_cart[product] = False # Mark as available again.
            self.logger.info("Product removed from cart")

    def place_order(self, cart_id):
        """
        Finalizes an order by removing products from producer inventories.

        Note: This method is not thread-safe. Iterating and modifying the
        producer lists without a lock can lead to race conditions.
        """
        self.logger.info("Method place_order started")
        for cart_product in self.carts[cart_id]:
            for prod_products in self.products:
                if cart_product in prod_products:
                    prod_products.remove(cart_product)
        self.logger.info("Method place_order returned " + str(self.carts[cart_id]))
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    
    def setUp(self):
        """Initializes a marketplace and products for each test."""
        self.marketplace = Marketplace(15)
        self.products = [Coffee("Espresso", 7, 4.00, "MEDIUM"),
                         Coffee("Irish", 10, 5.00, "MEDIUM"),
                         Tea("Black", 10, "Green")]

    def test_register_producer(self):
        """Tests that producer IDs are generated sequentially."""
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        """Tests that publishing a product works correctly."""
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(0, self.products[0]))
        self.assertTrue(self.marketplace.publish(0, self.products[1]))

    def test_new_cart(self):
        """Tests that cart IDs are generated sequentially."""
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)

    def test_add_to_cart(self):
        """Tests that a product can be successfully added to a cart."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[0]))
        self.assertEqual(len(self.marketplace.carts[0]), 1)

    def test_remove_from_cart(self):
        """Tests that removing a product from a cart makes it available again."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.remove_from_cart(0, self.products[0])
        self.assertEqual(len(self.marketplace.carts[0]), 0)
        self.assertFalse(self.marketplace.product_in_cart[self.products[0]])

    def test_place_order(self):
        """Tests that placing an order returns the correct items."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[0])
        self.assertEqual(self.marketplace.place_order(0), [self.products[0]])


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.
        Args:
            products (list): A list of (product, quantity, delay) tuples.
            marketplace (Marketplace): The marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the producer thread."""
        producer_id = self.marketplace.register_producer()
        # Invariant: The producer runs in an infinite loop.
        while True:
            for product in self.products:
                i = 0
                num_of_products = product[1]
                curr_product = product[0]
                curr_product_wait_time = product[2]
                # Invariant: Publish the specified quantity of the current product.
                while i < num_of_products:
                    published_ok = self.marketplace.publish(producer_id, curr_product)
                    if published_ok:
                        i += 1
                        time.sleep(curr_product_wait_time)
                    else:
                        # If publishing fails (queue is full), wait and retry.
                        time.sleep(self.republish_wait_time)
