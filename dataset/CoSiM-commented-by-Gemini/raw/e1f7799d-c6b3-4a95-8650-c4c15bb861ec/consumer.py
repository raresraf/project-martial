"""
consumer.py

@brief A complex, multithreaded simulation of a producer-consumer marketplace with logging and unit tests.
@description This module simulates an e-commerce marketplace where multiple Producer threads can
publish products and multiple Consumer threads can purchase them. The Marketplace class acts as the
central broker, managing inventory and carts. The implementation includes file-based logging for
all major operations and a suite of unit tests to verify the marketplace's logic.

NOTE: This implementation is not thread-safe. The Marketplace class, which manages shared state,
lacks the necessary synchronization (locks) to prevent race conditions when accessed by multiple
consumer and producer threads concurrently.
"""

from threading import Thread, Lock
import time
from logging.handlers import RotatingFileHandler
import unittest
import logging
# The relative import suggests this file may be part of a larger package.
from .product import Product
from dataclasses import dataclass


# --- Logging Configuration ---
# Sets up a rotating file logger to record marketplace events.
logging.basicConfig(
    handlers=[RotatingFileHandler('marketplace.log', maxBytes=500000, backupCount=20)],
    format="%(asctime)s %(levelname)s %(funcName)s %(message)s",
    level=logging.INFO)
logging.Formatter.converter = time.gmtime


class Consumer(Thread):
    """
    Represents a consumer thread that shops in the marketplace.

    A consumer is initialized with a set of "carts", which are lists of commands
    (add/remove products) that simulate a shopping session.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping sessions. Each session is a list of commands.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying to add an unavailable product.
            **kwargs: Accepts a 'name' for the consumer thread.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        The main execution logic for the consumer thread.

        It creates a single cart and uses it to process all its assigned shopping
        sessions one by one.
        """
        # Functional Utility: A single cart is created for the lifetime of the consumer.
        id_cart = self.marketplace.new_cart()

        # Block Logic: Process each list of commands as a separate shopping trip.
        for commands in self.carts:
            for command in commands:
                type_command = command["type"]
                product = command["product"]
                prod_quantity = command["quantity"]

                # Block Logic: Handle 'add' commands by repeatedly trying to add the product.
                if type_command == "add":
                    for _ in range(prod_quantity):
                        # Invariant: This loop will block until the product is successfully added.
                        while not self.marketplace.add_to_cart(id_cart, product):
                            time.sleep(self.retry_wait_time)
                
                # Block Logic: Handle 'remove' commands.
                if type_command == "remove":
                    for _ in range(prod_quantity):
                        self.marketplace.remove_from_cart(id_cart, product)

            # After processing a shopping trip's commands, place the order.
            for product in self.marketplace.place_order(id_cart):
                print(f'{self.name} bought {product}', flush=True)


class Marketplace:
    """
    The central marketplace, managing all producers, consumers, and products.

    This class acts as the shared state for the simulation. It uses a dictionary-based
    system to track producer inventories and consumer carts. It attempts to manage
    inventory by marking products as 'available' ('a') or 'unavailable'/'in-cart' ('u').
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): Max items a single producer can list.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.nr_producers = 0
        self.nr_carts = 0
        self.producers_dict = {}  # Stores producer inventories.
        self.consumers_dict = {}  # Stores consumer carts.

    def register_producer(self):
        """Allocates a unique ID for a new producer."""
        self.producers_dict[self.nr_producers] = []
        self.nr_producers += 1
        return self.nr_producers - 1

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product. The product is stored in a dictionary
        with a state ('a' for available).
        """
        logging.info('%d %s', producer_id, product)
        products_list = self.producers_dict[producer_id]
        if len(products_list) < self.queue_size_per_producer:
            # The product is wrapped in a dictionary to hold its state.
            products_list.append({product: 'a'})
            logging.info('True')
            return True
        logging.info('False')
        return False

    def new_cart(self):
        """Creates a new, empty cart for a consumer."""
        self.consumers_dict[self.nr_carts] = []
        self.nr_carts += 1
        return self.nr_carts - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by finding an 'available' one from any producer
        and marking it as 'unavailable'.
        """
        logging.info('%d %s', cart_id, product)
        cart_list = self.consumers_dict[cart_id]

        # Block Logic: Search all producers for an available instance of the product.
        for key in self.producers_dict:
            products_map = self.producers_dict[key]
            for dict_item in products_map:
                if product in dict_item:
                    # Check if the item is marked as available ('a').
                    if dict_item[product] == 'a':
                        # Add item to cart, storing which producer it came from.
                        cart_list.append({product: key})
                        # Mark the item as unavailable ('u') in the producer's inventory.
                        dict_item[product] = 'u'
                        logging.info('True')
                        return True
        logging.info('False')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and marks it as 'available' again in the
        producer's inventory.
        """
        logging.info('%d %s', cart_id, product)
        cart_list = self.consumers_dict[cart_id]
        for prod_dict in cart_list:
            if product in prod_dict:
                # Find the original producer using the ID stored in the cart.
                producer_id = prod_dict[product]
                product_list = self.producers_dict[producer_id]
                for prodd in product_list:
                    if product in prodd:
                        # Mark the product as available again.
                        prodd[product] = 'a'
                cart_list.remove(prod_dict)
                return

    def place_order(self, cart_id):
        """
        Finalizes an order by permanently removing items from producers' inventories.
        """
        logging.info('%d', cart_id)
        products_ordered_list = []
        # Block Logic: Iterate through items in the cart to finalize the purchase.
        for item in self.consumers_dict[cart_id]:
            for product_key in item:
                products_ordered_list.append(product_key)
                producer_id = item[product_key]
                # Find and remove the product from the original producer's list.
                for prod_in_stock in self.producers_dict[producer_id]:
                    if product_key in prod_in_stock:
                        self.producers_dict[producer_id].remove(prod_in_stock)
                        break
        
        self.consumers_dict[cart_id].clear()
        logging.info(products_ordered_list)
        return products_ordered_list


class TestMarketPlace(unittest.TestCase):
    """
    Unit tests for the Marketplace class to verify its core logic in isolation.
    """
    
    def setUp(self) -> None:
        """Set up a fresh marketplace and product list for each test."""
        self.market_place_object = Marketplace(10)
        self.products_list = [Product('product' + str(i), i * 10) for i in range(10)]
        self.cart = {}
        self.producer = None

    def test_register_producer(self):
        """Test that producer registration returns sequential IDs."""
        for i in range(10):
            self.assertEqual(self.market_place_object.register_producer(), i)

    def test_publish(self):
        """Test that products can be published up to the queue limit."""
        self.producer = self.market_place_object.register_producer()
        for product in self.products_list:
            self.assertTrue(self.market_place_object.publish(self.producer, product))
        
        # Test that publishing fails when the producer's queue is full.
        new_product = Product('product10', 0)
        self.assertFalse(self.market_place_object.publish(self.producer, new_product))

    def test_new_cart(self):
        """Test that new_cart returns sequential IDs."""
        for i in range(10):
            self.assertEqual(self.market_place_object.new_cart(), i)

    def test_add_to_cart(self):
        """Test that available products can be added to a cart."""
        self.producer = self.market_place_object.register_producer()
        for product in self.products_list:
            self.market_place_object.publish(self.producer, product)
        self.cart = self.market_place_object.new_cart()
        for product in self.products_list:
            self.assertTrue(self.market_place_object.add_to_cart(self.cart, product))

    def test_remove_from_cart(self):
        """Test that removing a product makes it available again."""
        product_to_be_removed = self.products_list[0]
        self.producer = self.market_place_object.register_producer()
        self.market_place_object.publish(self.producer, product_to_be_removed)
        self.cart = self.market_place_object.new_cart()
        self.market_place_object.add_to_cart(self.cart, product_to_be_removed)

        producer_list = self.market_place_object.producers_dict[self.producer]
        
        # Verify product is marked as 'unavailable' after being added to cart.
        self.assertTrue({product_to_be_removed: 'u'} in producer_list)
        self.market_place_object.remove_from_cart(self.cart, product_to_be_removed)

        # Verify product is marked as 'available' again after removal.
        self.assertTrue({product_to_be_removed: 'a'} in producer_list)

    def test_place_order(self):
        """Test that placing an order correctly returns the list of bought products."""
        self.producer = self.market_place_object.register_producer()
        for product in self.products_list:
            self.market_place_object.publish(self.producer, product)
        self.cart = self.market_place_object.new_cart()
        for product in self.products_list:
            self.market_place_object.add_to_cart(self.cart, product)

        products_ordered = self.market_place_object.place_order(self.cart)
        self.assertEqual(self.products_list, products_ordered)


class Producer(Thread):
    """
    Represents a producer thread that publishes a list of products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of (product, quantity, wait_time) tuples.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Accepts 'name' and 'daemon' flag.
        """
        Thread.__init__(self, daemon=kwargs['daemon'])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and then loops forever, trying to publish
        all its assigned products.
        """
        id_prod = self.marketplace.register_producer()
        # Invariant: The producer runs in an infinite loop to continuously supply products.
        while True:
            for (prod, prod_quantity, waiting_time) in self.products:
                time.sleep(waiting_time)
                for _ in range(prod_quantity):
                    # Invariant: This loop blocks until the product is successfully published.
                    while not self.marketplace.publish(id_prod, prod):
                        time.sleep(self.republish_wait_time)


# --- Data classes for products ---
# These appear to be duplicates or an alternative definition site.

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A product of type Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A product of type Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
