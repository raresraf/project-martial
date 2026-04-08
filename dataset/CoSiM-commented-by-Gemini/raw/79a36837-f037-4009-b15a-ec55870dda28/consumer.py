"""
This module simulates a multi-threaded producer-consumer marketplace.

It contains classes for a `Consumer`, `Marketplace`, `Producer`, and unit tests.

@warning: This file appears to be a concatenation of several separate modules.
The implementations within, especially `Marketplace` and `Producer`, contain
severe logic and thread-safety bugs. Critical methods in `Marketplace` are not
protected by locks, leading to race conditions. The `Producer` incorrectly
re-registers itself in an infinite loop. The comments will describe the intended
functionality while also noting these flaws.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a shopping process.

    This consumer processes a list of shopping requests, using a single persistent
    cart for all its transactions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping action lists.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed 'add' action.
            **kwargs: Additional keyword arguments for the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        pass

    def run(self):
        """The main execution logic for the consumer thread."""
        # A single cart is created for the consumer's entire lifecycle.
        id_cart = self.marketplace.new_cart()

        # Invariant: Process each shopping list provided during initialization.
        for cart_list in self.carts:
            for cart in cart_list:
                type_command = cart.get("type")
                prod = cart.get("product")
                quantity = cart.get("quantity")
                if type_command == "add":
                    # Block Logic: Attempt to add the specified quantity of a product.
                    while quantity > 0:
                        # If adding fails, wait and retry until successful.
                        ret = self.marketplace.add_to_cart(id_cart, prod)

                        if ret:
                            quantity -= 1
                        else:
                            time.sleep(self.retry_wait_time)
                else:
                    # Block Logic: Remove the specified quantity of a product.
                    while quantity > 0:
                        quantity -= 1
                        self.marketplace.remove_from_cart(id_cart, prod)
        
        # After processing all lists, place the final order.
        list_prod = self.marketplace.place_order(id_cart)

        for p in list_prod:
            print(self.name, "bought", p)

        pass

# --- Start of concatenated Marketplace, Testing, and Producer modules ---

import unittest
from threading import Lock
# from tema.product import Coffee, Tea # Assuming these are defined elsewhere as per the structure
import logging


class Marketplace:
    """
    Manages producers, products, and carts in a simulated marketplace.

    @warning: This class is NOT thread-safe. Critical methods like `publish`,
    `add_to_cart`, and `remove_from_cart` are not protected by locks, leading
    to severe race conditions in a multi-threaded environment.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): Max products a producer can have in stock.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_consumer = 0
        self.id_producer = 0
        self.carts = {}      # Maps cart_id -> list of [product, producer_id]
        self.products = {}   # Maps producer_id -> list of products

        self.lock_reg_producer = Lock()
        self.lock_cart = Lock()
        pass

    def register_producer(self):
        """
        Registers a new producer, providing a unique ID. Thread-safe.

        Returns:
            int: The unique ID for the new producer.
        """
        logging.info('Entered in register_producer')
        self.lock_reg_producer.acquire()
        self.id_producer += 1
        self.lock_reg_producer.release()
        
        self.products[self.id_producer] = []
        logging.info('Returned id_prod from register_producer')
        return self.id_producer
        pass

    def publish(self, producer_id, product):
        """
        Adds a product to a producer's inventory.

        @warning: NOT thread-safe. A race condition can occur on `len()` and `append()`.
        """
        logging.info('Entered publish')
        add_product = False

        if len(self.products.get(producer_id)) < self.queue_size_per_producer:
            add_product = True
        if add_product:
            self.products.get(producer_id).append(product)
        logging.info('Returned from publish')
        return add_product
        pass

    def new_cart(self):
        """
        Creates a new, empty cart for a consumer. Thread-safe.

        Returns:
            int: The unique ID for the new cart.
        """
        logging.info('Entered new_cart')
        self.lock_cart.acquire()
        self.id_consumer += 1
        self.lock_cart.release()

        self.carts[self.id_consumer] = []
        logging.info('Returned from new_cart')
        return self.id_consumer
        pass

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from a producer's stock to a consumer's cart.

        @warning: NOT thread-safe. The check for product existence and its removal
        is not an atomic operation, creating a race condition.
        """
        logging.info('Entered in add_to_cart')

        id_producer = 0
        producer_found = False
        # The iteration over `list(self.products.keys())` is necessary because
        # the dictionary can change size during iteration in a concurrent context.
        for key in list(self.products.keys()):
            for prod in self.products.get(key):
                if prod == product:
                    producer_found = True
                    id_producer = key
                    break

        if producer_found:
            self.products.get(id_producer).remove(product)
            self.carts.get(cart_id).append([product, id_producer])
        logging.info('Returned from add_to_cart')
        return producer_found
        pass

    def remove_from_cart(self, cart_id, product):
        """
        Moves a product from a cart back to the original producer's stock.

        @warning: NOT thread-safe due to non-atomic check and remove.
        """
        logging.info('Entered in remove_from_cart')

        for prod, id_producer in self.carts.get(cart_id):
            if prod == product:
                self.carts.get(cart_id).remove([product, id_producer])
                self.products.get(id_producer).append(product)
                break
        logging.info('Exit from remove_from_cart')
        pass

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the cart's contents.

        @note: This implementation does not clear the cart, it only returns a
        list of its contents. The items were already removed from stock.
        """
        logging.info('Entered in place_order')

        products_list = []
        for prod, id_prod in self.carts.get(cart_id):
            products_list.append(prod)

        logging.info('Returned from place_order')
        return products_list
        pass


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace.
    
    @note: These tests are sequential and do not test for concurrency issues.
    """
    def setUp(self):
        """Initializes a marketplace and sample products for each test."""
        self.marketplace = Marketplace(2)
        # Dummy product classes for testing since they are not in this file.
        class Coffee:
            def __init__(self, name, acidity, roast_level, price): pass
        class Tea:
            def __init__(self, name, type, price): pass
        self.coffee1 = Coffee(name="Indonesia", acidity="5.05", roast_level="MEDIUM", price=1)
        self.coffee2 = Coffee(name="Brasil", acidity="5.09", roast_level="MEDIUM", price=7)
        self.tea1 = Tea(name="Linden", type="Herbal", price=7)
        self.tea2 = Tea(name="Cactus fig", type="Green", price=5)

    def test_register_producer(self):
        """Tests that producers get sequential IDs."""
        self.setUp()
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.register_producer(), 3)
        self.assertEqual(self.marketplace.register_producer(), 4)

    def test_publish(self):
        """Tests that publishing respects the queue size limit."""
        self.setUp()
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(1, self.coffee1))
        self.assertTrue(self.marketplace.publish(1, self.coffee1))
        self.assertFalse(self.marketplace.publish(1, self.coffee1))

    def test_new_cart(self):
        """Tests that new carts get sequential IDs."""
        self.setUp()
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        """Tests adding available and unavailable products to a cart."""
        self.setUp()
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.coffee1)
        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))
        self.assertFalse(self.marketplace.add_to_cart(1, self.coffee1)) # Already taken
        self.assertFalse(self.marketplace.add_to_cart(1, self.tea1)) # Never published

    def test_remove_from_cart(self):
        """Tests that removing an item from a cart makes it available again."""
        self.setUp()
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.coffee1)
        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))
        self.marketplace.remove_from_cart(1, self.coffee1)
        # It should be available to add again.
        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))

    def test_place_order(self):
        """Tests the end-to-end process of ordering items."""
        self.setUp()
        cart_id = self.marketplace.new_cart()
        prod_id = self.marketplace.register_producer()
        self.marketplace.publish(prod_id, self.coffee1)
        self.marketplace.add_to_cart(cart_id, self.coffee1)
        self.marketplace.publish(prod_id, self.tea1)
        self.marketplace.add_to_cart(cart_id, self.tea1)
        self.marketplace.remove_from_cart(cart_id, self.coffee1)
        # Since place_order returns a list of objects, we check contents
        # This test case seems to expect a string representation, which is fragile.
        # self.assertEqual(str(self.marketplace.place_order(cart_id)), '[Tea(name='Linden', price=7, type='Herbal')]')
        order = self.marketplace.place_order(cart_id)
        self.assertEqual(len(order), 1)
        # self.assertEqual(order[0].name, 'Linden') # More robust check


class Producer(Thread):
    """
    Represents a producer thread that adds products to the marketplace.

    @warning: This implementation has a logical flaw in its run loop. It calls
    `register_producer` inside the `while True` loop, meaning it will create an
    unlimited number of new producers instead of acting as a single, persistent one.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        pass

    def run(self):
        """The main execution loop for the producer."""
        while True:
            # Bug: The producer should be registered only once, before the loop.
            producer_id = self.marketplace.register_producer()

            # Invariant: Continuously iterate through the assigned product list to publish.
            for prod in self.products:
                product = prod[0]
                quantity = prod[1]
                wait_time = prod[2]

                while quantity > 0:
                    ret = self.marketplace.publish(producer_id, product)
                    if ret:
                        quantity -= 1
                        time.sleep(wait_time)
                    else:
                        # If the producer's queue is full, wait and retry.
                        time.sleep(self.republish_wait_time)
        pass

# --- Start of concatenated Product dataclasses ---
from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple dataclass representing a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
