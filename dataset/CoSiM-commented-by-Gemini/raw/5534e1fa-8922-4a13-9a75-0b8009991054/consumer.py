"""
This module simulates a multi-threaded producer-consumer model for an e-commerce
marketplace.

It defines `Consumer`, `Producer`, and `Marketplace` classes. This implementation
contains several race conditions and logical flaws, particularly in its state
management and lack of fine-grained locking, making it unsuitable for a truly
concurrent environment.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer thread that processes a list of shopping commands.

    Note: This implementation creates a single cart and applies all operations
    from all its assigned "carts" to this one cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts," where each cart is a list of
                          operations (add/remove).
            marketplace (Marketplace): A reference to the central marketplace object.
            retry_wait_time (int): Time in seconds to wait before retrying an 'add'
                                   operation if the product is not available.
            **kwargs: Keyword arguments for the Thread parent class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the consumer thread."""
        cart_id = self.marketplace.new_cart()

        # Invariant: Process all operations for this consumer against a single cart.
        for product in self.carts:
            for attribute in product:
                command = attribute.get("type")
                product_item = attribute.get("product")
                quantity = attribute.get("quantity")
                if command == "remove":
                    i = 0
                    while i < quantity:
                        self.marketplace.remove_from_cart(cart_id, product_item)
                        i += 1
                elif command == "add":
                    i = 0
                    # Block Logic: Retry adding the product until successful.
                    while i < quantity:
                        no_wait = self.marketplace.add_to_cart(cart_id, product_item)
                        if no_wait:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)
        
        # Place the order and print the items.
        order = self.marketplace.place_order(cart_id)
        for prod in order:
            print(self.name, "bought", prod)


from threading import Lock
from logging.handlers import RotatingFileHandler
import logging
import time

# --- Global Logger Setup ---
LOGGER = logging.getLogger('marketplace_logger')
LOGGER.setLevel(logging.INFO)

FORMATTER = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
FORMATTER.converter = time.gmtime()

HANDLER = RotatingFileHandler('marketplace.log', maxBytes=5000, backupCount=10)
HANDLER.setFormatter(FORMATTER)

LOGGER.addHandler(HANDLER)


class Marketplace:
    """
    Coordinates producers and consumers in a simulated e-commerce environment.

    Warning: This implementation is not thread-safe. It contains race conditions
    in producer/cart registration and in inventory management. The logic for
    adding items to a cart is particularly flawed, as it does not reserve items,
    allowing multiple consumers to add the same item to their carts.
    """
    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the marketplace.
        
        Args:
            queue_size_per_producer (int): The max number of products a single
                                           producer can have published.
        """
        self.lock_consumer = Lock()
        self.lock_producer = Lock()
        self.producers = [[]] # Each inner list is a producer's inventory.
        self.carts = [[]]     # Each inner list is a consumer's cart.
        self.no_producers = 0
        self.no_carts = 0
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        Registers a new producer.

        BUG: This method is not thread-safe. `self.no_producers` is read and
        incremented without a lock, leading to a race condition if multiple
        producers register simultaneously.
        """
        LOGGER.info("A new producer is registered.")
        self.no_producers += 1
        self.producers.append([])
        LOGGER.info("Producer with id %s registerd.", self.no_producers)
        return self.no_producers

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer if they have capacity.

        BUG: Accessing `len(product_list)` is not protected by the lock, creating
        a race condition.
        """
        LOGGER.info('Producer with id %d is publishig the product: %s', producer_id, product)
        if producer_id > self.no_producers:
            LOGGER.error('Producer with id: %d does not exist', producer_id)
            raise ValueError("Producer does not exist!")

        product_list = self.producers[producer_id]
        with self.lock_producer:
            if len(product_list) >= self.queue_size_per_producer:
                can_publish = False
            else:
                product_list.append(product)
                can_publish = True
        LOGGER.info("Producer published: %s", str(can_publish))
        return can_publish

    def new_cart(self):
        """
        Creates a new, empty cart.

        BUG: This method is not thread-safe. `self.no_carts` is read and
        incremented without a lock, which is a race condition.
        """
        LOGGER.info("New cart with id %d is being created.", self.no_carts + 1)
        self.no_carts += 1
        self.carts.append([])
        return self.no_carts

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart if it is found in any producer's inventory.

        BUG: This method does not remove the item from the producer's inventory upon
        adding it to the cart. This allows multiple consumers to add the same physical
        item to their respective carts, which will cause errors during `place_order`.
        """
        LOGGER.info("Cart with id %d is adding %s.", cart_id, product)
        can_add = False
        index = -1
        with self.lock_consumer:
            for i in range(0, self.no_producers):
                for prod_in_list in self.producers[i]:
                    if prod_in_list == product:
                        index = i
                        break
            if index >= 0:
                self.carts[cart_id].append(product)
                can_add = True
        
        if can_add:
            LOGGER.info("Product was added to the cart.")
        else:
            LOGGER.info("Product could not be added to the cart.")
        return can_add

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart.

        BUG: This method does not return the product to any producer's inventory.
        The item is effectively lost from the simulation.
        """
        LOGGER.info("Cart with id %d is removing product %s.", cart_id, product)
        found = False
        with self.lock_consumer:
            if product in self.carts[cart_id]:
                found = True
            if found:
                self.carts[cart_id].remove(product)

    def place_order(self, cart_id):
        """
        Finalizes an order by attempting to remove purchased items from inventories.

        BUG: This method is not transactional and contains race conditions. Since
        `add_to_cart` does not reserve items, multiple consumers could have the
        same item in their carts. This method will fail when it tries to remove
        an item that has already been removed by another consumer's order.
        """
        LOGGER.info("Cart with id %d placed an order.", cart_id)
        if cart_id > self.no_carts:
            LOGGER.error("Cart with id %d is invalid!", cart_id)
            raise ValueError("Cart does not exist!")

        for prod in self.carts[cart_id]:
            for producer in self.producers:
                if prod in producer:
                    producer.remove(prod)
                    break

        LOGGER.info("Product list: %s.", self.carts[cart_id])
        return self.carts[cart_id].copy()

import unittest
from marketplace import Marketplace
from product import Product


class MarketplaceTestCase(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    product_test = Product("coffee", 10)
    marketplace = Marketplace(4)

    def test_place_order_exception(self):
        """Tests that placing an order for a non-existent cart raises an error."""
        marketplace = Marketplace(2)
        self.assertRaises(ValueError, marketplace.place_order, 1)

    def test_place_order(self):
        """Tests a basic order placement."""
        self.marketplace.carts = [[self.product_test]]
        self.marketplace.no_carts = 1
        response = self.marketplace.place_order(0)
        expected = [self.product_test]
        self.assertEqual(response, expected)

    def test_register_producer(self):
        """Tests producer ID generation."""
        marketplace = Marketplace(10)
        result = marketplace.register_producer()
        self.assertEqual(result, 1)

    def test_publish_exception(self):
        """Tests that publishing to a non-existent producer raises an error."""
        marketplace = Marketplace(5)
        self.assertRaises(ValueError, marketplace.publish, 2, Product("coffee", 10))

    def test_publish_method(self):
        """Tests successful product publication."""
        self.marketplace.register_producer()
        result = self.marketplace.publish(1, self.product_test)
        self.assertEqual(result, True)

    def test_publish_method_false(self):
        """Tests that publishing fails when the producer's queue is full."""
        self.marketplace.register_producer()
        for _ in range(0, 4):
            self.marketplace.publish(1, self.product_test)
        response = self.marketplace.publish(1, self.product_test)
        self.assertEqual(response, False)

    def test_new_cart(self):
        """Tests cart ID generation."""
        marketplace = Marketplace(2)
        result = marketplace.new_cart()
        self.assertEqual(result, 1)

    def test_add_cart(self):
        """Tests adding a product to a cart."""
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.product_test)
        self.marketplace.new_cart()
        result = self.marketplace.add_to_cart(1, self.product_test)
        self.assertEqual(result, True)

    def test_add_cart_false(self):
        """Tests that an unavailable product cannot be added."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        result = self.marketplace.add_to_cart(1, self.product_test)
        self.assertEqual(result, False)

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.product_test)
        self.marketplace.new_cart()
        self.assertIsNone(self.marketplace.remove_from_cart(0, self.product_test))


if __name__ == '__main__':
    unittest.main()


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer thread that continuously supplies products.

    BUG: This implementation has a severe bug where it re-registers as a new
    producer on every iteration of its main loop, leading to an infinite
    number of producers being created.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of (product, quantity, production_time) tuples.
            marketplace (Marketplace): A reference to the central marketplace.
            republish_wait_time (int): Time to wait before retrying a publish.
            **kwargs: Keyword arguments for the Thread parent class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the producer thread."""
        while True:
            # BUG: A new producer ID is acquired on every loop iteration.
            prod_id = self.marketplace.register_producer()
            for product in self.products:
                i = 0
                self.publish(i, prod_id, product)

    def publish(self, i, prod_id, product):
        """
        A recursive helper to publish a product, retrying if the queue is full.
        """
        while i < product[1]:
            no_wait = self.marketplace.publish(prod_id, product[0])
            if no_wait:
                i += 1
                time.sleep(product[2])
            else:
                time.sleep(self.republish_wait_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Represents a Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Represents a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
