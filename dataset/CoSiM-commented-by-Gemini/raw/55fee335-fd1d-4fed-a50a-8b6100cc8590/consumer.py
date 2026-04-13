
"""
This module implements a producer-consumer simulation using a shared marketplace.
It includes classes for the Marketplace itself, which acts as a broker,
as well as Producer and Consumer threads that interact with the marketplace.
The simulation handles product registration, publishing, and purchasing via carts.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.
    Each consumer simulates a user shopping by creating a cart, adding and
    removing products, and eventually placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts: A list of cart configurations, where each configuration
                   specifies products and quantities to add or remove.
            marketplace: The shared Marketplace object.
            retry_wait_time: The time to wait before retrying to add a product
                             to the cart if it's not immediately available.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.
        It simulates the process of shopping: getting a new cart, processing
        a list of actions (add/remove products), and placing an order.
        """
        cart_id = self.marketplace.new_cart()

        for i in self.carts:
            for j in i:
                quantity = j['quantity']
                product = j['product']
                action = j['type']

                # Perform the add or remove action for the specified quantity
                for _ in range(0, quantity):
                    if action == 'add':
                        added = self.marketplace.add_to_cart(cart_id, product)

                        # Retry adding to the cart if the product is not available
                        while not added:
                            time.sleep(self.retry_wait_time)
                            added = self.marketplace.add_to_cart(cart_id, product)

                    elif action == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)

            # After processing all items in a cart configuration, place the order.
            self.marketplace.place_order(cart_id)

import logging
import time
from logging.handlers import RotatingFileHandler


class Logger:
    """
    A simple factory for creating and configuring a rotating file logger.
    This provides a standardized way to set up logging across the application.
    """
    MAX_BYTE_COUNT = 1000000
    BACKUP_COUNT = 5

    @staticmethod
    def create_logger(name, log_file):
        """
        Creates a configured logger instance.

        Args:
            name: The name of the logger.
            log_file: The path to the log file.

        Returns:
            A configured logging.Logger instance.
        """
        # Set up the log message format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - '
                                      '%(funcName)s - %(message)s')
        formatter.converter = time.gmtime

        # Configure a rotating file handler to manage log file size
        handler = RotatingFileHandler(log_file,
                                      maxBytes=Logger.MAX_BYTE_COUNT,
                                      backupCount=Logger.BACKUP_COUNT)
        handler.setFormatter(formatter)

        # Create and configure the logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        return logger

import threading
from threading import Lock
import unittest
from tema.product import Tea
from tema.logger import Logger


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    These tests verify the core functionality of the marketplace, including
    producer registration, product publishing, and cart operations.
    """

    def setUp(self) -> None:
        """Set up a new Marketplace instance before each test."""
        self.marketplace = Marketplace(1)

    def test_register_producer(self):
        """Test that a producer can be successfully registered."""
        self.assertGreaterEqual(int(self.marketplace.register_producer()), 0)

    def test_publish(self):
        """
        Test the product publishing logic.
        This includes tests for unregistered producers, null products,
        full queues, and successful publishing.
        """
        # Test publishing with an unregistered producer ID
        self.assertEqual(self.marketplace.publish(123, None), False)

        # Test publishing a None product
        producer_id = self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(producer_id, None), False)

        # Test successful product publishing
        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.publish(producer_id, product), True)

        # Test publishing to a full queue
        self.assertEqual(self.marketplace.publish(producer_id, product), False)

        # Test that a product can be republished after it's been sold
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.publish(producer_id, product), True)

    def test_new_cart(self):
        """Test that a new cart can be successfully created."""
        self.assertGreaterEqual(int(self.marketplace.new_cart()), 0)

    def test_add_to_cart(self):
        """
        Test the logic for adding products to a cart.
        This includes handling of invalid cart IDs, null products, unavailable
        products, and race conditions between multiple carts.
        """
        # Test adding to an invalid cart
        self.assertEqual(self.marketplace.publish(12345, None), False)

        # Test adding a null product to a valid cart
        cart_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(cart_id, None), False)

        # Test adding a product that is not yet published
        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), False)

        # Test successfully adding a published product
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)
        self.marketplace.remove_from_cart(cart_id, product)

        # Test that a product in one cart cannot be added to another
        snd_cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(snd_cart_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), False)

        # Test that the product can be added after being removed from the other cart
        self.marketplace.remove_from_cart(snd_cart_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)

        # Test adding a product to a cart that has already been checked out
        self.marketplace.place_order(cart_id)
        self.marketplace.publish(producer_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)
        self.assertEqual(len(self.marketplace.place_order(cart_id)), 1)

    def test_remove_from_cart(self):
        """
        Test the logic for removing products from a cart.
        This covers cases like invalid cart IDs, null products, products not
        in the cart, and successful removal.
        """
        # Test removing from an invalid cart
        self.assertEqual(self.marketplace.remove_from_cart(123, None), False)

        # Test removing a null product
        cart_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, None), False)

        # Test removing a product that is not in the cart
        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), False)

        # Test successful removal
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), True)

        # Test removing from a cart that has already been checked out
        self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), False)

    def test_place_order(self):
        """
        Test the order placement logic.
        This verifies that placing an order for a cart with items returns the
        correct items and empties the cart.
        """
        # Test placing an order for an invalid cart ID
        self.assertEqual(self.marketplace.place_order(1234), None)

        # Test placing an order for a cart with one item
        producer_id = self.marketplace.register_producer()
        product = Tea("1", 2, "3")
        self.marketplace.publish(producer_id, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        result = self.marketplace.place_order(cart_id)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 1)

        # Test placing a new order with the same cart
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        result = self.marketplace.place_order(cart_id)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 1)

        # Test placing an order for an empty cart
        self.assertEqual(self.marketplace.place_order(cart_id), [])


class Marketplace:
    """
    A thread-safe marketplace that facilitates the interaction between
    producers and consumers. It manages product inventory, producer
    registrations, and customer shopping carts.
    """
    LOG_FILE = 'marketplace.log'
    MAX_BYTE_COUNT = 1000000
    BACKUP_COUNT = 5

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer: The maximum number of products a single
                                     producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producers = {}
        self.carts = {}
        self.register_lock = Lock()
        self.cart_lock = Lock()
        self.order_lock = Lock()
        self.crt_assignable_producer_idx = 0
        self.crt_assignable_cart_idx = 0

        self.logger = Logger.create_logger(__name__, Marketplace.LOG_FILE)

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.
        This method is thread-safe.

        Returns:
            The unique ID for the new producer.
        """
        with self.register_lock:
            producer_id = str(self.crt_assignable_producer_idx)
            self.crt_assignable_producer_idx += 1

        # Each producer gets a lock, a product list, and a product count
        self.producers[producer_id] = [Lock(), [], 0]

        self.logger.info("Registered a new producer with ID: %s.", producer_id)

        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a new product to the marketplace.
        The method fails if the producer's queue is full.

        Args:
            producer_id: The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            True if the product was published successfully, False otherwise.
        """
        if producer_id not in self.producers:
            self.logger.error("Unregistered producer ID: %s.", producer_id)
            return False

        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Producer: %s is trying to publish: %s", producer_id, product)

        # Check if the producer's queue is full
        if self.producers[producer_id][2] == self.queue_size_per_producer:
            self.logger.info("Producer: %s failed to "
                             "publish: %s. List is full.", producer_id, product)
            return False

        # Add product to the producer's list
        self.producers[producer_id][1].append(product)

        # Increment the product count for the producer
        with self.producers[producer_id][0]:
            self.producers[producer_id][2] += 1

        self.logger.info("Producer: %s successfully published: %s.", producer_id, product)

        return True

    def new_cart(self):
        """
        Creates a new shopping cart and returns its ID.
        This method is thread-safe.

        Returns:
            The unique ID for the new cart.
        """
        with self.cart_lock:
            cart_id = self.crt_assignable_cart_idx
            self.crt_assignable_cart_idx += 1

        self.carts[cart_id] = []

        self.logger.info("Generated a new cart with ID: %s", cart_id)

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.
        This involves finding the product in a producer's inventory and
        moving it to the cart. This operation is atomic for each producer.

        Args:
            cart_id: The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            True if the product was added successfully, False otherwise.
        """
        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return False

        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Consumer is trying to add "
                         "product: %s to cart: %s.", product, cart_id)

        # Iterate through all producers to find the product
        for key, value in self.producers.items():
            try:
                with value[0]:
                    # Atomically find and remove the product from producer's list
                    idx = value[1].index(product)
                    found_product = value[1].pop(idx)
            except ValueError:
                continue

            # Add the product and its original producer to the cart
            self.carts[cart_id].append((key, found_product))

            self.logger.info("Consumer has successfully "
                             "added product: %s to cart: %s", product, cart_id)

            return True

        self.logger.info("Consumer failed to add product: "
                         "%s to cart: %s. Product has not been found.", product, cart_id)

        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart, returning it to the
        original producer's inventory.

        Args:
            cart_id: The ID of the cart to remove from.
            product: The product to remove.

        Returns:
            True if removal was successful, False otherwise.
        """
        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return False

        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Consumer is trying to "
                         "remove product: %s from cart: %s.", product, cart_id)

        # Find the product in the cart
        for i in self.carts[cart_id]:
            if i[1] == product:
                # Return the product to the producer's inventory
                self.producers[i[0]][1].append(i[1])

                # Remove from cart
                self.carts[cart_id].remove(i)

                self.logger.info("Consumer has successfully "
                                 "removed product: %s from cart: %s.", product, cart_id)

                return True

        self.logger.info("Consumer failed to remove product: "
                         "%s from cart: %s. Product is not in cart.", product, cart_id)

        return False

    def place_order(self, cart_id):
        """
        Finalizes the purchase of all items in a cart.
        This decrements the product count for each producer whose item was sold
        and clears the cart.

        Args:
            cart_id: The ID of the cart to place an order for.

        Returns:
            A list of products that were purchased, or None if the cart ID is invalid.
        """
        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return None

        self.logger.info("Consumer is trying to "
                         "place order for contents of cart: %s", cart_id)

        result = []

        for i in self.carts[cart_id]:
            # Decrement the published product count for the producer
            with self.producers[i[0]][0]:
                self.producers[i[0]][2] -= 1

            # Log the purchase action
            with self.order_lock:
                print(str(threading.current_thread().name) + " bought " + str(i[1]))

            result.append(i[1])

        # Clear the cart after placing the order
        self.carts[cart_id] = []

        self.logger.info("Consumer is has successfully "
                         "placed order for contents of cart: %s", cart_id)

        return result

from copy import copy
import time
from threading import Thread


class Producer(Thread):
    """
    Represents a producer thread that continuously publishes products
    to the marketplace based on a given product list.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products: A list of products to be published by the producer.
                      Each item is a tuple of (product, quantity, wait_time).
            marketplace: The shared Marketplace object.
            republish_wait_time: The time to wait before retrying to publish
                                 if the marketplace queue is full.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer thread.
        It registers with the marketplace and then enters an infinite loop to
        publish its products.
        """
        producer_id = self.marketplace.register_producer()

        while True:
            for i in self.products:
                # Unpack product information
                product_template = i[0]
                quantity = i[1]
                waiting_time = i[2]

                for _ in range(0, quantity):
                    # Create a copy of the product to ensure uniqueness
                    product = copy(product_template)

                    # Attempt to publish the product, retrying if the queue is full
                    is_published = self.marketplace.publish(producer_id, product)
                    while not is_published:
                        time.sleep(self.republish_wait_time)
                        is_published = self.marketplace.publish(producer_id, product)

                    # Wait for a specified time before publishing the next product
                    time.sleep(waiting_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A basic data class representing a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class representing a type of Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    A data class representing a type of Coffee, inheriting from Product
    and adding acidity and roast level attributes.
    """
    acidity: str
    roast_level: str
