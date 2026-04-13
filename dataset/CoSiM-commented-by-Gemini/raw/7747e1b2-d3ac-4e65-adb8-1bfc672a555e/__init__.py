"""
# @raw/7747e1b2-d3ac-4e65-adb8-1bfc672a555e/__init__.py
# @brief A multithreaded producer-consumer marketplace simulation.
#
# This module provides a thread-safe marketplace where multiple producers can publish
# products and multiple consumers can buy them. The simulation uses locks to manage
# concurrent access to shared resources, ensuring data consistency.
"""
from threading import Thread
import time
import sys


class Consumer(Thread):
    """
    Represents a consumer in the marketplace. Each consumer is a thread that
    processes a list of shopping carts, adding or removing products as specified.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a sequence of actions.
            marketplace (Marketplace): The marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a failed action.
            **kwargs: Keyword arguments for the Thread, including the consumer's name.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        The main loop for the consumer thread. It processes each cart's actions,
        places an order, and prints the purchased items.
        """
        for cart in self.carts:
            # Each consumer gets a new cart ID for each cart they process.
            cart_id = self.marketplace.new_cart()

            for action in cart:
                action_type = action['type']
                product = action['product']
                quantity = action['quantity']

                # Perform 'add' or 'remove' actions for the specified quantity.
                if action_type == "add":
                    # Keep trying to add the product until successful.
                    for _ in range(quantity):
                        while not self.marketplace.add_to_cart(cart_id, product):
                            # If adding fails, wait and retry.
                            time.sleep(self.retry_wait_time)
                else:
                    # Remove the specified quantity of the product from the cart.
                    for _ in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

            # After processing all actions, place the order and print the results.
            for order in self.marketplace.place_order(cart_id):
                sys.stdout.flush()
                print(f"{self.name} bought {order}")


from threading import Lock
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Counter
import unittest
from tema.product import Coffee, Tea


def set_logger():
    """Configures and returns a rotating file logger for the marketplace."""
    formatter = logging.Formatter(
        '[%(asctime)s] --> %(levelname)s: %(message)s')
    formatter.converter = time.gmtime

    handler = RotatingFileHandler(
        'marketplace.log', maxBytes=100000, backupCount=10)
    handler.setFormatter(formatter)

    logger = logging.getLogger('marketplace info logger')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


class Marketplace:
    """
    The central marketplace class that manages producers, consumers, and inventory.
    It uses locks to ensure thread-safe operations.
    """

    logger = set_logger()

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a producer can
                                           have in the marketplace at one time.
        """
        self.logger.info(
            "Marketplace initialized with maximum queue size: %s.", queue_size_per_producer)

        self.queue_size_per_producer = queue_size_per_producer
        self.producer_number = 0
        self.consumer_cart_number = 0

        # Data structures for managing producers, consumers, and products.
        self.producers_products = {}
        self.consumers_carts = {}
        self.products_queue = {}

        # Locks to protect shared data from concurrent access.
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.queue_lock = Lock()

        self.logger.info("Initiated marketplace parameters.")

    def register_producer(self):
        """
        Registers a new producer and returns a unique producer ID.

        Returns:
            str: The unique ID for the new producer.
        """
        self.logger.info("A new producer tries to obtain an id.")
        # Atomically get a new producer ID.
        with self.producer_lock:
            producer_id = self.producer_number
            self.producer_number = self.producer_number + 1

        # Initialize the product count for the new producer.
        self.producers_products[producer_id] = 0
        self.logger.info(
            "Generated the producer id number %s.", producer_id)
        return str(producer_id)

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace from a given producer.

        Args:
            producer_id (str): The ID of the producer.
            product (Product): The product to publish.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue is full.
        """
        self.logger.info(
            "Producer with id %s wants to publish %s.", producer_id, product)
        producer_id = int(producer_id)

        # Check if the producer has reached their queue limit.
        if self.producers_products[producer_id] >= self.queue_size_per_producer:
            self.logger.info("Producer %s queue is full, cannot publish %s.", producer_id, product)
            return False

        # Increment the producer's product count.
        self.producers_products[producer_id] += 1
        with self.queue_lock:
            # Create a new queue for the product if it doesn't exist.
            if not product in self.products_queue:
                self.products_queue[product] = []

            # Add the product to the queue, marking it as available.
            self.products_queue[product].append((producer_id, True))

        self.logger.info(
            "Producer with id %s published %s.", producer_id, product)
        return True

    def new_cart(self):
        """
        Creates a new shopping cart for a consumer and returns a unique cart ID.

        Returns:
            int: The unique ID for the new cart.
        """
        self.logger.info("A consumer tries to obtain a new cart id.")
        # Atomically get a new cart ID.
        with self.consumer_lock:
            consumer_cart_id = self.consumer_cart_number
            self.consumer_cart_number = self.consumer_cart_number + 1

        # Initialize the cart in the consumers' carts dictionary.
        self.consumers_carts[consumer_cart_id] = {}
        self.logger.info(
            "Generated the cart id number %s.", consumer_cart_id)
        return consumer_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was added successfully, False otherwise.
        """
        self.logger.info(
            "Consumer with cart id %s wants to add %s.", cart_id, product)
        with self.queue_lock:
            # Check if the product is available in the marketplace.
            if not product in self.products_queue:
                self.logger.info("Product %s not available for cart %s.", product, cart_id)
                return False

            for index in range(len(self.products_queue[product])):
                product_queue = self.products_queue[product][index]
                # Check if the product is marked as available.
                if product_queue[1] is True:
                    # Mark the product as unavailable (reserved for the cart).
                    self.products_queue[product][index] = (
                        product_queue[0], False)

                    # Add the product to the consumer's cart.
                    if not product in self.consumers_carts[cart_id]:
                        self.consumers_carts[cart_id][product] = []
                    self.consumers_carts[cart_id][product].append(index)

                    # Decrement the product count for the original producer.
                    with self.producer_lock:
                        self.producers_products[self.products_queue[product]
                                                [index][0]] -= 1
                    self.logger.info(
                        "Consumer with cart id %s added %s.", cart_id, product)
                    return True

        self.logger.info("Could not add product %s to cart %s.", product, cart_id)
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart, making it available again.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Product): The product to remove.
        """
        self.logger.info(
            "Consumer with cart id %s wants to remove %s.", cart_id, product)
        with self.queue_lock:
            if len(self.consumers_carts[cart_id][product]) == 0:
                raise Exception("No product to be removed from cart")

            # Get the product's index from the cart and mark it as available again.
            index = self.consumers_carts[cart_id][product].pop()
            product_queue = self.products_queue[product][index]
            self.products_queue[product][index] = (product_queue[0], True)

            # Increment the product count for the original producer.
            with self.producer_lock:
                self.producers_products[self.products_queue[product]
                                        [index][0]] += 1
        self.logger.info(
            "Consumer with cart id %s removed %s.", cart_id, product)

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart. The products in the cart are
        considered "sold" and are removed from the system.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of products that were in the cart.
        """
        self.logger.info(
            "Consumer with cart id %s wants to place order.", cart_id)
        order = []
        consumer_cart = self.consumers_carts[cart_id]
        for product in consumer_cart.keys():
            for _ in consumer_cart[product]:
                order.append(product)

        # Clear the cart after placing the order.
        self.consumers_carts[cart_id] = {}
        self.logger.info(
            "Consumer with cart id %s placed order: %s.", cart_id, order)
        return order


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        queue_size_per_producer = 5
        self.marketplace = Marketplace(queue_size_per_producer)
        # Sample products for testing.
        self.teas = [Tea("Tabiets", 5, "Black"), Tea("Aroma Tea", 7.5, "Mint"), Tea("Honey", 5, "Green")]
        self.coffees = [Coffee("Davidoff", 10, 4.5, "STRONG"), Coffee("Romantique", 6, 3.0, "MILD"), Coffee("Costa", 8, 5.0, "EXTRA STRONG")]

    def test_register_producer(self):
        """Tests that producer registration returns unique and sequential IDs."""
        self.assertEqual(self.marketplace.register_producer(), '0', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(), '1', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(), '2', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(), '3', "Incorrect producer id.")

    def test_publish(self):
        """Tests the product publishing logic, including queue limits."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()

        # Test publishing up to the queue limit.
        self.assertTrue(self.marketplace.publish('0', self.teas[0]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish('0', self.teas[1]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish('0', self.teas[0]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish('0', self.teas[2]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish('0', self.coffees[0]), "Did not publish coffee")
        # This should fail as the queue is full.
        self.assertFalse(self.marketplace.publish('0', self.coffees[1]), "Reached max queue")

        # Test that publishing is possible again after a product is removed.
        self.marketplace.add_to_cart(0, self.coffees[0])
        self.assertTrue(self.marketplace.publish('0', self.coffees[1]), "Did not publish coffee")
        self.assertTrue(self.marketplace.publish('1', self.teas[2]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish('1', self.coffees[2]), "Did not publish coffee")

    def test_new_cart(self):
        """Tests that new carts get unique and sequential IDs."""
        self.assertEqual(self.marketplace.new_cart(), 0, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 1, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 2, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 3, "Incorrect cart id.")

    def test_add_to_cart(self):
        """Tests adding available and unavailable products to a cart."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()

        # Test adding a product that has not been published.
        self.assertFalse(self.marketplace.add_to_cart(0, self.coffees[1]), "Inexistent coffee")

        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])
        # Test adding an available product.
        self.assertTrue(self.marketplace.add_to_cart(1, self.teas[0]), "Did not add tea")
        # Test that the same product cannot be added again if it's already reserved.
        self.assertFalse(self.marketplace.add_to_cart(0, self.teas[0]), "Inexistent tea")

        self.marketplace.publish('1', self.coffees[1])
        self.marketplace.publish('1', self.coffees[2])
        self.assertFalse(self.marketplace.add_to_cart(1, self.coffees[0]), "Inexistent coffee")
        self.assertTrue(self.marketplace.add_to_cart(0, self.coffees[1]), "Did not add coffee")
        self.assertTrue(self.marketplace.add_to_cart(0, self.coffees[2]), "Did not add coffee")

    def test_remove_from_cart(self):
        """Tests removing products from a cart."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()

        # Test removing from an empty cart.
        self.assertRaises(Exception, self.marketplace.remove_from_cart, 0, self.coffees[0])

        self.marketplace.publish('1', self.coffees[0])
        self.marketplace.add_to_cart(1, self.coffees[0])
        self.assertFalse(self.marketplace.add_to_cart(0, self.coffees[0]), "Inexistent coffee")

        # Test that after removing a product, it becomes available again.
        self.marketplace.remove_from_cart(1, self.coffees[0])
        self.assertTrue(self.marketplace.add_to_cart(0, self.coffees[0]), "Did not add coffee")

        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])
        self.marketplace.publish('0', self.teas[2])
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])

        self.marketplace.add_to_cart(0, self.teas[0])
        self.marketplace.publish('0', self.teas[2])
        self.marketplace.add_to_cart(0, self.teas[1])
        self.marketplace.remove_from_cart(0, self.teas[1])
        # Test that a producer can publish again after a product is removed from a cart.
        self.assertFalse(self.marketplace.publish('0', self.coffees[0]), "Reached max queue")

    def test_place_order(self):
        """Tests the order placement logic."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()

        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('1', self.coffees[0])
        self.marketplace.publish('0', self.teas[2])
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('1', self.coffees[1])

        self.marketplace.add_to_cart(0, self.teas[0])
        self.marketplace.add_to_cart(0, self.coffees[1])
        # Test that the order contains the correct items.
        self.assertEqual(Counter(self.marketplace.place_order(0)), Counter([self.teas[0], self.coffees[1]]))

        self.marketplace.add_to_cart(0, self.coffees[0])
        self.assertEqual(Counter(self.marketplace.place_order(0)), Counter([self.coffees[0]]))

        self.marketplace.add_to_cart(1, self.teas[0])
        self.marketplace.remove_from_cart(1, self.teas[0])
        # Test that an empty cart results in an empty order.
        self.assertEqual(self.marketplace.place_order(1), [])


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer in the marketplace. Each producer is a daemon thread that
    continuously produces and publishes products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to be produced. Each item in the list
                             is a tuple of (product_data, quantity, wait_time).
            marketplace (Marketplace): The marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Keyword arguments for the Thread.
        """
        Thread.__init__(self)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()
        self.daemon = True

    def run(self):
        """
        The main loop for the producer. It iterates through its products,
        produces them, and publishes them to the marketplace.
        """
        while True:
            for product in self.products:
                product_data = product[0]
                quantity = product[1]
                wait_time = product[2]

                # Produce the specified quantity of the product.
                for _ in range(quantity):
                    # Keep trying to publish until successful.
                    while not self.marketplace.publish(self.producer_id, product_data):
                        sleep(self.republish_wait_time)

                # Wait for a specified time before producing the next product.
                sleep(wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    A base class for products, implemented as a dataclass.
    It is hashable and equatable, allowing it to be used in dictionaries and sets.
    """
    name: str
    price: int

    def __hash__(self):
        return hash((self.name, self.price))

    def __eq__(self, other):
        return (self.name, self.price) == (other.name, other.price)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
