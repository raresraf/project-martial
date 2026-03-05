"""
This module provides a robust, object-oriented simulation of a marketplace.

It implements a producer-consumer model using several classes to represent the
system's components. This version includes professional features such as:
- A `Marketplace` with separate locks for producer and customer operations to
  improve concurrency.
- The use of UUIDs for unique producer and cart IDs.
- A suite of unit tests (`TestMarketplace`) for verifying functionality.
- Configuration of logging for monitoring the simulation's execution.

Note: The file appears to be a composite of several separate modules.
"""

from time import sleep
from threading import Thread


class Consumer(Thread):
    """Represents a consumer that interacts with the marketplace."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping actions to perform.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a
                failed action.
            **kwargs: Keyword arguments for the Thread, including 'name'.
        """
        Thread.__init__(self, name=kwargs['name'])
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main execution logic for the consumer."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                if operation['type'] == 'add':
                    # For 'add' operations, retry until the product is successfully added.
                    for _ in range(operation['quantity']):
                        while not self.marketplace.add_to_cart(
                                cart_id, operation['product']):
                            sleep(self.retry_wait_time)

                if operation['type'] == 'remove':
                    # For 'remove' operations, assume the product is in the cart.
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(
                            cart_id, operation['product'])

            products = self.marketplace.place_order(cart_id)
            for prod in products:
                # Use a lock to ensure print statements are not garbled.
                with self.marketplace.printing_lock:
                    print(f'{self.name} bought {prod}')

from uuid import uuid4
import unittest


import logging
from logging.handlers import RotatingFileHandler

from threading import Lock
import time

from .product import Product


def logger_set_up():
    """Configures a rotating file logger for the marketplace."""
    logging.basicConfig(
        handlers=[RotatingFileHandler(
            'marketplace.log', maxBytes=10000, backupCount=10)],
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')

    logging.Formatter.converter = time.gmtime


class Marketplace:
    """
    A thread-safe marketplace implementation with improved concurrency and data
    structures.
    """

    def __init__(self, queue_size_per_producer: int):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The max number of items a single
                producer can have listed at one time.
        """
        logger_set_up()
        self.queue_size_per_producer = queue_size_per_producer

        # Data structures for managing the marketplace state.
        self.producers_queues: dict[str, list[Product]] = {}
        self.available_products: dict[Product, int] = {}
        self.carts: dict[int, list] = {}
        
        # Separate locks for customer and producer operations to reduce contention.
        self.customer_lock = Lock()
        self.producer_lock = Lock()
        self.printing_lock = Lock()

    def register_producer(self):
        """Registers a new producer with a unique ID."""
        logging.info('register producer started.')
        with self.producer_lock:
            p_id = uuid4().hex
            self.producers_queues[p_id] = []
        logging.info('register producer finished. Returned %s.', p_id)
        return p_id



    def publish(self, producer_id: str, product: Product):
        """
        Publishes a product from a producer.

        Note: This method is not thread-safe. A race condition can occur when
        multiple threads modify `self.available_products` simultaneously. A lock
        is needed here.
        """
        logging.info(
            'publish started. Parameters: producer_id = %s, product = %s.', producer_id, product)
        
        if len(self.producers_queues[producer_id]) == self.queue_size_per_producer:
            logging.info('publish finished. Returned False.')
            return False

        self.producers_queues[producer_id].append(product)

        if product not in self.available_products:
            self.available_products[product] = 1
        else:
            self.available_products[product] += 1

        logging.info('publish finished. Returned True.')
        return True

    def new_cart(self):
        """Creates a new cart for a consumer, returning a unique cart ID."""
        logging.info('new_cart started.')
        with self.customer_lock:
            cart_id = uuid4().int
            self.carts[cart_id] = []
        logging.info('new_cart finished. Returned %i', cart_id)
        return cart_id

    def add_to_cart(self, cart_id: int, product: Product):
        """Adds a product to a consumer's cart.

        This implementation is efficient, checking an aggregate dictionary
        `available_products` in O(1) average time.
        """
        logging.info(
            'add_to_cart started. Parameters: cart_id=%i, product=%s.', cart_id, product)

        with self.customer_lock:
            # Check for product availability.
            if product not in self.available_products or self.available_products[product] == 0:
                logging.info('add_to_cart finished. Returned False.')
                return False
            
            self.available_products[product] -= 1
            self.carts[cart_id].append(product)

        logging.info('add_to_cart finished. Returned True.')
        logging.debug('added')
        return True

    def remove_from_cart(self, cart_id: int, product: Product):
        """Removes a product from a cart and makes it available again."""
        logging.info(
            'remove_from_cart started. Parameters: cart_id=%i, product=%s.', cart_id, product)
        with self.customer_lock:
            self.carts[cart_id].remove(product)
            self.available_products[product] += 1
        logging.info('remove_from_cart finished.')



    def place_order(self, cart_id: int):
        """Finalizes an order, permanently removing items from the market.

        Note: The logic to find and remove products involves nested loops, making
        it inefficient (O(Items in Cart * Total Items in Market)).
        """
        logging.info('place_order started. Parameters: cart_id=%i.', cart_id)
        bought_items = []

        for product in self.carts[cart_id]:
            for producer_queue in self.producers_queues.values():
                if product in producer_queue:
                    bought_items.append(product)
                    producer_queue.remove(product)
                    break

        logging.info('place_order finished. Returned %s.', bought_items)
        return bought_items


class TestMarketplace(unittest.TestCase):
    """A suite of unit tests for the Marketplace class."""

    def setUp(self):
        """Set up a new marketplace instance for each test."""
        self.marketplace = Marketplace(1)

    def test_register_producer_return_str(self):
        """Tests that register_producer returns a string ID."""
        p_id = self.marketplace.register_producer()
        self.assertEqual(type(p_id), str)

    def test_new_cart_return_int(self):
        """Tests that new_cart returns an integer ID."""
        c_id = self.marketplace.new_cart()
        self.assertEqual(type(c_id), int)

    def test_publish_if_queue_not_full_then_return_true(self):
        """Tests that publishing succeeds when producer queue has capacity."""
        p_id = self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(p_id, Product('Tea', 11)),
                         True)

    def test_publish_if_queue_full_then_return_true(self):
        """Tests that publishing fails when producer queue is full."""
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11))
        self.assertEqual(self.marketplace.publish(
            p_id, Product('Tea', 11)), False)

    def test_add_to_cart_if_product_not_available_return_false(self):
        """Tests that adding an unavailable product to a cart fails."""
        c_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(
            c_id, Product('Tea', 11)), False)

    def test_add_to_cart_if_product_available_return_true(self):
        """Tests that adding an available product to a cart succeeds."""
        c_id = self.marketplace.new_cart()
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11))

        self.assertEqual(self.marketplace.add_to_cart(
            c_id, Product('Tea', 11)), True)

    def test_remove_from_cart(self):
        """Tests the functionality of removing an item from a cart."""
        c_id = self.marketplace.new_cart()
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11))
        self.marketplace.add_to_cart(
            c_id, Product('Tea', 11))
        self.marketplace.remove_from_cart(c_id, Product('Tea', 11))
        self.assertEqual(len(self.marketplace.carts[c_id]), 0)

    def test_place_order(self):
        """Tests that placing an order correctly removes items from the market."""
        p_id = self.marketplace.register_producer()
        c_id = self.marketplace.new_cart()
        self.marketplace.publish(p_id, Product('Tea', 11))
        self.marketplace.add_to_cart(
            c_id, Product('Tea', 11))
        self.marketplace.place_order(c_id)
        self.assertEqual(len(self.marketplace.producers_queues[p_id]), 0)


from time import sleep
from threading import Thread


class Producer(Thread):
    """Represents a producer that publishes products to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread.

        Args:
            products (list): A list of products to publish.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.p_id = ''

    def run(self):
        """The main execution logic for the producer."""
        self.p_id = self.marketplace.register_producer()
        while True:
            for prod_info in self.products:
                for _ in range(prod_info[1]):
                    sleep(prod_info[2])
                    # Retry publishing until it succeeds.
                    while not self.marketplace.publish(self.p_id, prod_info[0]):
                        sleep(self.republish_wait_time)
