"""
This module simulates a multi-threaded producer-consumer marketplace.

It contains classes for a `Consumer`, `Marketplace`, `Producer`, unit tests,
and product data definitions.

Note: This file appears to be a concatenation of several separate modules.
This version of the Marketplace uses a coarse-grained lock to provide
thread safety for its operations.
"""
from threading import Thread
from time import sleep

# This import is defined later in the file, indicating concatenation.
# from tema.marketplace import Marketplace


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a shopping process.

    Each consumer is initialized with a list of shopping requests and processes
    them by creating a new cart for each request list, adding/removing items,
    and placing an order.
    """

    def __init__(self, carts: list, marketplace: 'Marketplace', retry_wait_time: int, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping action lists.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed 'add' action.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main execution logic for the consumer thread."""
        # Invariant: Process each list of actions as a separate shopping journey.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            # Invariant: Process each action within the current journey.
            for operation in cart:
                # Select the appropriate marketplace function based on the action type.
                function = self.marketplace.add_to_cart if operation['type'] == 'add' 
                    else self.marketplace.remove_from_cart

                # Block Logic: Perform the action for the specified quantity.
                for _ in range(operation['quantity']):
                    # If an 'add' action fails (e.g., product unavailable), wait and retry.
                    while function(cart_id, operation['product']) is False:
                        sleep(self.retry_wait_time)

            # Finalize the transaction for the current cart.
            product_list = self.marketplace.place_order(cart_id)

            if len(product_list) > 0:
                print("
".join([f"{self.name} bought {product}" for product in product_list]))

# --- Start of concatenated Marketplace, Testing, and Producer modules ---

from logging.handlers import RotatingFileHandler
from multiprocessing import Lock
import unittest
import logging
import time
# These imports suggest the classes were originally in a 'tema' package.
# from tema.product import Product, Coffee, Tea

# --- Logging Setup ---
logging.Formatter.converter = time.gmtime
ROTATING_FILE = RotatingFileHandler(filename='marketplace.log', maxBytes=1048576,
                                    backupCount=5)
ROTATING_FILE.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
LOGGER = logging.getLogger('marketplace')
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(ROTATING_FILE)


class Marketplace:
    """
    A thread-safe marketplace that manages producers, products, and customer carts.

    This class uses a single `multiprocessing.Lock` to serialize access to all
    shared data structures, ensuring consistency in a multi-threaded environment.
    """
    def __init__(self, queue_size_per_producer: int):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): Max products a producer can have in stock.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = 0
        self.producers_queue = {}  # Maps producer_id -> count of their items in stock.
        self.lock = Lock()
        self.carts = {}            # Maps cart_id -> list of (product, producer_id) tuples.
        self.cart_ids = 0
        self.products = []         # A global list of all available products as (product, producer_id).

    def register_producer(self):
        """
        Registers a new producer, returning a unique ID. This is thread-safe.

        Returns:
            int: The new producer's unique ID.
        """
        LOGGER.info('Registering a producer')
        with self.lock:
            self.producers += 1
            producer_id = self.producers
            self.producers_queue[producer_id] = 0
        LOGGER.info('Producer registered with id %s', producer_id)
        return producer_id

    def publish(self, producer_id: str, product: 'Product'):
        """
        Allows a producer to add a product to the marketplace.

        This method attempts to acquire a lock with a timeout to avoid blocking
        indefinitely if the marketplace is busy.

        Returns:
            bool: True if published successfully, False if the producer's queue is
                  full or the lock could not be acquired in time.
        """
        LOGGER.info('Producer %s is publishing a %s product', producer_id, product)
        acquired = self.lock.acquire(timeout=0.5)
        if not acquired or self.producers_queue[producer_id] >= self.queue_size_per_producer:
            if acquired:
                self.lock.release()
            LOGGER.info('Producer %s was not able to publish a %s product', producer_id, product)
            return False

        self.products.append((product, producer_id))
        self.producers_queue[producer_id] += 1
        self.lock.release()
        LOGGER.info('Producer %s published a %s product', producer_id, product)
        return True

    def new_cart(self):
        """
        Creates a new, empty cart for a consumer. This is thread-safe.

        Returns:
            int: The unique ID for the new cart.
        """
        LOGGER.info('Creating a new cart')
        with self.lock:
            self.cart_ids += 1
            cart_id = self.cart_ids
            self.carts[cart_id] = []
        LOGGER.info('Cart was created with id %s', cart_id)
        return cart_id

    def add_to_cart(self, cart_id: int, product: 'Product'):
        """
        Moves a product from the global inventory to a consumer's cart.

        This operation is atomic, protected by the marketplace lock.

        Returns:
            bool: True if the product was found and moved, False otherwise.
        """
        LOGGER.info('Adding a %s product to cart %s', product, cart_id)
        with self.lock:
            try:
                # Find the first available instance of the product.
                product_index = list(
                    map(lambda product_tuple: product_tuple[0], self.products)
                ).index(product)
            except ValueError:
                LOGGER.info('Product %s was not found in the marketplace', product)
                return False

            if cart_id not in self.carts:
                LOGGER.info('Cart %s was not found in the marketplace', cart_id)
                return False
            
            # Block Logic: Decrement producer's stock count, add to cart, and remove from global stock.
            self.producers_queue[self.products[product_index][1]] -= 1
            self.carts[cart_id].append((product, self.products[product_index][1]))
            del self.products[product_index]
        LOGGER.info('Product %s was added to cart %s', product, cart_id)
        return True

    def remove_from_cart(self, cart_id: int, product: 'Product'):
        """
        Moves a product from a cart back to the global inventory.

        This operation is atomic, protected by the marketplace lock.

        Returns:
            bool: True if the product was found and returned, False otherwise.
        """
        LOGGER.info('Removing a %s product from cart %s', product, cart_id)
        with self.lock:
            cart_product_list = list(map(lambda product_tuple: product_tuple[0], self.carts[cart_id]))
            if cart_id not in self.carts or product not in cart_product_list:
                LOGGER.info('Product %s was not found in cart %s', product, cart_id)
                return False

            try:
                product_index = cart_product_list.index(product)
            except ValueError:
                LOGGER.info('Product %s was not found in cart %s', product, cart_id)
                return False
            
            # Block Logic: Add product back to global stock, increment producer count, and remove from cart.
            self.products.append(self.carts[cart_id][product_index])
            self.producers_queue[self.carts[cart_id][product_index][1]] += 1
            del self.carts[cart_id][product_index]
        LOGGER.info('Product %s was removed from cart %s', product, cart_id)
        return True

    def place_order(self, cart_id: int):
        """
        Finalizes an order by returning the cart's contents and deleting the cart.

        This is a destructive read operation, as the cart is removed after the order.
        This operation is atomic.

        Returns:
            list: A list of the products that were in the cart, or None if the cart doesn't exist.
        """
        LOGGER.info('Placing order for cart %s', cart_id)
        with self.lock:
            if cart_id not in self.carts:
                LOGGER.info('Cart %s was not found in the marketplace', cart_id)
                return None
            products = self.carts[cart_id]
            del self.carts[cart_id]
        LOGGER.info('Order was placed for cart %s', cart_id)
        return list(map(lambda x: x[0], products))


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace logic."""
    def setUp(self) -> None:
        """Initializes a marketplace for each test."""
        super().setUp()
        self.marketplace = Marketplace(queue_size_per_producer=5)

    def test_register_producer(self):
        """Tests that producers get sequential IDs."""
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        """Tests that publishing respects the queue size limit."""
        producer_id = self.marketplace.register_producer()
        for _ in range(5):
            self.assertTrue(self.marketplace.publish(producer_id, Product("p1", 10)))
        self.assertFalse(self.marketplace.publish(producer_id, Product("p6", 10)))

    def test_new_cart(self):
        """Tests that new carts get sequential IDs."""
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
    
    # Other test cases follow a similar pattern...

class Producer(Thread):
    """
    Represents a producer thread that continuously adds products to the marketplace.
    """

    def __init__(self, products: list, marketplace: Marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """The main execution loop for the producer."""
        # A producer registers itself once at the beginning of its lifecycle.
        producer_id = self.marketplace.register_producer()

        while True:
            # Invariant: Continuously iterate through the assigned product list to publish.
            for (product, quantity, produce_time) in self.products:
                for _ in range(quantity):
                    sleep(produce_time)
                    # If publishing fails (e.g., queue is full), wait and retry.
                    while self.marketplace.publish(producer_id, product) is False:
                        sleep(self.republish_wait_time)

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
