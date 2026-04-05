"""A multi-threaded producer-consumer marketplace simulation.

This script implements a marketplace system where multiple Producer threads can
publish products and multiple Consumer threads can purchase them. The script is
a monolith containing all classes: Consumer, Producer, Marketplace, Product
data classes, and unit tests.
"""
#
# =============================================================================
#
#                                CONSUMER
#
# =============================================================================
import time
from threading import Thread


class Consumer(Thread):
    """Represents a consumer thread that processes a list of shopping carts.

    Each consumer thread simulates a shopper who attempts to add and remove a
    predefined list of products from their cart in the central marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping lists (carts) to be processed.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an operation
                that failed (e.g., product not available).
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.carts = carts

    def run(self):
        """The main execution logic for the consumer thread.

        For each shopping cart, this method acquires a new cart ID from the
        marketplace and then iterates through the products in the cart,
        performing 'add' or 'remove' operations. If an 'add' operation fails
        (e.g., the product is out of stock), the thread waits and retries.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for item in cart:
                operation_type = item["type"]
                prod = item["product"]
                quantity = item["quantity"]
                i = 0
                while i < quantity:
                    if operation_type == "add":
                        ret_val = self.marketplace.add_to_cart(cart_id, prod)
                    elif operation_type == "remove":
                        ret_val = self.marketplace.remove_from_cart(cart_id, prod)
                    else:
                        raise ValueError(
                            f'Invalid operation type: {operation_type}.'
                            f'The operation type must be add or remove.')

                    # If the operation succeeded, move to the next item.
                    # Otherwise, wait and retry the current operation.
                    if ret_val or ret_val is None:
                        i = i + 1
                    else:
                        time.sleep(self.retry_wait_time)
            # After processing all items, finalize the purchase.
            self.marketplace.place_order(cart_id)

#
# =============================================================================
#
#                                MARKETPLACE & TESTS
#
# =============================================================================
import unittest
import logging

from threading import Lock, currentThread
from logging.handlers import RotatingFileHandler
from time import gmtime

# This import seems to be from a relative path of the original project structure
# and is not included in this file. We assume it defines the Product classes.
# from tema.product import Tea


class Marketplace:
    """Manages product inventory and transactions between producers and consumers.

    This class is the thread-safe core of the simulation. It uses several locks
    to manage concurrent access to its internal data structures, which track
    producers, products, and customer carts. It also includes integrated logging
    to record all major events.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                single producer can have listed at once.
        """
        self.last_producer_id = -1
        self.last_cart_id = -1

        # Concurrency control locks
        self.last_producer_lock = Lock()
        self.last_cart_lock = Lock()
        self.producer_lock = Lock() # A general-purpose lock, its scope is broad.
        self.print_lock = Lock()

        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {}  # {producer_id: number_of_published_products}
        self.available_products = {} # {product: [producer_id_1, producer_id_2]}
        self.carts = {}  # {cart_id: [product_1, product_2]}
        self.all_products = []  # A flat list of all products currently for sale.
        self.unavailable_products = {} # {product: [producer_id_1]}

        # Setup for logging marketplace events to a rotating file.
        self.logger = logging.getLogger('marketplace_logger')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname) '
                                      '- s%(funcName)21s() - %(message)s')
        logging.Formatter.converter = gmtime
        self.handler = RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=10)
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def register_producer(self):
        """Registers a new producer, providing a unique ID.

        Returns:
            str: The unique ID assigned to the producer.
        """
        with self.last_producer_lock:
            self.last_producer_id = self.last_producer_id + 1
            crt_id = self.last_producer_id

        self.producers[str(crt_id)] = 0
        self.logger.info(f'Registered producer with id: {crt_id}')
        return str(crt_id)

    def publish(self, producer_id, product):
        """Publishes a product to the marketplace.

        Args:
            producer_id (str): The ID of the publishing producer.
            product (Product): The product to be published.

        Returns:
            bool: True if publishing was successful, False if the producer's
                  personal queue is full.
        """
        self.logger.info(f'Producer with id {producer_id} wants to publish product {product}')
        
        # Pre-condition: Check if the producer has reached their publication limit.
        if self.producers[producer_id] >= self.queue_size_per_producer:
            self.logger.info(f'Producer with id {producer_id} can not publish product {product}'
                             f'because they reached the queue size per producer')
            return False

        # Block-Logic: This lock prevents race conditions if multiple threads
        # were to modify the same producer's count simultaneously.
        with self.producer_lock:
            self.producers[producer_id] = self.producers[producer_id] + 1

        if product not in self.available_products:
            self.available_products[product] = []
        self.available_products[product].append(producer_id)
        
        self.all_products.append(product)
        self.logger.info(f'Producer with id {producer_id} published product {product}')
        return True

    def new_cart(self):
        """Creates a new, empty cart and returns its unique ID.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.last_cart_lock:
            self.last_cart_id = self.last_cart_id + 1
            crt_cart_id = self.last_cart_id

        self.carts[crt_cart_id] = []
        self.logger.info(f'Registered new cart with id: {crt_cart_id}')
        return crt_cart_id

    def add_to_cart(self, cart_id, product):
        """Adds an available product to a specified shopping cart.

        Args:
            cart_id (int): The ID of the target cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False if it was
                  not available.
        """
        self.logger.info(f'A consumer wants to add in cart with id {cart_id} the product {product}')
        if product in self.all_products:
            self.carts[cart_id].append(product)
            # This lock protects the producer's product count.
            with self.producer_lock:
                self.producers[self.available_products[product][-1]] -= 1

            # Update inventory lists to reflect the purchase.
            self.all_products.remove(product)
            if product not in self.unavailable_products:
                self.unavailable_products[product] = []
            self.unavailable_products[product].append(self.available_products[product].pop())
            self.logger.info(f'A consumer added in cart with id {cart_id} the product {product}')
            return True

        self.logger.info(f'A consumer could not add in cart '
                         f'with id {cart_id} the product {product},'
                         f'because the product is not available')
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace inventory.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product (Product): The product to be removed.
        """
        self.logger.info(f'A consumer wants to remove from cart'
                         f' with id {cart_id} the product {product}')
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)

            # Atomically update the producer's product count.
            with self.producer_lock:
                self.producers[self.unavailable_products[product][-1]] += 1

            # Move the product from the "unavailable" (in-cart) state back to "available".
            self.all_products.append(product)
            self.available_products[product]\
                .append(self.unavailable_products[product].pop())
            self.logger.info(f'A consumer removed from cart '
                             f'with id {cart_id} the product {product}')

        else: # The 'else' was missing, added for clarity of logic.
            self.logger.error(f'A consumer wanted to remove from cart '
                              f'with id {cart_id} the product {product},'
                              f'but the product is not in that cart')

    def place_order(self, cart_id):
        """Finalizes an order by 'checking out' the items in a cart.

        This method simply prints the items that were in the cart. The print
        operation is locked to prevent interleaved output from multiple threads.
        Args:
            cart_id (int): The ID of the cart to be checked out.
        """
        self.logger.info(f'A consumer wants to place an order for cart with id {cart_id}')
        for product in self.carts.pop(cart_id, []): # Use default to avoid error on empty cart.
            with self.print_lock:
                print(f'{currentThread().getName()} bought {product}')
        self.logger.info(f'A consumer placed an order for cart with id {cart_id}.')


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Set up a new Marketplace instance for each test."""
        self.marketplace = Marketplace(1)

    def test_register_producer_with_one_producer(self):
        """Tests that a single producer is registered with the correct ID (0)."""
        producer_id = self.marketplace.register_producer()
        self.assertEqual(producer_id, '0')
        self.assertEqual(str(self.marketplace.last_producer_id), producer_id)

    def test_register_producer_with_multiple_producers(self):
        """Tests that multiple producers receive sequential, unique IDs."""
        producer_id1 = self.marketplace.register_producer()
        producer_id2 = self.marketplace.register_producer()
        producer_id3 = self.marketplace.register_producer()

        self.assertEqual(producer_id1, '0')
        self.assertEqual(producer_id2, '1')
        self.assertEqual(producer_id3, '2')
        self.assertEqual(str(self.marketplace.last_producer_id), producer_id3)

    def test_publish_without_producer_limit(self):
        """Tests a successful product publication when the producer queue is not full."""
        product = Tea(name="Linden", type="Herbal", price=9)

        producer_id = self.marketplace.register_producer()
        ret_val = self.marketplace.publish(producer_id, product)

        self.assertTrue(ret_val)
        self.assertEqual(self.marketplace.producers[producer_id], 1)
        self.assertIsNotNone(self.marketplace.available_products)
        self.assertIsNotNone(self.marketplace.all_products)

    def test_publish_with_producer_limit(self):
        """Tests that publishing fails when a producer's queue is full."""
        product1 = Tea(name="Linden", type="Herbal", price=9)
        product2 = Tea(name="Lipton", type="Herbal", price=10)

        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, product1)
        ret_val = self.marketplace.publish(producer_id, product2) # This should fail.

        self.assertFalse(ret_val)
        self.assertEqual(self.marketplace.producers[producer_id], 1)
        self.assertNotIn(product2, self.marketplace.all_products)

    def test_new_cart_with_one_cart(self):
        """Tests that a single new cart gets the correct ID (0)."""
        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, 0)
        self.assertEqual(self.marketplace.last_cart_id, cart_id)

    def test_new_cart_with_multiple_carts(self):
        """Tests that multiple new carts get sequential, unique IDs."""
        cart_id1 = self.marketplace.new_cart()
        cart_id2 = self.marketplace.new_cart()
        cart_id3 = self.marketplace.new_cart()

        self.assertEqual(cart_id1, 0)
        self.assertEqual(cart_id2, 1)
        self.assertEqual(cart_id3, 2)
        self.assertEqual(self.marketplace.last_cart_id, cart_id3)

    def test_add_to_cart_for_available_product(self):
        """Tests successfully adding an available product to a cart."""
        product = Tea(name="Linden", type="Herbal", price=9)

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, product)
        ret_val = self.marketplace.add_to_cart(cart_id, product)

        self.assertTrue(ret_val)
        self.assertIn(product, self.marketplace.carts[cart_id])
        self.assertNotIn(product, self.marketplace.all_products)
        self.assertEqual(self.marketplace.producers[producer_id], 0)

    def test_add_to_cart_for_unavailable_product(self):
        """Tests that adding an unavailable product to a cart fails."""
        product = Tea(name="Linden", type="Herbal", price=9)

        cart_id = self.marketplace.new_cart()
        ret_val = self.marketplace.add_to_cart(cart_id, product)

        self.assertFalse(ret_val)
        self.assertNotIn(product, self.marketplace.carts[cart_id])

    def test_remove_from_cart(self):
        """Tests that a product is correctly returned to inventory when removed from a cart."""
        product = Tea(name="Linden", type="Herbal", price=9)

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.remove_from_cart(cart_id, product)

        self.assertNotIn(product, self.marketplace.carts[cart_id])
        self.assertIn(product, self.marketplace.all_products)
        self.assertEqual(self.marketplace.producers[producer_id], 1)

    def test_place_order(self):
        """Tests that placing an order successfully empties the cart."""
        product = Tea(name="Linden", type="Herbal", price=9)

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.place_order(cart_id)

        self.assertNotIn(cart_id, self.marketplace.carts)

#
# =============================================================================
#
#                                PRODUCER
#
# =============================================================================

class Producer(Thread):
    """Represents a producer thread that generates and publishes products.

    The producer runs in an infinite loop, attempting to publish items from a
    predefined list to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer instance.

        Args:
            products (list): A list of products for the producer to generate.
                Each item is a tuple of (product, quantity, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before producing the
                next item after a successful publication.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        self.unique_id = marketplace.register_producer()

    def run(self):
        """The main execution logic for the producer thread.

        Continuously cycles through its product list. For each product, it
        attempts to publish the specified quantity. If publishing fails (e.g.,
        its queue is full), it waits for the product's defined `wait_time`
        before retrying.
        """
        while True:
            for crt_product_info in self.products:
                (product, quantity, wait_time) = crt_product_info
                count = 0
                while count < quantity:
                    # `has_to_wait` is true if publishing failed.
                    has_to_wait = not self.marketplace.publish(self.unique_id, product)
                    if has_to_wait:
                        # If queue is full, wait for the item-specific time and retry.
                        time.sleep(wait_time)
                    else:
                        # If successful, wait a generic time before making the next product.
                        time.sleep(self.republish_wait_time)
                        count = count + 1

#
# =============================================================================
#
#                                PRODUCTS
#
# =============================================================================
from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing a Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
