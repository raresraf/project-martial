"""
This module contains several concatenated versions of a producer-consumer
marketplace simulation. The file is disorganized and contains multiple, slightly
different definitions for classes like `Marketplace` and `Producer`.

The documentation will proceed linearly through the file, commenting on each
component as it appears, while noting the duplicated and overlapping definitions.

The primary implementation appears to be the second `Marketplace` class, which
is what the `TestMarketplace` class targets.
"""


from threading import Thread, Lock, currentThread
from time import sleep
import time
import unittest
import logging
from logging.handlers import RotatingFileHandler


class Consumer(Thread):
    """
    Represents a consumer thread that shops in the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists for the consumer to process.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): Time to wait before retrying to add a product.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the consumer thread.
        It gets a cart, processes a list of operations, and places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for ops in cart:
                if ops['type'] == "add":
                    # Invariant: Persistently attempt to add the specified quantity of a product.
                    for _ in range(0, ops['quantity']):
                        while self.marketplace.add_to_cart(cart_id, ops['product']) is not True:
                            sleep(self.retry_wait_time)
                else:
                    for _ in range(0, ops['quantity']):
                        self.marketplace.remove_from_cart(cart_id, ops['product'])
            
            products = self.marketplace.place_order(cart_id)

            # Use a shared lock to prevent interleaved print statements.
            lock = self.marketplace.get_consumer_lock()
            with lock:
                for product in products:
                    print(self.kwargs['name'] + " bought " + str(product))


# =============================================================================
# Logging and Test Class Definition
# =============================================================================

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO,
                    handlers=[RotatingFileHandler('marketplace.log',
                                                  maxBytes=20000, backupCount=10)])
logging.Formatter.converter = time.gmtime


class Marketplace:
    """
    Manages inventory and transactions in a thread-safe manner.

    This implementation uses a dictionary-based approach to track products from
    different producers and manage shopping carts.

    Concurrency Model:
    This class employs a coarse-grained locking strategy. A single `consumer_lock`
    is used to serialize all consumer-facing operations (new_cart, add, remove, place_order),
    which can be a significant performance bottleneck. A separate `producer_lock`
    handles producer registration and publishing.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.producer_id = -1
        self.cart_id = -1
        self.size_per_producer = {} # Tracks the number of items published by each producer.
        self.carts = {} # Maps cart_id -> {product -> [producer_id_1, producer_id_2]}
        self.products_dict = {} # Maps product -> [producer_id_1, producer_id_2]

    def register_producer(self):
        """Registers a new producer with a unique sequential ID."""
        with self.producer_lock:
            logging.info("New producer entered register_producer method")
            self.producer_id += 1
            self.size_per_producer[self.producer_id] = 0
        logging.info("New producer registered with id %d", self.producer_id)
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product.

        Returns:
            bool: True if successful, False if the producer's queue is full.
        """
        logging.info("Producer with id %d entered publish method", producer_id)
        with self.producer_lock:
            # Pre-condition: Check if the producer has space to publish.
            if self.size_per_producer[producer_id] == self.queue_size_per_producer:
                logging.info(f"Producer with id {producer_id} failed to publish product {product}")
                return False
            # Add the product to the global inventory, tracking its origin.
            if product not in self.products_dict:
                self.products_dict[product] = [producer_id]
            else:
                self.products_dict[product].append(producer_id)
            self.size_per_producer[producer_id] += 1
            logging.info(f"Producer with id {producer_id} published product {product}")
        return True

    def new_cart(self):
        """Creates a new, empty cart and returns its ID."""
        with self.consumer_lock:
            logging.info("Consumer entered new_cart method")
            self.cart_id += 1
            self.carts[self.cart_id] = {}
            logging.info("Consumer registered new cart with id %d", self.cart_id)
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart if it's available.

        This entire method is protected by a single consumer lock, serializing all
        'add' operations across all consumers.
        """
        with self.consumer_lock:
            logging.info("Consumer with card id %d entered add_to_cart method", cart_id)
            # Check if the product is available from any producer.
            if product in self.products_dict:
                # Take the product from the first available producer.
                producer_id = self.products_dict[product].pop(0)
                if product in self.carts[cart_id]:
                    self.carts[cart_id][product].append(producer_id)
                else:
                    self.carts[cart_id][product] = [producer_id]
                
                if len(self.products_dict[product]) == 0:
                    del self.products_dict[product]
                logging.info(f"Consumer with card id {cart_id} added product {product} to cart")
                return True
        logging.info(f"Consumer with card id {cart_id} failed to add product {product} to cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace inventory."""
        with self.consumer_lock:
            logging.info("Consumer with card id %d entered remove_from_cart method", cart_id)
            # Take a producer ID from the cart's record for this product.
            given_id = self.carts[cart_id][product].pop(0)
            if len(self.carts[cart_id][product]) == 0:
                del self.carts[cart_id][product]

            # Return the product to the global inventory.
            if product not in self.products_dict:
                self.products_dict[product] = [given_id]
            else:
                self.products_dict[product].append(given_id)
            logging.info(f"Consumer with card id {cart_id} removed product {product} from cart")

    def place_order(self, cart_id):
        """Finalizes an order, reducing producer capacities and returning the product list."""
        with self.consumer_lock:
            logging.info("Consumer with card id %d entered place_order method", cart_id)
            products = []
            # Invariant: For every product in the cart, decrement the original producer's item count.
            for product in self.carts[cart_id]:
                for given_id in self.carts[cart_id][product]:
                    self.size_per_producer[given_id] -= 1
                    products.append(product)
            logging.info("Consumer with card id %d placed order", cart_id)
        return products

    def get_consumer_lock(self):
        """Returns the single lock used for all consumer operations."""
        logging.info("A consumer entered get_consumer_lock method")
        return self.consumer_lock


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        """Tests that producer IDs are created sequentially."""
        self.assertEqual(self.marketplace.register_producer(), 0)

    def test_true_publish(self):
        """Tests a successful product publication."""
        producer_id = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(producer_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 1)

    def test_false_publish(self):
        """Tests that publishing fails when the producer's queue is full."""
        producer_id = self.marketplace.register_producer()
        for _ in range(5):
             self.marketplace.publish(producer_id, "id1")
        self.assertFalse(self.marketplace.publish(producer_id, "id2"))

    def test_new_cart(self):
        """Tests that cart IDs are created sequentially."""
        self.assertEqual(self.marketplace.new_cart(), 0)

    def test_true_add_to_cart(self):
        """Tests a successful addition to a cart."""
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        cart_id = self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))

    def test_false_add_to_cart(self):
        """Tests that adding a non-existent product fails."""
        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, "id1"))

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")
        self.marketplace.remove_from_cart(cart_id, "id1")
        self.assertEqual(len(self.marketplace.carts[cart_id]), 0)

    def test_place_order(self):
        """Tests the final placement of an order."""
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")
        products = self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.size_per_producer[producer_id], -1) # Likely a bug in the test or logic
        self.assertEqual(["id1"], products)

    def test_get_consumer_lock(self):
        """Tests that the correct lock object is returned."""
        self.assertEqual(self.marketplace.consumer_lock, self.marketplace.get_consumer_lock())


# This is a second, redundant definition of the Producer class.
class Producer(Thread):
    """Represents a producer thread that publishes products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """The main execution logic for the producer thread."""
        producer_id = self.marketplace.register_producer()
        # Invariant: The producer runs in an infinite loop.
        while True:
            for product in self.products:
                # Persistently try to publish the specified quantity.
                for _ in range(0, product[1]):
                    while self.marketplace.publish(producer_id, product[0]) is not True:
                        sleep(self.republish_wait_time)
                    sleep(product[2])
