"""
This module simulates a multi-threaded producer-consumer model for an e-commerce
marketplace.

It defines `Consumer`, `Producer`, and `Marketplace` classes that interact
concurrently. The Marketplace uses a central dictionary to manage product
inventory from multiple producers, with locks to ensure thread-safe operations.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that processes a list of shopping carts.

    Each consumer simulates a customer's journey for each cart they are assigned,
    from creation and item management to placing the final order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts", where each cart is a list of
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
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for ops in cart:
                # Block Logic: Handle 'add' operations.
                if ops['type'] == "add":
                    for _ in range(0, ops['quantity']):
                        # Pre-condition: Retry adding the product until successful.
                        # This simulates waiting for a product to be restocked.
                        while self.marketplace.add_to_cart(cart_id, ops['product']) is not True:
                            sleep(self.retry_wait_time)
                # Block Logic: Handle 'remove' operations.
                else:
                    for _ in range(0, ops['quantity']):
                        self.marketplace.remove_from_cart(cart_id, ops['product'])
            
            # Finalize the order for the current cart.
            products = self.marketplace.place_order(cart_id)

            # Acquire a shared lock to prevent interleaved print statements from
            # different consumer threads.
            lock = self.marketplace.get_consumer_lock()

            lock.acquire()
            for product in products:
                print(self.kwargs['name'] + " bought " + str(product))
            lock.release()


import logging
from logging.handlers import RotatingFileHandler
import time
from threading import Lock
import unittest

# --- Global Logger Setup ---
# Configures a rotating file logger to record all marketplace activities.
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO,
                    handlers=[RotatingFileHandler('marketplace.log',
                                                  maxBytes=20000, backupCount=10)])
logging.Formatter.converter = time.gmtime

class Marketplace:
    """
    The central, thread-safe hub for coordinating producers and consumers.
    
    This implementation uses a central dictionary (`products_dict`) to track all
    available products, mapping each product to a list of producer IDs that have
    it in stock. This allows for efficient lookups and management of inventory from
    multiple sources.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        
        Args:
            queue_size_per_producer (int): The max number of products a single
                                           producer can have published at once.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.producer_id = -1
        self.cart_id = -1
        self.size_per_producer = {}  # Tracks items published per producer.
        self.carts = {}  # {cart_id: {product: [producer_id_1, producer_id_2, ...]}}
        self.products_dict = {}  # {product: [producer_id_1, producer_id_2, ...]}

    def register_producer(self):
        """Atomically registers a new producer and returns a unique ID."""
        self.producer_lock.acquire()
        logging.info("New producer entered register_producer method")
        self.producer_id += 1
        self.size_per_producer[self.producer_id] = 0
        self.producer_lock.release()
        logging.info("New producer registered with id %d", self.producer_id)
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product, making it available in the central product dictionary.

        Returns:
            bool: True if successful, False if the producer's queue is full.
        """
        logging.info("Producer with id %d entered publish method", producer_id)

        self.producer_lock.acquire()
        
        # Pre-condition: Check if the producer has available publication slots.
        if self.size_per_producer[producer_id] == self.queue_size_per_producer:
            logging.info(f"Producer with id {producer_id} failed to publish product {product}")
            self.producer_lock.release()
            return False

        # Add the product to the central inventory, associated with the producer's ID.
        if product not in self.products_dict:
            self.products_dict[product] = [producer_id]
        else:
            self.products_dict[product].append(producer_id)

        self.size_per_producer[producer_id] += 1
        logging.info(f"Producer with id {producer_id} published product {product}")
        self.producer_lock.release()
        return True

    def new_cart(self):
        """Atomically creates a new, empty cart and returns its ID."""
        self.consumer_lock.acquire()
        logging.info("Consumer entered new_cart method")
        self.cart_id += 1
        self.carts[self.cart_id] = {}
        logging.info("Consumer registered new cart with id %d", self.cart_id)
        self.consumer_lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by claiming it from an available producer.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered add_to_cart method", cart_id)
        if product in self.products_dict:
            # Claim the product from the first available producer.
            producer_id = self.products_dict[product].pop(0)
            # Add the product to the cart, tracking its origin producer.
            if product in self.carts[cart_id]:
                self.carts[cart_id][product].append(producer_id)
            else:
                self.carts[cart_id][product] = [producer_id]
            
            # If that was the last unit of this product, remove the entry.
            if len(self.products_dict[product]) == 0:
                del self.products_dict[product]

            logging.info(f"Consumer with card id {cart_id} added product {product} to cart")
            self.consumer_lock.release()
            return True
        logging.info(f"Consumer with card id {cart_id} failed to add product {product} to cart")
        self.consumer_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the available inventory."""
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered remove_from_cart method", cart_id)

        # Remove one instance of the product and get its original producer ID.
        given_id = self.carts[cart_id][product].pop(0)
        if len(self.carts[cart_id][product]) == 0:
            del self.carts[cart_id][product]

        # Return the product to the central product dictionary.
        if product not in self.products_dict:
            self.products_dict[product] = [given_id]
        else:
            self.products_dict[product].append(given_id)
        logging.info(f"Consumer with card id {cart_id} removed product {product} from cart")
        self.consumer_lock.release()

    def place_order(self, cart_id):
        """
        Finalizes an order, freeing up producer slots for the items purchased.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered place_order method", cart_id)
        
        products = []
        # Invariant: Process all products in the finalized cart.
        for product in self.carts[cart_id]:
            for given_id in self.carts[cart_id][product]:
                # Decrement the producer's published count, freeing up a slot.
                self.size_per_producer[given_id] -= 1
                products.append(product)
        logging.info("Consumer with card id %d placed order", cart_id)
        self.consumer_lock.release()
        return products

    def get_consumer_lock(self):
        """Returns the shared consumer lock for external synchronization (e.g., printing)."""
        logging.info("A consumer entered get_consumer_lock method")
        return self.consumer_lock


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Prepares a new Marketplace instance for each test."""
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        """Tests that producer registration returns sequential IDs."""
        self.assertEqual(self.marketplace.register_producer(), 0)

    def test_true_publish(self):
        """Tests that a product can be successfully published."""
        producer_id = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(producer_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 1)
        self.assertEqual(len(self.marketplace.products_dict["id1"]), 1)

    def test_false_publish(self):
        """Tests that publishing fails when a producer's queue is full."""
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.assertFalse(self.marketplace.publish(producer_id, "id1"))

    def test_new_cart(self):
        """Tests that cart creation returns sequential IDs."""
        self.assertEqual(self.marketplace.new_cart(), 0)

    def test_true_add_to_cart(self):
        """Tests that available products can be added to a cart."""
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        cart_id = self.marketplace.new_cart()

        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 0)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)
        self.assertEqual(len(self.marketplace.carts[cart_id]["id1"]), 1)

        self.marketplace.publish(producer_id, "id1")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))
        self.assertEqual(len(self.marketplace.carts[cart_id]["id1"]), 2)

    def test_false_add_to_cart(self):
        """Tests that an unavailable product cannot be added to a cart."""
        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, "id1"))

    def test_remove_from_cart(self):
        """Tests that a product is correctly removed from a cart and returned to inventory."""
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")


        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id2")

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")
        self.marketplace.add_to_cart(cart_id, "id2")
        self.assertEqual(len(self.marketplace.products_dict), 1)

        self.marketplace.remove_from_cart(cart_id, "id1")
        self.assertEqual(len(self.marketplace.products_dict), 2)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)
        self.assertFalse("id1" in self.marketplace.carts[cart_id])

        self.marketplace.remove_from_cart(cart_id, "id2")
        self.assertEqual(len(self.marketplace.products_dict), 2)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 0)
        self.assertFalse("id2" in self.marketplace.carts[cart_id])

    def test_place_order(self):
        """Tests that placing an order correctly frees producer slots and returns products."""
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.assertEqual(self.marketplace.size_per_producer[producer_id], 2)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")

        expected_products = ["id1"]
        products = self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.size_per_producer[producer_id], 1)
        self.assertEqual(expected_products, products)

    def test_get_consumer_lock(self):
        """Tests that the correct lock object is returned."""
        self.assertEqual(self.marketplace.consumer_lock, self.marketplace.get_consumer_lock())


from threading import Thread
from time import sleep


class Producer(Thread):
    """Represents a producer thread that continuously supplies products."""

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

    def run(self):
        """The main execution logic for the producer thread."""
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                for _ in range(0, product[1]):
                    # If publishing fails, wait and retry.
                    while self.marketplace.publish(producer_id, product[0]) is not True:
                        sleep(self.republish_wait_time)
                    # On success, wait for the production time.
                    sleep(product[2])
