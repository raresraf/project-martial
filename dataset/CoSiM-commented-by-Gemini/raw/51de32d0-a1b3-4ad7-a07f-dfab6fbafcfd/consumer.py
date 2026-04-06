"""
This module simulates a multi-threaded producer-consumer model for an e-commerce
marketplace.

It defines `Consumer`, `Producer`, and `Marketplace` classes that interact
concurrently. The Marketplace uses separate locks for different operations to
manage thread safety and a dictionary-based inventory system.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer thread that simulates a customer's shopping process.

    Each consumer is given a list of shopping carts (which are lists of actions)
    and executes them against the marketplace.
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

    def safe_add_to_cart(self, cart_id, product):
        """
        Helper method to add an item to the cart, retrying if unavailable.
        This simulates a user waiting for a product to be restocked.
        """
        while not self.marketplace.add_to_cart(cart_id, product):
            sleep(self.retry_wait_time)

    def run(self):
        """The main execution logic for the consumer thread."""
        for cart in self.carts:
            my_cart = self.marketplace.new_cart()

            for crt_op in cart:
                if crt_op["type"] == "add":
                    for _ in range(crt_op["quantity"]):
                        self.safe_add_to_cart(my_cart, crt_op["product"])
                elif crt_op["type"] == "remove":
                    for _ in range(crt_op["quantity"]):
                        self.marketplace.remove_from_cart(my_cart, crt_op["product"])
                else:
                    print("[Error] No such operation")

            ordered_prods = self.marketplace.place_order(my_cart)
            # Use a shared lock to prevent interleaved print statements from
            # different consumer threads.
            for prod in ordered_prods:
                self.marketplace.print_lock.acquire()
                print(self.name, "bought", prod)
                self.marketplace.print_lock.release()

import time
import unittest
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler

# --- Global Logger Setup ---
# Configures a rotating file logger to record all marketplace activities.
logging.basicConfig(
    handlers=[RotatingFileHandler('./marketplace.log', maxBytes=2000, backupCount=5)],
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s"
)
logging.Formatter.converter = time.gmtime

class Marketplace:
    """
    The central, thread-safe hub for coordinating producers and consumers.

    This implementation uses a dictionary (`producers`) to store each producer's
    inventory as a list of products. It employs separate locks for different
    operations to reduce contention between threads.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        
        Args:
            queue_size_per_producer (int): The max number of products a single
                                           producer can have published.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.next_cart_id = 0
        self.next_producer_id = 0
        self.producers = {} # {producer_id: [product1, product2, ...]}
        self.carts = {}     # {cart_id: [(product, producer_id), ...]}
        self.producers_stock = {} # {producer_id: count}

        # --- Synchronization Primitives ---
        self.producer_reg_lock = Lock()
        self.new_cart_lock = Lock()
        self.products_lock = Lock()
        self.print_lock = Lock()

    def register_producer(self):
        """Atomically registers a new producer and returns a unique ID."""
        logging.info('register_producer was called')
        self.producer_reg_lock.acquire()
        new_producer_id = str(self.next_producer_id)
        self.next_producer_id = self.next_producer_id + 1
        self.producer_reg_lock.release()

        self.producers[new_producer_id] = []
        self.producers_stock[new_producer_id] = 0

        logging.info("register_producer done: id = %s", new_producer_id)
        return new_producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product if they have inventory capacity.

        Returns:
            bool: True if successful, False if the producer's queue is full.
        """
        logging.info("publish was called: producer_id = %s; 
                    prod = %s", producer_id, str(product))

        # Pre-condition: Check if producer is below their publication limit.
        if self.producers_stock[producer_id] >= self.queue_size_per_producer:
            logging.info("publish done")
            return False

        self.producers[producer_id].append(product)
        self.producers_stock[producer_id] = self.producers_stock[producer_id] + 1

        logging.info("publish done")
        return True

    def new_cart(self):
        """Atomically creates a new, empty cart and returns its ID."""
        logging.info("new_cart was called")
        self.new_cart_lock.acquire()
        my_id = self.next_cart_id
        self.next_cart_id = self.next_cart_id + 1
        self.new_cart_lock.release()

        self.carts[my_id] = []
        logging.info("new_cart done: id = %s", str(my_id))
        return my_id

    def add_to_cart(self, cart_id, product):
        """
        Searches all producer inventories for a product and adds it to the cart.
        
        This method acquires a global product lock, finds an available product,
        removes it from the producer's list, and adds it to the consumer's cart.
        """
        logging.info("add_to_cart was called: cart_id = %s; 
                    prod = %s", str(cart_id), str(product))
        if cart_id not in self.carts:
            return False

        self.products_lock.acquire()
        # Invariant: Search all producers for the requested product.
        for producer in self.producers:
            for crt_prod in self.producers[producer]:
                if crt_prod == product:
                    # Add to cart, tracking original producer.
                    self.carts[cart_id].append((product, producer))
                    # Remove from producer's inventory.
                    self.producers[producer].remove(crt_prod)
                    self.products_lock.release()
                    logging.info("add_to_cart done: True")
                    return True

        self.products_lock.release()
        logging.info("add_to_cart done: False")
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the original producer."""
        logging.info("remove_from_cart was called: cart_id = %s; 
                        product = %s", str(cart_id), str(product))
        if cart_id not in self.carts:
            logging.info("remove_from_cart done")
            return

        for (prod, producer) in self.carts[cart_id]:
            if prod == product:
                # Add the product back to the original producer's inventory list.
                self.products_lock.acquire()
                self.producers[producer].append(prod)
                self.products_lock.release()

                # Remove from the consumer's cart.
                self.carts[cart_id].remove((prod, producer))
                break
        logging.info("remove_from_cart done")

    def place_order(self, cart_id):
        """
        Finalizes an order, freeing up producer slots for the items purchased.
        """
        logging.info("place_order called")
        result = []

        if cart_id not in self.carts:
            logging.info("place_order done: %s", result)
            return result

        # Invariant: Process all items in the cart.
        for (prod, producer) in self.carts[cart_id]:
            result.append(prod)
            # Atomically decrement the producer's stock count, freeing up a slot.
            self.products_lock.acquire()
            self.producers_stock[producer] = self.producers_stock[producer] - 1
            self.products_lock.release()

        logging.info("place_order done: %s", result)
        return result

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Prepares a new Marketplace instance for each test."""
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        """Tests sequential producer ID generation."""
        for i in range(1000):
            self.assertEqual(self.marketplace.register_producer(), str(i))

    def test_publish(self):
        """Tests product publishing and queue limits."""
        self.marketplace.register_producer()
        for i in range(2):
            self.assertEqual(self.marketplace.publish("0", str(i)), True)
        for i in range(2):
            self.assertEqual(self.marketplace.publish("0", str(i)), False)

        self.marketplace.register_producer()
        for i in range(2):
            self.assertEqual(self.marketplace.publish("1", str(i)), True)
        for i in range(2):
            self.assertEqual(self.marketplace.publish("1", str(i)), False)

    def test_new_cart(self):
        """Tests sequential cart ID generation."""
        for i in range(1000):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_add_to_cart(self):
        """Tests adding available and unavailable products to a cart."""
        for i in range(10):
            self.assertEqual(self.marketplace.new_cart(), i)
            self.assertEqual(self.marketplace.register_producer(), str(i))
        for i in range(10):
            self.assertEqual(self.marketplace.publish(str(i), str(i + 1000)), True)
        for i in range(10):
            self.assertEqual(self.marketplace.add_to_cart(i, str(i + 1000)), True)
        for i in range(10):
            self.assertEqual(self.marketplace.add_to_cart(i, str(i + 1000)), False)

    def test_remove_from_cart(self):
        """Tests that removing a product returns it to the correct producer."""
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.register_producer(), "0")
        self.assertEqual(self.marketplace.publish("0", "00"), True)
        self.assertEqual(self.marketplace.publish("0", "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), False)
        self.marketplace.remove_from_cart(0, "00")
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)

    def test_place_order(self):
        """Tests that an order contains the correct products and stock is updated."""
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.register_producer(), "0")
        self.assertEqual(self.marketplace.publish("0", "00"), True)
        self.assertEqual(self.marketplace.publish("0", "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.marketplace.remove_from_cart(0, "01")
        self.marketplace.remove_from_cart(0, "02") # This should have no effect
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), False)
        self.marketplace.remove_from_cart(0, "00")
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.place_order(0), ["01", "00"])


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that continuously supplies products.
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

    def safe_publish(self, product, producer_id):
        """Helper method to publish a product, retrying if the queue is full."""
        while not self.marketplace.publish(producer_id, product):
            sleep(self.republish_wait_time)

    def run(self):
        """The main execution logic for the producer thread."""
        my_id = self.marketplace.register_producer()

        # Invariant: This thread will run forever, simulating continuous production.
        while True:
            for (id_prod, quantity_prod, wait_time_prod) in self.products:
                sleep(wait_time_prod)

                for _ in range(quantity_prod):
                    self.safe_publish(id_prod, my_id)


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
