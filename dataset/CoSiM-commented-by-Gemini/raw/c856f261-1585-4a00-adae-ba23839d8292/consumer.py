
"""
@file consumer.py (and others)
@brief A multi-threaded producer-consumer simulation using a central marketplace.
@details This module defines a complete, concurrent system simulating an e-commerce
platform. It features a central marketplace that manages inventories from multiple
producers, and consumers that process shopping lists.

@warning CRITICAL CONCURRENCY FLAW: This implementation has a high risk of deadlocks.
The `Marketplace.add_to_cart` and `Marketplace.remove_from_cart` methods acquire
the `lock` and `cart_lock` in inconsistent orders, creating a classic deadlock
scenario that can halt the entire system.

NOTE: This file appears to be a concatenation of multiple Python files.
The documentation will proceed by addressing each class in sequence.
"""

from threading import Thread, Lock
from time import sleep, gmtime
import unittest
import logging
import logging.handlers
from dataclasses import dataclass

# --- Consumer Logic ---

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread, processing a list of shopping carts by
    executing add/remove commands and then placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.
        
        :param carts: A list of shopping lists for the consumer to process.
        :param marketplace: The shared Marketplace instance.
        :param retry_wait_time: Time to wait before retrying a failed operation.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.curr_cart = []
        self.cart_id = -1
        self.kwargs = kwargs
        Thread.__init__(self, **kwargs)


    def print_cart(self, cart):
        """
        Prints the contents of a purchased cart in a thread-safe manner.
        
        :param cart: A dictionary representing the cart's contents.
        """
        # Obtains a dedicated lock from the marketplace to prevent interleaved printing.
        lock = self.marketplace.get_print_lock()
        with lock:
            for prod in cart:
                for _ in range(len(cart[prod])):
                    print(self.kwargs['name'] + " bought " + str(prod))


    def add_to_cart(self, product, cart_id):
        """
        Helper method to add a product to a cart, with a busy-wait retry loop.
        
        :param product: The product to add.
        :param cart_id: The ID of the target cart.
        """
        # Block Logic: This implements a busy-wait pattern. The thread will
        # continuously try to add the product, sleeping for a short interval
        # if the marketplace cannot fulfill the request immediately.
        res = False
        while res is False:
            res = self.marketplace.add_to_cart(cart_id, product)
            if res is False:
                sleep(self.wait_time)

    def run(self):
        """Main execution loop for the consumer thread."""
        for cart in self.carts:
            # Each journey gets a new cart ID from the marketplace.
            cart_id = self.marketplace.new_cart()
            for cmd in cart:
                cmd_type = cmd['type']
                product = cmd['product']
                quantity = cmd['quantity']
                for _ in range(0, quantity):
                    if cmd_type == "add":
                        self.add_to_cart(product, cart_id)
                    else:
                        self.marketplace.remove_from_cart(cart_id, product)
            
            # Place the order and print the results.
            final_cart = self.marketplace.place_order(cart_id)
            self.print_cart(final_cart)

# --- Marketplace Logic (appears to be from marketplace.py) ---

# --- Logging Setup ---
LOGGER = logging.getLogger("marketlogger")
LOGGER.setLevel(logging.INFO)
HANDLER = logging.handlers.RotatingFileHandler(
    "marketplace.log", maxBytes=20000, backupCount=5)
FORMATTER = logging.Formatter("%(asctime)s;%(message)s")
HANDLER.setFormatter(FORMATTER)
logging.Formatter.converter = time.gmtime
LOGGER.addHandler(HANDLER)


class Marketplace:
    """
    A thread-safe marketplace that coordinates producers and consumers.

    @warning This class is prone to deadlocks due to inconsistent lock acquisition
    order between its `add_to_cart` and `remove_from_cart` methods.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        
        :param queue_size_per_producer: Max items a producer can publish.
        """
        self.queue_size = queue_size_per_producer
        self.producer_id_indexer = 0
        self.cart_id_indexer = 0
        self.producers_dict = {}  # Stores current item count for each producer.
        self.all_products = {}    # Maps product names to a list of producer IDs that have them.
        self.lock = Lock()        # General lock for producer and product data.
        self.cart_lock = Lock()   # Lock specifically for cart data.
        self.print_lock = Lock()  # Lock to serialize print statements.
        self.carts = {}           # Stores active shopping carts.

    def get_print_lock(self):
        """Returns the lock used for synchronizing print statements."""
        LOGGER.info("print_lock returned")
        return self.print_lock

    def register_producer(self):
        """Registers a new producer, assigning a unique ID."""
        LOGGER.info(" (register) started")
        with self.lock:
            new_id = self.producer_id_indexer
            self.producer_id_indexer += 1
            self.producers_dict[new_id] = 0
            LOGGER.info("(register) id %d", new_id)
            return new_id

    def publish(self, producer_id, product):
        """Allows a producer to list a product."""
        LOGGER.info("(publish) %d, %s", producer_id, str(product))
        with self.lock:
            # Check if the producer's queue is full.
            if self.producers_dict[producer_id] == self.queue_size:
                LOGGER.info(" (publish) producer has too many items")
                return False

            self.producers_dict[producer_id] += 1

            # Add the product to the central inventory.
            if product not in self.all_products:
                self.all_products[product] = [producer_id]
            else:
                self.all_products[product].append(producer_id)
            
            LOGGER.info(" (publish) the product was published")
            return True

    def new_cart(self):
        """Creates a new, empty shopping cart."""
        LOGGER.info("(new cart) started")
        with self.cart_lock:
            new_id = self.cart_id_indexer
            self.cart_id_indexer += 1
            self.carts[new_id] = {}
            LOGGER.info(" (new_cart) created cart with id %d", new_id)
            return new_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart from the marketplace inventory.
        
        @warning DEADLOCK RISK: This method acquires `self.lock` then `self.cart_lock`.
        This is the opposite order of `remove_from_cart`, creating a deadlock hazard.
        """
        LOGGER.info(" (add_to_cart) params %d, %s",
                    cart_id, str(product))
        self.lock.acquire()
        try:
            if product not in self.all_products or len(self.all_products[product]) == 0:
                LOGGER.info(
                    "(add_to_cart) no product %s is published", str(product))
                return False
            
            # Take a product from a producer's inventory.
            producer_id = self.all_products[product].pop(0)

            with self.cart_lock:
                # Add the product and its source producer to the cart.
                if product not in self.carts[cart_id]:
                    self.carts[cart_id][product] = [producer_id]
                else:
                    self.carts[cart_id][product].append(producer_id)
        finally:
            self.lock.release()
        
        LOGGER.info(" (add_to_cart) product %s was added", str(product))
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the inventory.
        
        @warning DEADLOCK RISK: This method acquires `self.cart_lock` then `self.lock`.
        This is the opposite order of `add_to_cart`, creating a deadlock hazard.
        """
        LOGGER.info('(remove from cart) params %s %d',
                    str(product), cart_id)
        # Pre-condition: Check if the product is actually in the cart.
        if product in self.carts[cart_id] and len(self.carts[cart_id][product]) != 0:
            with self.cart_lock:
                with self.lock:
                    # Retrieve the source producer and return the product to the central inventory.
                    producer_id = self.carts[cart_id][product].pop(0)
                    self.all_products[product].append(producer_id)
        
        LOGGER.info(" (remove_from_cart) finished %s was removed %d",
                    str(product), cart_id)

    def place_order(self, cart_id):
        """Finalizes an order and decrements the producer's published item count."""
        LOGGER.info(" (place order) param %d", cart_id)
        with self.lock:
            products = self.carts[cart_id]
            # For each product in the finalized cart, decrement the inventory
            # count for the original producer.
            for _, ids in products.items():
                for producer_id in ids:
                    self.producers_dict[producer_id] -= 1
        
        LOGGER.info(" (place order) %d was placed", cart_id)
        return self.carts[cart_id]


# --- Unit Test Suite ---
QUEUE_SIZE = 3

class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    Note: These tests run sequentially and do not test the concurrency issues.
    """

    def publish_products(self, prod_id, number_of_products):
        """Helper to publish multiple products for testing."""
        for i in range(number_of_products):
            self.marketplace.publish(prod_id, str("prod_" + str(i)))

    def setUp(self):
        """Sets up a test environment before each test case."""
        self.marketplace = Marketplace(QUEUE_SIZE)

    def test_register_producer(self):
        """Tests sequential producer registration."""
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)

    def test_publish(self):
        """Tests the product publishing limit."""
        prod_id = self.marketplace.register_producer()
        for i in range(3):
            self.assertTrue(self.marketplace.publish(prod_id, str("prod_" + str(i))))
        self.assertFalse(self.marketplace.publish(id, "some_product")) # Queue is full.

    # ... other tests ...


# --- Producer Logic (appears to be from producer.py) ---
PRODUCT_POS = 0
NUMBER_OF_PRODUCTS_POS = 1
WAITING_TIME_POS = 2

class Producer(Thread):
    """
    Represents a producer that supplies products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.
        :param products: A list of product tuples to publish (product, quantity, wait_time).
        :param marketplace: The shared Marketplace instance.
        :param republish_wait_time: Time to wait before retrying a failed publish.
        """
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)


    def run(self):
        """Main execution loop for the producer thread."""
        prod_id = self.marketplace.register_producer()
        while True:
            for prod in self.products:
                product = prod[PRODUCT_POS]
                no_prods = prod[NUMBER_OF_PRODUCTS_POS]
                pause_time = prod[WAITING_TIME_POS]
                for _ in range(0, no_prods):
                    # Implements a busy-wait retry loop for publishing.
                    res = False
                    while res is False:
                        res = self.marketplace.publish(prod_id, product)
                        if res is False:
                            sleep(self.wait_time)
                    sleep(pause_time)


# --- Product Definitions (appears to be from product.py) ---
@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee."""
    acidity: str
    roast_level: str
