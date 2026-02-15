"""
This module simulates a marketplace with producers, consumers, and products.

It defines a multi-threaded producer-consumer model where Producer threads add
products to a central Marketplace, and Consumer threads perform actions on
shopping carts. The file also includes a unit test suite. The product
definitions (e.g., Coffee, Tea) are assumed to be in a 'tema' module.
"""
import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer that performs a series of shopping actions.
    
    Each consumer is a thread that is initialized with a list of actions
    (e.g., add/remove products) and executes them against the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of actions.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed action.
            **kwargs: Keyword arguments for the `Thread` base class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # A single cart is created for this consumer instance.
        self.id_cart = self.marketplace.new_cart()

    def run(self):
        """The main loop for the consumer thread."""
        # The outer loop iterates through "carts", which are lists of actions.
        for cart_actions in self.carts:
            for action in cart_actions:
                # Perform the action (add/remove) for the specified quantity.
                for _ in range(action["quantity"]):
                    if action["type"] == "add":
                        # Continuously try to add the product until successful.
                        added_successfully = False
                        while not added_successfully:
                            added_successfully = self.marketplace.add_to_cart(self.id_cart, action["product"])
                            if not added_successfully:
                                sleep(self.retry_wait_time)
                    else: # "remove" action
                        self.marketplace.remove_from_cart(self.id_cart, action["product"])
                        # Note: The original code sleeps even after a remove operation.
                        sleep(self.retry_wait_time)
        
        # After all actions are done, place the order and print the final items.
        final_items = self.marketplace.place_order(self.id_cart)
        for order in final_items:
            with self.marketplace.print_lock:
                print(f"cons{self.id_cart} bought {order}")

# Note: The following classes appear to be concatenated from different files.
import unittest
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock

# from tema.consumer import Consumer
# from tema.producer import Producer
# from tema.product import Coffee, Tea


class Marketplace:
    """
    The central marketplace that manages all producers, products, and carts.

    This class is intended to be the synchronized hub for the simulation.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace, including logging.
        
        Args:
            queue_size_per_producer (int): Max products a producer can have listed.
        """
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(asctime)s;%(message)s")
        self.rotating_file_handler = RotatingFileHandler('marketplace.log', 'w')
        self.rotating_file_handler.setLevel(logging.INFO)
        self.rotating_file_handler.setFormatter(self.formatter)
        self.log.addHandler(self.rotating_file_handler)

        self.prods = {}         # Stores products per producer ID.
        self.cons = {}          # Stores products per consumer cart ID.
        self.no_prods = 0       # Producer ID counter.
        self.no_cons = 0        # Cart ID counter.
        self.producer_lock = Lock()
        self.publish_lock = Lock()
        self.cart_lock = Lock()
        self.add_cart_lock = Lock() # Note: Declared but never used.
        self.print_lock = Lock()    # For synchronized printing.
        self.available_prods = []   # Flat list of all products in stock.
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """Registers a new producer, returning a unique ID. Thread-safe."""
        self.log.info("Register product")
        with self.producer_lock:
            self.no_prods += 1
        self.prods[self.no_prods] = []
        self.log.info("Register product final")
        return self.no_prods

    def publish(self, producer_id, product):
        """
        Adds a product from a producer to the marketplace stock. Thread-safe.
        """
        self.log.info("Publish")
        with self.publish_lock:
            if len(self.prods[producer_id]) < self.queue_size_per_producer:
                self.prods[producer_id].append(product)
                self.available_prods.append(product)
                self.log.info("Publish final")
                return True
            return False

    def new_cart(self):
        """Creates a new cart, returning a unique ID. Thread-safe."""
        self.log.info("New Cart")
        with self.cart_lock:
            self.no_cons += 1
        self.cons[self.no_cons] = []
        self.log.info("New Cart final")
        return self.no_cons

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from stock to a cart. Thread-safe, but potentially slow
        due to iterating through all producers' products inside a lock.
        """
        self.log.info("Add to Cart")
        with self.cart_lock:
            if product in self.available_prods:
                self.available_prods.remove(product)
                for i, products in self.prods.items():
                    if product in products:
                        self.cons[cart_id].append(product)
                        self.prods[i].remove(product)
                        self.log.info("Add to Cart final")
                        return True
            return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to stock.

        @note This method is NOT thread-safe. It modifies shared state
              (`self.cons`, `self.available_prods`) without acquiring any locks.
        """
        self.log.info("Remove from cart")
        if product in self.cons[cart_id]:
            self.cons[cart_id].remove(product)
            self.available_prods.append(product)
        self.log.info("Remove from cart final")

    def place_order(self, cart_id):
        """Returns the items in the cart, simulating an order placement."""
        self.log.info("Place order")
        return self.cons[cart_id]

class TestMarketplace(unittest.TestCase):
    """A suite of unit tests for the Marketplace class."""
    
    def setUp(self):
        """Sets up the test environment before each test."""
        self.marketplace = Marketplace(3)
    def test_register(self):
        """Tests producer registration."""
        self.assertEqual(self.marketplace.register_producer(), 1)
    def test_publish(self):
        """Tests product publishing."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        self.assertEqual(self.marketplace.available_prods,
                         ["Tea(name='Wild Cherry', price=5, type='Black')"])
    def test_new_cart(self):
        """Tests cart creation."""
        self.assertEqual(self.marketplace.new_cart(), 1)
    def test_add_to_cart(self):
        """Tests adding a product to a cart."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        self.marketplace.add_to_cart(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        print(self.marketplace.cons)
    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        self.marketplace.add_to_cart(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        print(self.marketplace.cons)
        self.marketplace.remove_from_cart(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        print(self.marketplace.cons)
    def test_place_order(self):
        """Tests the order placement method."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        self.marketplace.add_to_cart(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        print(self.marketplace.cons)
        print(self.marketplace.place_order(1))


# from threading import Thread
# from time import sleep

class Producer(Thread):
    """Represents a producer that publishes products to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer, registering it with the marketplace once.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer.
        
        Continuously attempts to publish its assigned products.
        
        @note The sleep logic is flawed. It sleeps the `republish_wait_time`
              on every iteration, regardless of whether the publish was
              successful, slowing down production unnecessarily.
        """
        while True:
            for prod in self.products:
                for _ in range(prod[1]):
                    ret = self.marketplace.publish(self.id_producer, prod[0])
                    if ret:
                        sleep(prod[2])
                    # BUG: This sleep should be in an 'else' block.
                    sleep(self.republish_wait_time)
