"""
This module simulates a marketplace with producers, consumers, and products.

It defines a multi-threaded producer-consumer model where Producer threads add
products to a central Marketplace, and Consumer threads add/remove products
from carts and place orders. The file also includes product definitions and a
unit test for the marketplace.
"""
import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer that performs a series of actions on the marketplace.
    
    Each consumer is a thread that processes a list of "carts", where each cart is
    a sequence of actions like adding or removing products.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of action dicts.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying to add a product.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.activities = {}

    def extract_activities(self, activity, consumer_id):
        """Extracts and stores the details of a single consumer activity."""
        self.activities[consumer_id] = []

        activity_type = activity.get("type")
        activity_product = activity.get("product")
        activity_quantity = activity.get("quantity")

        self.activities[consumer_id] = (activity_type, activity_product, activity_quantity)

        return self.activities[consumer_id]

    def get_info(self, consumer_id, cart):
        """
        Processes all the actions (add/remove) for a given cart.

        Args:
            consumer_id (int): The unique ID for the current cart.
            cart (list): A list of action dictionaries to perform.
        """
        for activity in cart:
            activity_type, product, quantity = self.extract_activities(activity, consumer_id)

            if activity_type == "add":
                # Attempt to add the specified quantity of a product to the cart.
                product_counter = 0
                while product_counter < quantity:
                    add_ok = self.marketplace.add_to_cart(consumer_id, product)
                    if add_ok:
                        product_counter += 1
                    else:
                        # If adding fails (e.g., product not in stock), wait and retry.
                        time.sleep(self.retry_wait_time)

            elif activity_type == "remove":
                # Remove the specified quantity of a product from the cart.
                product_counter = 0
                while product_counter < quantity:
                    self.marketplace.remove_from_cart(consumer_id, product)
                    product_counter += 1

    def run(self):
        """
        The main execution loop for the consumer thread.

        It processes each cart sequentially, performs the actions, places the order,
        and prints the items bought.
        """
        for cart in self.carts:
            consumer_id = self.marketplace.new_cart()
            self.get_info(consumer_id, cart)

            for product in self.marketplace.place_order(consumer_id):
                print(f"{self.name} bought {product}")

# Note: The following classes appear to be concatenated from different files.
from threading import Lock
import unittest

# from tema.consumer import Consumer
# from tema.producer import Producer
# from tema.product import Coffee, Tea


class Marketplace:
    """
    The central marketplace that manages producers, products, and carts.
    
    This class is intended to be the synchronized hub for all interactions, but
    contains several thread-safety issues.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can publish.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.carts_counter = -1
        self.producers_counter = -1
        self.carts = []
        self.products = [] # Stores product counts per producer.
        self.in_stock_products = [] # A flat list of all available products.
        self.in_stock_products_producers = []
        self.carts_lock = Lock()
        self.lock_publish = Lock() # Note: This lock is declared but never used.
        self.producers_lock = Lock()
        self.in_stock_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer and returns a unique producer ID.
        @note This method is thread-safe.
        """
        with self.producers_lock:
            self.producers_counter += 1
            self.products.append(0)
            return self.producers_counter

    def publish(self, producer_id, product):
        """
        Adds a product from a producer to the marketplace stock.
        
        @note This method is NOT thread-safe. It reads and modifies shared lists
              (`in_stock_products`, `products`) without acquiring a lock, which can
              lead to race conditions if multiple producers publish concurrently.
        """
        number_of_products = self.products[producer_id]
        if number_of_products < self.queue_size_per_producer:
            self.in_stock_products.append(product)
            self.products[producer_id] += 1
            self.in_stock_products_producers.append((producer_id, product))
            return True
        else:
            return False

    def new_cart(self):
        """
        Creates a new empty cart and returns a unique cart ID.
        @note This method is thread-safe.
        """
        with self.carts_lock:
            self.carts_counter += 1
            self.carts.append([])
            return self.carts_counter

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from the marketplace stock to a consumer's cart.
        @note This method is thread-safe.
        """
        with self.in_stock_lock:
            if product in self.in_stock_products:
                self.carts[cart_id].append(product)
                self.in_stock_products.remove(product)
                return True
            else:
                return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the marketplace stock.
        
        @note This method is NOT thread-safe. It modifies shared lists (`carts`,
              `in_stock_products`) without acquiring a lock.
        """
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            self.in_stock_products.append(product)

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        
        @note This method is NOT thread-safe and contains logical errors. It iterates
              over a list while attempting to modify a related list, and the
              lookup `prod in self.in_stock_products_producers` is incorrect as it
              compares a Product object to a tuple `(id, Product)`.
        """
        for prod in self.carts[cart_id]:
            if prod in self.in_stock_products_producers:
                with self.producers_lock:
                    element = self.in_stock_products_producers[prod]
                    self.products[element] -= 1
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    """A suite of unit tests for the Marketplace class."""

    def setUp(self):
        """Sets up the test environment before each test."""
        self.marketplace = Marketplace(24)
        self.tea1 = Tea("Mint", 15, "Green")
        self.tea2 = Tea("Earl grey", 30, "Black")
        self.coffee = Coffee("Lavazza", 14, "2.23", "MEDIUM")
        self.producer = Producer([[self.tea1, 8, 0.11],
                                  [self.tea2, 5, 0.7],
                                  [self.coffee, 1, 0.13]],
                                 self.marketplace,
                                 0.35)
        self.consumer = Consumer([[{"type": "add", "product": self.coffee, "quantity": 1},
                                   {"type": "add", "product": self.tea1, "quantity": 4},
                                   {"type": "add", "product": self.tea2, "quantity": 2},
                                   {"type": "remove", "product": self.tea2, "quantity": 1}
                                   ]],
                                 self.marketplace,
                                 0.31)
        self.cart_id = self.marketplace.new_cart()

    def test_register_function(self):
        """Tests the producer registration functionality."""
        self.assertEqual(str(self.marketplace.register_producer()), "0")

# from threading import Thread
# import time

class Producer(Thread):
    """Represents a producer that publishes products to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.
        
        Args:
            products (list): A list of products to produce, where each item is
                             another list: [Product, quantity, wait_time].
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait if publishing fails.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution loop for the producer.

        Continuously attempts to publish its assigned products to the marketplace.
        
        @note This method contains a significant bug: `register_producer()` is called
              repeatedly inside the loop, creating a new producer ID for every single
              item published instead of registering only once.
        """
        while True:
            for prod in self.products:
                product_counter = 0
                product = prod[0]
                quantity = prod[1]
                wait_time = prod[2]

                while product_counter < quantity:
                    # BUG: This registers a new producer on every attempt.
                    if self.marketplace.publish(self.marketplace.register_producer(), product):
                        product_counter += 1
                        time.sleep(wait_time)
                    else:
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A dataclass representing a generic product."""
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
