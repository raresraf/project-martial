"""
Models a multi-producer, multi-consumer marketplace simulation with logging.

This module uses threading to simulate the behavior of producers who publish
products and consumers who purchase them. The Marketplace class acts as the
central, thread-safe intermediary for all transactions. The design is
object-oriented, with dedicated classes for `Cart` and different `Product` types,
and features a logging decorator for tracing marketplace operations.

Classes:
    Consumer: A thread simulating a customer's shopping session.
    Marketplace: A thread-safe hub for all producer/consumer interactions.
    Cart: A class representing a single shopping cart.
    Producer: A thread simulating a vendor supplying products.
    Product, Tea, Coffee: Dataclasses for product types.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace. 
    
    Each consumer processes a list of shopping carts, where each cart is a
    sequence of 'add' and 'remove' commands.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping lists for the consumer to process.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying to
                                     add a product if it's unavailable.
            **kwargs: Keyword arguments for the Thread constructor (e.g., name).
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name")

    def run(self):
        """The main logic for a consumer, processing a list of shopping commands."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                product = operation.get("product")
                quantity = operation.get("quantity")
                for _ in range(quantity):
                    if operation.get("type") == "add":
                        res = False
                        # This retry loop has a bug: it sleeps even on success,
                        # unnecessarily slowing down the consumer.
                        while not res:
                            res = self.marketplace.add_to_cart(cart_id, product)
                            time.sleep(self.retry_wait_time)
                    elif operation.get("type") == "remove":
                        self.marketplace.remove_from_cart(cart_id, product)

            # Finalize the order and print the purchased items.
            products = self.marketplace.place_order(cart_id)

            for product in products:
                print(f"{self.name} bought {product}")


import logging
from logging.handlers import RotatingFileHandler
import functools
import inspect
import time

def setup_logger():
    """Configures a rotating file logger for the marketplace."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('marketplace.log', maxBytes=500000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.Formatter.converter = time.gmtime
    logger.addHandler(handler)


def log_function(wrapped_function):
    """A decorator that logs function entry, arguments, and return values."""
    @functools.wraps(wrapped_function)
    def wrapper(*args):
        logger = logging.getLogger(__name__)
        func_name = wrapped_function.__name__
        logger.info("Entering %s", {func_name})

        # Log function arguments.
        func_args = inspect.signature(wrapped_function).bind(*args).arguments
        func_args_str = '\n\t'.join(
            f"{var_name} = {var_value}"
            for var_name, var_value
            in func_args.items()
        )
        logger.info("\t%s", func_args_str)

        out = wrapped_function(*args)

        logger.info("Return: %s - %s", type(out), out)
        logger.info("Done running %s", func_name)
        return out

    return wrapper

# NOTE: This relative import suggests a project structure that is not fully
# represented by this single file. The imported functions are defined above.
from threading import Lock
from .logger import setup_logger, log_function

class Cart:
    """Represents a shopping cart, holding products and their source producers."""
    def __init__(self):
        # NOTE: Using parallel lists can be error-prone if not kept in sync.
        self.products = []
        self.producer_ids = []

    def add_to_cart(self, product, producer_id):
        """Adds a product and its producer source to the cart."""
        self.products.append(product)
        self.producer_ids.append(producer_id)

    def remove_from_cart(self, product):
        """
        Removes a product and returns the ID of the producer it came from.
        NOTE: This may be buggy if the cart contains identical products
        from different producers, as it only removes the first match.
        """
        for i in range(len(self.products)):
            if self.products[i] == product:
                producer_id = self.producer_ids[i]
                self.products.remove(product)
                self.producer_ids.pop(i)
                return producer_id
        return None


class Marketplace:
    """A thread-safe marketplace managing inventories, carts, and transactions."""
    def __init__(self, queue_size_per_producer):
        setup_logger()
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_queues = {} # Inventory for each producer, keyed by producer_id.
        self.producer_queues_locks = {} # A lock for each producer's inventory.
        self.producer_id_counter = 0
        self.producer_id_lock = Lock()
        self.carts = {} # Holds Cart objects, keyed by cart_id.
        self.cart_id_counter = 0
        self.cart_id_lock = Lock()

    @log_function
    def register_producer(self):
        """Atomically registers a new producer and returns a unique ID."""
        with self.producer_id_lock:
            producer_id = self.producer_id_counter
            self.producer_queues[producer_id] = []
            self.producer_queues_locks[producer_id] = Lock()
            self.producer_id_counter += 1
            return producer_id

    @log_function
    def publish(self, producer_id, product):
        """Adds a product to a producer's inventory if space is available."""
        with self.producer_queues_locks[producer_id]:
            if len(self.producer_queues[producer_id]) <= self.queue_size_per_producer:
                self.producer_queues[producer_id].append(product)
                return True
        return False

    @log_function
    def new_cart(self):
        """Creates a new Cart object and returns its unique ID."""
        with self.cart_id_lock:
            cart_id = self.cart_id_counter
            self.carts[cart_id] = Cart()
            self.cart_id_counter += 1
            return cart_id

    @log_function
    def add_to_cart(self, cart_id, product):
        """Moves a product from any producer's inventory to a consumer's cart."""
        producers_no = 0
        with self.producer_id_lock:
            producers_no = self.producer_id_counter

        for i in range(producers_no):
            with self.producer_queues_locks[i]:
                # Using a list's `in` and `remove` is O(N), inefficient for large inventories.
                if product in self.producer_queues[i]:
                    self.producer_queues[i].remove(product)
                    self.carts[cart_id].add_to_cart(product, i)
                    return True
        return False

    @log_function
    def remove_from_cart(self, cart_id, product):
        """Returns a product from a cart back to its original producer's inventory."""
        producer_id = self.carts[cart_id].remove_from_cart(product)
        if producer_id is not None:
            with self.producer_queues_locks[producer_id]:
                self.producer_queues[producer_id].append(product)

    @log_function
    def place_order(self, cart_id):
        """Returns a simple list of all products in a given cart."""
        return self.carts[cart_id].products


class Producer(Thread):
    """Represents a producer thread that supplies products to the marketplace."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        thread_arg = kwargs["daemon"]
        Thread.__init__(self, daemon=thread_arg)
        self.operations = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """Registers with the marketplace and enters an infinite loop of publishing products."""
        producer_id = self.marketplace.register_producer()
        while True:
            for operation in self.operations:
                product, quantity, sleep_time = operation
                time.sleep(sleep_time)
                for _ in range(quantity):
                    # Persistently try to publish the product.
                    if not self.marketplace.publish(producer_id, product):
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str