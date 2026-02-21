"""A refined producer-consumer marketplace simulation.

This module provides a well-structured, multi-threaded simulation of an
e-commerce marketplace. It features a robust, fine-grained locking strategy
and decorator-based logging for observability.

The key architectural choice is the `Marketplace`'s use of per-producer queues,
each with its own lock. This allows for high concurrency, as operations on
different producers' inventories do not block each other.
"""

import time
from threading import Thread


class Consumer(Thread):
    """Represents a consumer thread that purchases items from the marketplace."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread."""
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name")

    def run(self):
        """The main execution loop for the consumer.
        
        Processes a list of shopping actions, using a busy-wait loop to retry
        adding items if they are not yet available. After processing all actions,
        it places the order and prints the items bought.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                product = operation.get("product")
                quantity = operation.get("quantity")
                for _ in range(quantity):
                    if operation.get("type") == "add":
                        # Busy-wait until the product is successfully added.
                        res = False
                        while not res:
                            res = self.marketplace.add_to_cart(cart_id, product)
                            time.sleep(self.retry_wait_time)
                    elif operation.get("type") == "remove":
                        self.marketplace.remove_from_cart(cart_id, product)

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
        logger.info("Entering %s", func_name)

        # Log function arguments.
        try:
            func_args = inspect.signature(wrapped_function).bind(*args).arguments
            func_args_str = '\n\t'.join(
                f"{var_name} = {var_value}"
                for var_name, var_value
                in func_args.items()
            )
            logger.info("\t%s", func_args_str)
        except Exception:
            logger.error("Could not log function arguments.")

        out = wrapped_function(*args)

        # Log return value.
        logger.info("Return: %s - %s", type(out), out)
        logger.info("Done running %s", func_name)

        return out
    return wrapper


from threading import Lock
# This local import suggests a project structure that was not preserved.
from .logger import setup_logger, log_function

class Cart:
    """A simple data class representing a consumer's shopping cart."""

    def __init__(self):
        """Initializes an empty cart."""
        self.products = []
        self.producer_ids = []  # Tracks the origin of each product.

    def add_to_cart(self, product, producer_id):
        """Adds a product and its original producer ID to the cart."""
        self.products.append(product)
        self.producer_ids.append(producer_id)

    def remove_from_cart(self, product):
        """Removes a product and returns the ID of the producer it came from."""
        for i in range(len(self.products)):
            if self.products[i] == product:
                producer_id = self.producer_ids[i]
                self.products.pop(i)
                self.producer_ids.pop(i)
                return producer_id
        return None


class Marketplace:
    """The central marketplace, using fine-grained locking for high concurrency.

    This class manages inventories in per-producer queues, each protected by its
    own lock. This allows concurrent operations across different producers,
    improving performance over a single global lock.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace and its logging."""
        setup_logger()
        self.queue_size_per_producer = queue_size_per_producer

        self.producer_queues = {}
        self.producer_queues_locks = {}
        self.producer_id_counter = 0
        self.producer_id_lock = Lock()

        self.carts = {}
        self.cart_id_counter = 0
        self.cart_id_lock = Lock()

    @log_function
    def register_producer(self) -> int:
        """Atomically registers a new producer, creating a queue and lock for it."""
        with self.producer_id_lock:
            producer_id = self.producer_id_counter
            self.producer_queues[producer_id] = []
            self.producer_queues_locks[producer_id] = Lock()
            self.producer_id_counter += 1
            return producer_id

    @log_function
    def publish(self, producer_id, product) -> bool:
        """Adds a product to a specific producer's queue, if not full."""
        with self.producer_queues_locks[producer_id]:
            if len(self.producer_queues[producer_id]) < self.queue_size_per_producer:
                self.producer_queues[producer_id].append(product)
                return True
        return False

    @log_function
    def new_cart(self) -> int:
        """Atomically creates a new, empty `Cart` object."""
        with self.cart_id_lock:
            cart_id = self.cart_id_counter
            self.carts[cart_id] = Cart()
            self.cart_id_counter += 1
            return cart_id

    @log_function
    def add_to_cart(self, cart_id, product) -> bool:
        """Atomically moves a product from a producer's queue to a cart.
        
        This method iterates through all producer queues. When it finds the
        product, it locks that specific queue and performs the transfer,
        ensuring the operation is atomic and preventing race conditions.
        """
        producers_no = 0
        with self.producer_id_lock:
            producers_no = self.producer_id_counter

        for i in range(producers_no):
            with self.producer_queues_locks[i]:
                if product in self.producer_queues[i]:
                    self.producer_queues[i].remove(product)
                    self.carts[cart_id].add_to_cart(product, i)
                    return True
        return False

    @log_function
    def remove_from_cart(self, cart_id, product):
        """Atomically removes a product from a cart and returns it to the producer."""
        producer_id = self.carts[cart_id].remove_from_cart(product)
        if producer_id is not None:
            with self.producer_queues_locks[producer_id]:
                self.producer_queues[producer_id].append(product)

    @log_function
    def place_order(self, cart_id) -> list:
        """Returns the list of products currently in the specified cart."""
        return self.carts[cart_id].products


from threading import Thread
import time


class Producer(Thread):
    """Represents a producer thread that adds products to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer thread."""
        thread_arg = kwargs.get("daemon")
        Thread.__init__(self, daemon=thread_arg)
        self.operations = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """The main loop for the producer.
        
        Registers itself and then continuously attempts to publish products,
        using a busy-wait loop if its queue in the marketplace is full.
        """
        producer_id = self.marketplace.register_producer()
        while True:
            for operation in self.operations:
                product = operation[0]
                quantity = operation[1]
                sleep_time = operation[2]
                for _ in range(quantity):
                    time.sleep(sleep_time)
                    # Busy-wait until the product can be published.
                    while not self.marketplace.publish(producer_id, product):
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """An immutable data class for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A `Product` subclass representing Tea."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A `Product` subclass representing Coffee."""
    acidity: str
    roast_level: str
