


"""
Implements a modular, multi-threaded producer-consumer marketplace simulation.

Note: This file is named `consumer.py` but contains the full simulation
logic, including the Marketplace, Cart, Producer, and Product classes, as
well as a logging framework.

This module features a clean design with:
- A `Marketplace` using fine-grained, per-producer locks.
- A `Cart` class to encapsulate cart logic.
- A `@log_function` decorator for clear, non-invasive logging of marketplace
  operations.
"""
import functools
import inspect
import logging
import time
from logging.handlers import RotatingFileHandler
from threading import Lock, Thread
from dataclasses import dataclass


class Consumer(Thread):
    """A thread simulating a consumer that buys products from the marketplace."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread."""
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name")

    def run(self):
        """Processes its assigned shopping carts sequentially."""
        for cart_operations in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart_operations:
                product = operation.get("product")
                quantity = operation.get("quantity")
                op_type = operation.get("type")

                for _ in range(quantity):
                    if op_type == "add":
                        # Retry adding to cart until the product is available.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif op_type == "remove":
                        self.marketplace.remove_from_cart(cart_id, product)

            products = self.marketplace.place_order(cart_id)
            for product in products:
                print(f"{self.name} bought {product}")


def setup_logger():
    """Configures a rotating file logger for the marketplace."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Avoid adding handlers multiple times
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=500000, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logging.Formatter.converter = time.gmtime
        logger.addHandler(handler)


def log_function(wrapped_function):
    """A decorator that logs function entry, arguments, and return values."""
    @functools.wraps(wrapped_function)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        func_name = wrapped_function.__name__
        logger.info("Entering %s", func_name)

        # Log function arguments.
        bound_args = inspect.signature(wrapped_function).bind(*args, **kwargs)
        func_args_str = '\n\t'.join(f"{name} = {value}" for name, value in bound_args.arguments.items())
        logger.info("\t%s", func_args_str)

        # Execute the wrapped function.
        out = wrapped_function(*args, **kwargs)

        logger.info("Return: %s - %s", type(out), out)
        logger.info("Done running %s", func_name)
        return out
    return wrapper


class Cart:
    """Encapsulates the data for a single shopping cart."""

    def __init__(self):
        """Initializes an empty cart."""
        self.products = []
        self.producer_ids = []  # Tracks the original producer for each product.

    def add_to_cart(self, product, producer_id):
        """Adds a product and its producer ID to the cart."""
        self.products.append(product)
        self.producer_ids.append(producer_id)

    def remove_from_cart(self, product):
        """Removes a product and returns the ID of the producer it came from."""
        for i, p in enumerate(self.products):
            if p == product:
                producer_id = self.producer_ids.pop(i)
                self.products.pop(i)
                return producer_id
        return None


class Marketplace:
    """A thread-safe marketplace with per-producer product queues."""

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace."""
        setup_logger()
        self.queue_size_per_producer = queue_size_per_producer

        # Use dictionaries for efficient, thread-safe producer management.
        self.producer_queues = {}
        self.producer_queues_locks = {}
        self.producer_id_counter = 0
        self.producer_id_lock = Lock()

        # Use a dictionary to map cart IDs to Cart objects.
        self.carts = {}
        self.cart_id_counter = 0
        self.cart_id_lock = Lock()

    @log_function
    def register_producer(self):
        """Atomically registers a new producer, returning a unique ID."""
        with self.producer_id_lock:
            producer_id = self.producer_id_counter
            self.producer_queues[producer_id] = []
            self.producer_queues_locks[producer_id] = Lock()
            self.producer_id_counter += 1
            return producer_id

    @log_function
    def publish(self, producer_id, product):
        """Publishes a product to a specific producer's queue."""
        with self.producer_queues_locks[producer_id]:
            if len(self.producer_queues[producer_id]) < self.queue_size_per_producer:
                self.producer_queues[producer_id].append(product)
                return True
        return False

    @log_function
    def new_cart(self):
        """Creates a new, empty Cart object and returns its ID."""
        with self.cart_id_lock:
            cart_id = self.cart_id_counter
            self.carts[cart_id] = Cart()
            self.cart_id_counter += 1
            return cart_id

    @log_function
    def add_to_cart(self, cart_id, product):
        """Adds a product to a cart by searching all producer queues."""
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
        """Removes a product from a cart and returns it to its producer."""
        producer_id = self.carts[cart_id].remove_from_cart(product)
        if producer_id is not None:
            with self.producer_queues_locks[producer_id]:
                self.producer_queues[producer_id].append(product)

    @log_function
    def place_order(self, cart_id):
        """Finalizes an order by returning the products in the cart."""
        return self.carts.get(cart_id, Cart()).products


class Producer(Thread):
    """A thread that produces products and publishes them to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread."""
        Thread.__init__(self, **kwargs)
        self.operations = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """Registers itself and then enters a loop to produce and publish items."""
        producer_id = self.marketplace.register_producer()
        while True:
            for product, quantity, sleep_time in self.operations:
                for _ in range(quantity):
                    if not self.marketplace.publish(producer_id, product):
                        time.sleep(self.republish_wait_time)
                    time.sleep(sleep_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee."""
    acidity: str
    roast_level: str
