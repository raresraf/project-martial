"""A producer-consumer marketplace simulation using a global lock.

This module provides a thread-safe simulation of an e-commerce marketplace.
It features Producer threads that add products, Consumer threads that purchase
them, and a central Marketplace that manages the state.

The key architectural choice in this implementation is the use of a single,
coarse-grained lock within the Marketplace to protect all shared data. This
ensures thread safety and atomicity of operations, but serializes all access,
creating a performance bottleneck as only one thread can operate at a time.
"""

"""
Implements a multi-threaded producer-consumer simulation for a marketplace.

Note: This file is named `consumer.py` but contains the full simulation
logic, including the Marketplace, Producer, and Product classes.

This module defines a system that uses a coarse-grained locking strategy, with
a single global lock protecting most marketplace operations. This simplifies
concurrency control at the cost of performance, as it serializes all access
to the shared marketplace state. The marketplace also features logging for
all its public methods.
"""
import time
from threading import Thread, Lock
from logging.handlers import RotatingFileHandler
import logging
from dataclasses import dataclass

# Constants for cart operations
ADD_COMMAND = "add"
REMOVE_COMMAND = "remove"
COMMAND_TYPE = "type"
ITEM_QUANTITY = "quantity"
PRODUCT = "product"
NAME = "name"

class Consumer(Thread):
    """A thread that simulates a consumer buying products."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs[NAME]

    def run(self):
        """Processes a list of shopping carts.

        For each assigned cart, it creates a new cart in the marketplace,
        executes the add/remove operations, and then places the final order.
        """
        id_cart = self.marketplace.new_cart()

        for item in self.carts:
            for command in item:
                if command[COMMAND_TYPE] == ADD_COMMAND:
                    # Retry adding to cart until the operation is successful.
                    for _ in range(command[ITEM_QUANTITY]):
                        while not self.marketplace.add_to_cart(id_cart, command[PRODUCT]):
                            time.sleep(self.retry_wait_time)
                elif command[COMMAND_TYPE] == REMOVE_COMMAND:
                    for _ in range(command[ITEM_QUANTITY]):
                        self.marketplace.remove_from_cart(id_cart, command[PRODUCT])

        order_result = self.marketplace.place_order(id_cart)

        # Print the purchased items, using the marketplace lock for thread-safe printing.
        for item in order_result:
            with self.marketplace.lock:
                print(f"{self.consumer_name} bought {str(item[1])}")


class Marketplace:
    """A thread-safe marketplace synchronized with a single global lock.

    This class manages all shared state, including products, producers, and carts.
    Nearly all operations are serialized by a single lock, which can become a
    performance bottleneck under high contention.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.consumer_id = 0
        self.products = []
        self.producers = []
        self.carts = []

        self.lock = Lock()  # The single global lock for most operations.

        # Setup for logging marketplace events to a file.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = RotatingFileHandler("marketplace.log")
        self.logger.addHandler(file_handler)

    def register_producer(self) -> int:
        """Registers a new producer, returning a unique ID."""
        self.logger.info("Entered method: register_producer")
        with self.lock:
            self.producer_id += 1
            producer_id = self.producer_id
        self.logger.info("Exited method: register_producer")
        return producer_id

    def publish(self, producer_id, product) -> bool:
        """Allows a producer to publish a product."""
        self.logger.info(f"Entered method: publish, Params: producer_id: {producer_id}, product: {product.name}")
        with self.lock:
            # Check if producer has capacity to publish more.
            if self.producers[producer_id - 1].nr_products < self.queue_size_per_producer:
                self.products.append((producer_id, product))
                self.producers[producer_id - 1].nr_products += 1
                self.logger.info("Exited method: publish")
                return True
        self.logger.info("Exited method: publish")
        return False

    def new_cart(self) -> int:
        """Creates a new, empty shopping cart."""
        self.logger.info("Entered method: new_cart")
        with self.lock:
            self.consumer_id += 1
            consumer_id = self.consumer_id
            self.carts.append([])
        self.logger.info("Exited method: new_cart")
        return consumer_id

    def add_to_cart(self, cart_id, product) -> bool:
        """Moves a product from inventory to a cart."""
        self.logger.info(f"Entered method: add_to_cart, Params: cart_id: {cart_id}, product: {product.name}")
        with self.lock:
            for item in self.products:
                if product == item[1]:
                    self.carts[cart_id - 1].append(item)
                    self.products.remove(item)
                    self.producers[item[0] - 1].nr_products -= 1
                    self.logger.info("Exited method: add_to_cart")
                    return True
        self.logger.info("Exited method: add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """Moves a product from a cart back to inventory."""
        self.logger.info(f"Entered method: remove_from_cart, Params: cart_id: {cart_id}, product: {product.name}")
        with self.lock:
            for item in self.carts[cart_id - 1]:
                if product == item[1]:
                    self.carts[cart_id - 1].remove(item)
                    self.products.append(item)
                    self.producers[item[0] - 1].nr_products += 1
                    self.logger.info("Exited method: remove_from_cart")
                    return
        self.logger.info("Exited method: remove_from_cart")

    def place_order(self, cart_id) -> list:
        """Returns the list of items in a cart to finalize the order."""
        self.logger.info("Entered method: place_order")
        return self.carts[cart_id - 1]


class Producer(Thread):
    """A thread that simulates a producer creating and publishing products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer.

        Note the direct coupling: the Producer adds itself to the Marketplace's
        list of producers upon creation.
        """
        Thread.__init__(self, **kwargs)
        self.nr_products = 0
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # The producer instance is directly coupled with the marketplace state.
        self.marketplace.producers.append(self)
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """Continuously produces items and retries publishing on failure."""
        while True:
            for item in self.products:
                for _ in range(item[1]):
                    while not self.marketplace.publish(self.producer_id, item[0]):
                        time.sleep(self.republish_wait_time)
                    time.sleep(item[2])


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
