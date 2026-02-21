"""A producer-consumer marketplace simulation using a global lock.

This module provides a thread-safe simulation of an e-commerce marketplace.
It features Producer threads that add products, Consumer threads that purchase
them, and a central Marketplace that manages the state.

The key architectural choice in this implementation is the use of a single,
coarse-grained lock within the Marketplace to protect all shared data. This
ensures thread safety and atomicity of operations, but serializes all access,
creating a performance bottleneck as only one thread can operate at a time.
"""

from threading import Thread
import time

ADD_COMMAND = "add"
REMOVE_COMMAND = "remove"
COMMAND_TYPE = "type"
ITEM_QUANTITY = "quantity"
PRODUCT = "product"
NAME = "name"

class Consumer(Thread):
    """Represents a consumer thread that purchases items from the marketplace."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread."""
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs[NAME]

    def run(self):
        """The main execution loop for the consumer.
        
        Processes a list of shopping actions, using a busy-wait loop to retry
        adding items if they are not yet available. After processing all actions,
        it places the order and prints the items bought.
        """
        cart_id = self.marketplace.new_cart()

        for item in self.carts:
            for command in item:
                if command[COMMAND_TYPE] == ADD_COMMAND:
                    for _ in range(command[ITEM_QUANTITY]):
                        # Busy-wait until the product is available to be added.
                        while not self.marketplace.add_to_cart(id_cart, command[PRODUCT]):
                            time.sleep(self.retry_wait_time)
                elif command[COMMAND_TYPE] == REMOVE_COMMAND:
                    for _ in range(command[ITEM_QUANTITY]):
                        self.marketplace.remove_from_cart(id_cart, command[PRODUCT])

        order_result = self.marketplace.place_order(id_cart)

        for item in order_result:
            # Acquires the marketplace's global lock to ensure atomic printing.
            with self.marketplace.lock:
                print(f"{self.consumer_name} bought {item[1]}")


from logging.handlers import RotatingFileHandler
from threading import Lock
import logging

class Marketplace:
    """The central marketplace, synchronized with a single global lock.

    This class manages all shared state, including product inventory, carts,
    and producer information. It enforces thread safety by wrapping almost
    every method in a single, coarse-grained lock. This approach simplifies
    concurrency control and prevents race conditions but eliminates the
    possibility of parallel operations.
    """
    
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace state and its single lock."""
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.consumer_id = 0
        self.products = []
        self.producers = []
        self.carts = []
        self.lock = Lock()  # The single lock for all operations.

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = RotatingFileHandler("marketplace.log")
        self.logger.addHandler(file_handler)

    def register_producer(self):
        """Atomically registers a new producer and returns a new ID."""
        self.logger.info("Entered method: register_producer")
        with self.lock:
            self.producer_id += 1
            producer_id = self.producer_id
        self.logger.info("Exited method: register_producer")
        return producer_id

    def publish(self, producer_id, product):
        """Atomically adds a product to the marketplace inventory."""
        self.logger.info(f"Entered method: publish, Params: producer_id: {producer_id}, product: {product.name}")
        with self.lock:
            if self.producers[producer_id - 1].nr_products < self.queue_size_per_producer:
                self.products.append((producer_id, product))
                self.producers[producer_id - 1].nr_products += 1
                self.logger.info("Exited method: publish")
                return True
        self.logger.info("Exited method: publish")
        return False

    def new_cart(self):
        """Atomically creates a new, empty cart for a consumer."""
        self.logger.info("Entered method: new_cart")
        with self.lock:
            self.consumer_id += 1
            consumer_id = self.consumer_id
            self.carts.append([])
        self.logger.info("Exited method: new_cart")
        return consumer_id

    def add_to_cart(self, cart_id, product):
        """Atomically finds a product and moves it from inventory to a cart."""
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
        """Atomically finds a product in a cart and returns it to inventory."""
        self.logger.info(f"Entered method: remove_from_cart, Params: cart_id: {cart_id}, product: {product.name}")
        with self.lock:
            for item in self.carts[cart_id - 1]:
                if product == item[1]:
                    self.carts[cart_id - 1].remove(item)
                    self.products.append(item)
                    self.producers[item[0] - 1].nr_products += 1
                    self.logger.info("Exited method: remove_from_cart")
                    return
        
    def place_order(self, cart_id):
        """Returns the contents of a cart. This operation is read-only."""
        self.logger.info("Entered method: place_order")
        self.logger.info("Exited method: place_order")
        return self.carts[cart_id - 1]


from threading import Thread
import time


class Producer(Thread):
    """Represents a producer thread that adds products to the marketplace."""
    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer and registers it with the marketplace."""
        Thread.__init__(self, **kwargs)
        self.nr_products = 0
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Inversion of control: Producer adds itself to the marketplace's list.
        self.marketplace.producers.append(self)
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main loop for the producer.
        
        Continuously tries to publish products, using a busy-wait loop if the
        producer's capacity in the marketplace is full.
        """
        while True:
            for item in self.products:
                for _ in range(item[1]):
                    # Busy-wait until publish is successful.
                    while not self.marketplace.publish(self.producer_id, item[0]):
                        time.sleep(self.republish_wait_time)
                    time.sleep(item[2])


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
