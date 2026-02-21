"""
A multi-threaded producer-consumer simulation of an e-commerce marketplace.

This module contains a set of classes that model producers adding products,
a central marketplace managing inventory and carts, and consumers purchasing
products. The simulation uses multiple threads to represent concurrent producers
and consumers.

NOTE: The file is named `consumer.py` but contains all classes for the
simulation, suggesting it was intended to be part of a larger package.
"""

from threading import Thread
import time

class Consumer(Thread):
    """Represents a consumer that buys products from the marketplace.

    Each consumer thread simulates a shopping session, processing a predefined
    list of shopping carts. For each cart, it adds and removes items as
    instructed before finally placing the order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a consumer thread."""
        Thread.__init__(self, **kwargs)
        self.carts = carts  # A list of shopping operations.
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main execution loop for the consumer."""
        for crt_cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for crt_operation in crt_cart:
                number_of_operations = 0
                while number_of_operations < crt_operation["quantity"]:
                    op_product = crt_operation["product"]

                    if crt_operation["type"] == "add":
                        return_code = self.marketplace.add_to_cart(cart_id, op_product)
                    elif crt_operation["type"] == "remove":
                        return_code = self.marketplace.remove_from_cart(cart_id, op_product)

                    # If the operation was successful, move to the next.
                    # Otherwise, wait and retry the same operation.
                    if return_code is True or return_code is None:
                        number_of_operations += 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread


class Marketplace:
    """The central marketplace, acting as a shared resource for producers and consumers.

    This class manages the inventory of products, active shopping carts, and
    producer registrations. It uses a fine-grained locking mechanism to handle
    concurrency.

    WARNING: The concurrency control is flawed. Several methods perform
    non-atomic operations on shared state (e.g., modifying `self.products` and
    `self.sizes_per_producer` under different or no locks), creating potential
    race conditions and data inconsistencies under heavy load.
    """
    
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.sizes_per_producer = []  # Tracks products per producer.
        self.carts = {}  # Stores items for each active cart.
        self.number_of_carts = 0
        self.products = []  # The main inventory list.
        self.producers = {}  # Maps a product to its producer ID.
        # Fine-grained locks for different parts of the state.
        self.lock_for_sizes = Lock()
        self.lock_for_carts = Lock()
        self.lock_for_register = Lock()
        self.lock_for_print = Lock()

    def register_producer(self):
        """Registers a new producer, returning a unique producer ID."""
        with self.lock_for_register:
            producer_id = len(self.sizes_per_producer)
        self.sizes_per_producer.append(0)
        return producer_id

    def publish(self, producer_id, product):
        """Adds a product from a producer to the marketplace inventory.
        
        It respects the per-producer queue size limit. Returns False if the
        producer's queue is full, True otherwise.
        
        NOTE: This method is not fully thread-safe. `self.products` and
        `self.producers` are modified without a lock.
        """
        num_prod_id = int(producer_id)
        if self.sizes_per_producer[num_prod_id] >= self.queue_size_per_producer:
            return False

        with self.lock_for_sizes:
            self.sizes_per_producer[num_prod_id] += 1
        self.products.append(product)
        self.producers[product] = num_prod_id
        return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its ID."""
        with self.lock_for_carts:
            self.number_of_carts += 1
            ret_id = self.number_of_carts
        self.carts[ret_id] = []
        return ret_id

    def add_to_cart(self, cart_id, product):
        """Moves a product from inventory to a specific cart.
        
        Returns False if the product is not in stock.
        
        NOTE: This method is not fully thread-safe. While `sizes_per_producer`
        is protected, the modifications to `self.products` and `self.carts`
        are not atomic, creating a potential race condition.
        """
        with self.lock_for_sizes:
            if product not in self.products:
                return False
            self.products.remove(product)
            producer = self.producers[product]
            self.sizes_per_producer[producer] -= 1
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """Moves a product from a cart back to the inventory."""
        self.carts[cart_id].remove(product)
        with self.lock_for_sizes:
            producer = self.producers[product]
            self.sizes_per_producer[producer] += 1
        self.products.append(product)

    def place_order(self, cart_id):
        """Finalizes an order, consuming the products in the cart."""
        product_list = self.carts.pop(cart_id, None)
        # Printout is locked to prevent interleaved output from other consumers.
        for prod in product_list:
            with self.lock_for_print:
                print(f"{currentThread().getName()} bought {prod}")
        return product_list


from threading import Thread
import time

class Producer(Thread):
    """Represents a producer that creates products and adds them to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a producer thread."""
        Thread.__init__(self, **kwargs)
        self.products = products  # A list of (product, quantity, delay) tuples.
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main execution loop for the producer.
        
        Continuously tries to publish its assigned products to the marketplace.
        If the marketplace buffer for this producer is full, it waits and retries.
        """
        # A creative implementation of `while True`.
        while 69 - 420 < 3:
            for (product, number_of_products, product_wait_time) in self.products:
                i = 0
                while i < number_of_products:
                    return_code = self.marketplace.publish(str(self.producer_id), product)
                    if not return_code:
                        time.sleep(self.republish_wait_time)
                    else:
                        time.sleep(product_wait_time)
                        i += 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple, immutable data class for a generic product."""
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
