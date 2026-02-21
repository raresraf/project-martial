


"""
Implements a multi-threaded producer-consumer simulation for a marketplace.

Note: This file is named `consumer.py` but contains the full simulation
logic, including the Marketplace, Producer, and Consumer classes.

This module defines a system where:
- `Producer` threads generate products and publish them to a `Marketplace`.
- `Consumer` threads simulate users by creating carts, adding/removing
  products, and placing orders.
- `Marketplace` is the central, thread-safe class that manages the inventory,
  producer queues, and customer carts.
- `Product`, `Tea`, `Coffee` are dataclasses for the items being traded.
"""

from threading import Thread, Lock, currentThread
from Queue import Queue
import time
from dataclasses import dataclass


class Consumer(Thread):
    """A thread that simulates a consumer buying products from the marketplace."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of "cart" operations to perform. Each cart is a
                          list of dictionaries specifying an operation ('add' or
                          'remove'), a product, and a quantity.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait in seconds before retrying a
                                     failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main lifecycle of the consumer.

        Iterates through its assigned carts, creates them in the marketplace,
        executes all add/remove operations for each, and finally places the order.
        """
        for crt_cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for crt_operation in crt_cart:
                # Perform the operation (e.g., 'add') for the specified quantity.
                number_of_operations = 0
                while number_of_operations < crt_operation["quantity"]:
                    op_product = crt_operation["product"]

                    if crt_operation["type"] == "add":
                        return_code = self.marketplace.add_to_cart(cart_id, op_product)
                    elif crt_operation["type"] == "remove":
                        return_code = self.marketplace.remove_from_cart(cart_id, op_product)

                    # If the operation succeeded, move to the next.
                    # If it failed (e.g., product not available), wait and retry.
                    if return_code is True or return_code is None:
                        number_of_operations += 1
                    else:
                        time.sleep(self.retry_wait_time)

            # Once all operations for the cart are done, place the order.
            self.marketplace.place_order(cart_id)


class Marketplace:
    """A thread-safe marketplace for producers to sell and consumers to buy.

    This class manages all shared state, including product inventory, producer
    publishing limits, and active shopping carts. It uses multiple locks to
    ensure safe concurrent access.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at once.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.sizes_per_producer = []  # Tracks current product count for each producer.

        self.carts = {}  # Stores active shopping carts.
        self.number_of_carts = 0

        self.products = []  # Global list of available products.
        self.producers = {}  # Maps a product to its producer ID.

        # Locks to protect different parts of the shared state.
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
        """Allows a producer to publish a product to the marketplace.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue was full.
        """
        num_prod_id = int(producer_id)
        # Check if the producer has reached their publication limit.
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
            cart_id = self.number_of_carts
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """Moves a product from the marketplace inventory to a user's cart."""
        with self.lock_for_sizes:
            if product not in self.products:
                return False  # Product not available.

            self.products.remove(product)
            # Decrement the producer's product count, allowing them to publish more.
            producer = self.producers[product]
            self.sizes_per_producer[producer] -= 1

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """Moves a product from a user's cart back to the marketplace."""
        self.carts[cart_id].remove(product)
        # Increment the producer's count, as the product is available again.
        with self.lock_for_sizes:
            producer = self.producers[product]
            self.sizes_per_producer[producer] += 1
        self.products.append(product)

    def place_order(self, cart_id):
        """Finalizes an order, printing the items and clearing the cart."""
        product_list = self.carts.pop(cart_id, None)
        if product_list is None:
            return []

        for prod in product_list:
            with self.lock_for_print:
                print(str(currentThread().getName()) + " bought " + str(prod))
        return product_list


class Producer(Thread):
    """A thread that simulates a producer creating and publishing products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main lifecycle of the producer.

        Continuously attempts to publish its products to the marketplace.
        If a publication fails (e.g., queue is full), it waits and retries.
        """
        # A creative way of writing `while True`.
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


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product with a name and a price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea, inheriting from Product and adding a type."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, inheriting from Product and adding attributes."""
    acidity: str
    roast_level: str
