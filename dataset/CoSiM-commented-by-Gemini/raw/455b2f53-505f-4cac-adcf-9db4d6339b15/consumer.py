"""
This module implements a multithreaded producer-consumer simulation of an
e-commerce marketplace.

It defines roles for Producers who publish products, Consumers who add/remove
products from carts and place orders, and a central Marketplace that manages
the inventory and transactions. The simulation uses threading and locks to handle
concurrent operations.
"""

import time
from threading import Thread


class Consumer(Thread):
    """Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread and processes a predefined list of
    shopping actions (e.g., adding or removing items from a cart).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts", where each is a list of
                          operation dictionaries.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait in seconds before retrying a
                                     failed operation.
            **kwargs: Arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Maps operation names to marketplace methods.
        self.operations = {"add": marketplace.add_to_cart,
                           "remove": marketplace.remove_from_cart}

    def run(self):
        """The main execution loop for the consumer.

        Processes each cart sequentially. For each operation in a cart, it
        repeatedly attempts the action (e.g., adding a product) until it
        succeeds, waiting between retries. Once all operations for a cart are
        done, it places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]

                # Loop until the desired quantity of the operation is fulfilled.
                while quantity > 0:
                    operation_type = operation["type"]
                    product = operation["product"]

                    # Attempt the operation (add/remove).
                    if self.operations[operation_type](cart_id, product) is not False:
                        quantity -= 1
                    else:
                        # If the operation fails (e.g., product not available), wait and retry.
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)

import sys
from threading import Lock, currentThread


class Marketplace:
    """A thread-safe marketplace that manages producers, products, and carts.

    This central class orchestrates the interactions between producers and
    consumers, using locks to manage concurrent access to its shared state.
    Note: Some methods may have subtle race conditions in their implementation.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        self.carts_lock = Lock()
        self.carts = []

        self.producers_lock = Lock()
        self.producers_capacity = queue_size_per_producer
        self.producers_sizes = [] # Tracks current item count per producer.
        self.products = [] # List of available products as (product, producer_id).

    def register_producer(self):
        """Registers a new producer, returning a unique producer ID."""
        with self.producers_lock:
            self.producers_sizes.append(0)
            return len(self.producers_sizes) - 1

    def publish(self, producer_id, product):
        """Allows a producer to list a product in the marketplace.

        Fails if the producer is already at their capacity.
        Returns:
            bool: True if the product was published successfully, False otherwise.
        """
        with self.producers_lock:
            if self.producers_sizes[producer_id] == self.producers_capacity:
                return False

            self.producers_sizes[producer_id] += 1
            self.products.append((product, producer_id))
            return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its ID."""
        with self.carts_lock:
            self.carts.append([])
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """Moves a product from the marketplace to a consumer's cart.

        This method searches for the specified product, and if found, transfers
        it to the cart. It locks the producer/product list during the operation.

        Returns:
            bool: True on success, False if the product is not found.
        """
        self.producers_lock.acquire()
        for (prod, prod_id) in self.products:
            if prod == product:
                self.producers_sizes[prod_id] -= 1
                self.products.remove((prod, prod_id))
                self.producers_lock.release()
                # Note: Accessing self.carts is not protected by carts_lock here.
                self.carts[cart_id].append((prod, prod_id))
                return True

        self.producers_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """Moves a product from a consumer's cart back to the marketplace."""
        # Note: Iterating and modifying self.carts is not protected by a lock,
        # which could be a race condition if multiple consumers shared a cart.
        for (prod, prod_id) in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove((prod, prod_id))
                self.producers_lock.acquire()
                self.products.append((prod, prod_id))
                self.producers_sizes[prod_id] += 1
                self.producers_lock.release()
                return

    def place_order(self, cart_id):
        """Finalizes an order, printing the cart's contents to stdout."""
        order = ""
        # Note: Accessing self.carts is not protected by carts_lock here.
        for (product, _) in self.carts[cart_id]:
            order += "{} bought {}\n".format(
                currentThread().getName(), product)
        sys.stdout.write(order)
        return self.carts[cart_id]


import time
from threading import Thread


class Producer(Thread):
    """Represents a producer that publishes products to the marketplace.

    Each producer runs in its own thread, attempting to publish a list of
    products according to their specified quantities and timings.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each is
                             (product, quantity, wait_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to
                                         publish a product if at capacity.
            **kwargs: Arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time


        self.producer_id = marketplace.register_producer()

    def run(self):
        """The main execution loop for the producer.

        Continuously loops through its product list, attempting to publish
        each one. If a publish fails (due to capacity), it waits and retries.
        """
        while True:
            for (product, quantity, wait_time) in self.products:
                while quantity > 0:
                    if self.marketplace.publish(self.producer_id, product):
                        quantity -= 1
                        time.sleep(wait_time)
                    else:
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product, with a name and a price."""
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
