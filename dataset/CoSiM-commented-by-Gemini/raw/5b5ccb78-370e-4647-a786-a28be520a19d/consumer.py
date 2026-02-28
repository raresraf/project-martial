# -*- coding: utf-8 -*-
"""
Models a Producer-Consumer scenario simulating a simple e-commerce marketplace.

This script implements three main components:
- Marketplace: A central class that manages product inventory and customer carts.
- Producer: A thread that creates products and adds them to the marketplace.
- Consumer: A thread that simulates a customer adding products to a cart and placing an order.

The simulation uses basic threading and locking, but it should be noted that the
concurrency control is not fully robust and contains potential race conditions. The
retry logic uses busy-waiting with `time.sleep`, which is inefficient.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer thread processes a list of shopping carts, where each cart contains
    a series of actions (add/remove products).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts", where each cart is a list of actions (dicts).
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): The time in seconds to wait before retrying to add a product.
            **kwargs: Keyword arguments passed to the Thread constructor (e.g., name).
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def add_product(self, cart_id, product, quantity):
        """
        Adds a specified quantity of a product to a cart.

        This method will repeatedly try to add the product if the marketplace
        cannot fulfill the request immediately, waiting a fixed time between retries.

        Args:
            cart_id (int): The ID of the cart to add to.
            product (str): The name of the product to add.
            quantity (int): The number of units of the product to add.
        """
        # Invariant: Loop until the desired quantity of the product is added.
        for _ in range(quantity):
            tmp = self.marketplace.add_to_cart(cart_id, product)

            # Pre-condition: If adding to the cart fails (product is unavailable),
            # enter a busy-wait loop.
            while tmp is False:
                time.sleep(self.retry_wait_time)
                tmp = self.marketplace.add_to_cart(cart_id, product)

    def remove_product(self, cart_id, product, quantity):
        """
        Removes a specified quantity of a product from a cart.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product (str): The name of the product to remove.
            quantity (int): The number of units of the product to remove.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """
        The main execution loop for the consumer thread.

        Processes each cart, executes the add/remove operations, places the order,
        and prints the items bought.
        """
        for cart in self.carts:
            # Each consumer gets a new cart ID from the marketplace.
            cart_id = self.marketplace.new_cart()

            # Process all requests (add/remove) in the current cart.
            for request in cart:
                order = request["type"]
                product = request["product"]
                quantity = request["quantity"]

                if order == "add":
                    self.add_product(cart_id, product, quantity)
                elif order == "remove":
                    self.remove_product(cart_id, product, quantity)

            # Finalize the cart by placing the order.
            order = self.marketplace.place_order(cart_id)

            # Print the final list of purchased products.
            for product in order:
                print(self.name + " bought " + str(product))

from threading import Lock


class Marketplace:
    """
    A central marketplace that manages producers, products, and carts.

    This class acts as the shared resource in the Producer-Consumer model.
    Note: The locking mechanism is minimal and only protects counter increments,
    making the data structures (`products`, `carts`) vulnerable to race conditions
    in a highly concurrent environment.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.prod_count = 0
        self.cart_count = 0

        # Dictionary to store products available from each producer.
        # Key: producer_id, Value: list of products.
        self.products = {}
        # Dictionary to store the contents of active shopping carts.
        # Key: cart_id, Value: list of (product, producer_id) tuples.
        self.carts = {}
        # A single lock to protect shared counters.
        self.lock = Lock()

    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID.

        Returns:
            int: The new producer's unique ID.
        """
        # Atomically increment the producer count.
        self.lock.acquire()
        self.prod_count = self.prod_count + 1
        self.lock.release()

        self.products[self.prod_count] = []
        return self.prod_count

    def publish(self, producer_id, product):
        """
        Adds a product to a producer's public inventory.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The product being published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue is full.
        """
        # Check if the producer's product queue is at capacity.
        lenght = len(self.products[producer_id])
        if lenght > self.queue_size_per_producer:
            return False

        # Add the product to the producer's list.
        self.products[producer_id].append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its ID.

        Returns:
            int: The new cart's unique ID.
        """
        # Atomically increment the cart count.
        self.lock.acquire()
        self.cart_count = self.cart_count + 1
        self.lock.release()

        self.carts[self.cart_count] = []
        return self.cart_count

    def add_to_cart(self, cart_id, product):
        """
        Finds a product in the marketplace and adds it to a cart.

        This method iterates through all producers to find the requested product.
        Note: This operation (checking for and removing the product) is not atomic
        and can lead to race conditions.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The name of the product to find and add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        # Search for the product across all producers.
        for producer_id in self.products:
            if product in self.products[producer_id]:
                # Found the product.
                tmp = (product, producer_id)

                # Move the product from the producer's inventory to the consumer's cart.
                self.carts[cart_id].append(tmp)
                self.products[producer_id].remove(product)
                return True

        # Product not found in any producer's inventory.
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer.

        Args:
            cart_id (int): The ID of the cart.
            product (str): The name of the product to remove.
        """
        # Find the product in the specified cart.
        for tmp in self.carts[cart_id]:
            current_prod = tmp[0]
            producer_id = tmp[1]

            if product == current_prod:
                # Return the product to the original producer's inventory.
                self.products[producer_id].append(product)
                self.carts[cart_id].remove(tmp)
                return

    def place_order(self, cart_id):
        """
        Finalizes an order, returning the list of products and deleting the cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of product names that were in the cart.
        """
        order = []
        # Create a simple list of product names for the final order.
        for product in self.carts[cart_id]:
            order.append(product[0])

        # Remove the cart from the marketplace.
        self.carts.pop(cart_id)

        return order


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer that creates products and publishes them to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tasks, where each task is a tuple of
                             (product_name, quantity, production_time).
            marketplace (Marketplace): The shared marketplace object.
            republish_wait_time (float): The time to wait before retrying to publish.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Register with the marketplace to get a unique producer ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread.
        """
        while True:
            # Invariant: Continuously cycle through the product creation tasks.
            for task in self.products:
                product = task[0]
                quantity = task[1]
                making_time = task[2]

                # Produce the specified quantity of the current product.
                for _ in range(quantity):
                    temp = self.marketplace.publish(self.producer_id, product)

                    # Pre-condition: If publishing fails (queue is full), enter a
                    # busy-wait loop.
                    while not temp:
                        time.sleep(self.republish_wait_time)
                        temp = self.marketplace.publish(self.producer_id, product)

                    # Simulate the time it takes to "make" the product.
                    time.sleep(making_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple, immutable data class representing a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """An immutable data class representing a type of Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """An immutable data class representing a type of Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
