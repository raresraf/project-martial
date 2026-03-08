# -*- coding: utf-8 -*-
"""
A multi-threaded simulation of a producer-consumer marketplace.

This script models a simple e-commerce system with the following components:
- Marketplace: The central hub where products are listed and carts are managed.
- Producers: Threads that create products and add them to the marketplace inventory.
- Consumers: Threads that simulate customers adding products to a cart and placing orders.

The simulation uses threading and locks to manage concurrent access to shared resources,
demonstrating a classic producer-consumer problem.
"""

from threading import Thread, Lock
from time import sleep
from queue import Full
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a customer shopping in the marketplace.

    Each consumer has a predefined list of shopping carts, where each cart contains
    a series of 'add' or 'remove' operations. The consumer processes these operations
    sequentially for each cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "shopping carts", where each cart is a list of
                          product operations (add/remove).
            marketplace (Marketplace): The central marketplace object.
            retry_wait_time (float): Time in seconds to wait before retrying to add
                                     a product if it's not available.
            **kwargs: Catches the 'name' of the thread.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through its assigned carts, simulates adding and removing products,
        and finally "buys" the products by placing an order.
        """
        # Invariant: Process each shopping cart provided during initialization.
        for cart in self.carts:
            # Get a new, unique cart ID from the marketplace for this shopping session.
            cart_id = self.marketplace.new_cart()

            # Pre-condition: Each cart contains a list of operations to perform.
            for operation in cart:
                # Perform the 'add' or 'remove' operation for the specified quantity.
                for _ in range(operation["quantity"]):
                    if operation["type"] == "add":
                        # Block-Logic: Attempt to add a product to the cart. If the marketplace
                        # cannot fulfill the request (i.e., product is out of stock),
                        # wait for a specified time and retry.
                        while not self.marketplace.add_to_cart(cart_id, operation["product"]):
                            sleep(self.retry_wait_time)
                    elif operation["type"] == "remove":
                        # Remove a product from the cart. This action returns the
                        # product to the original producer's inventory.
                        self.marketplace.remove_from_cart(cart_id, operation["product"])

            # Finalize the shopping session by placing the order.
            products = self.marketplace.place_order(cart_id)

            # Announce the purchased products.
            for product in products:
                print("{0} bought {1}".format(self.name, product))


class SafeList:
    """
    A simple thread-safe list implementation using a mutex (Lock).

    This class wraps a standard Python list to ensure that all modifications
    (put, remove) are atomic operations, preventing race conditions when accessed
    by multiple threads.
    """

    def __init__(self, maxsize=0):
        """
        Initializes the SafeList.

        Args:
            maxsize (int): The maximum number of items the list can hold.
                           If 0, the list size is unbounded.
        """
        self.mutex = Lock()
        self.list = []
        self.maxsize = maxsize

    def put(self, item):
        """
        Adds an item to the list in a thread-safe manner.

        Args:
            item: The item to add to the list.

        Raises:
            Full: If the list is at its maximum capacity.
        """
        with self.mutex:
            if self.maxsize != 0 and self.maxsize == len(self.list):
                raise Full
            self.list.append(item)

    def put_anyway(self, item):
        """
        Adds an item to the list, ignoring the maxsize constraint.

        This is used for operations like returning an item to stock where
        the capacity limit should be bypassed.

        Args:
            item: The item to add to the list.
        """
        with self.mutex:
            self.list.append(item)

    def remove(self, item):
        """
        Removes an item from the list in a thread-safe manner.

        Args:
            item: The item to remove from the list.

        Returns:
            bool: True if the item was found and removed, False otherwise.
        """
        with self.mutex:
            if item not in self.list:
                return False
            self.list.remove(item)
            return True


class Cart:
    """
    Represents a single shopping cart, holding products for a consumer.
    """

    def __init__(self):
        """Initializes an empty cart."""
        self.products = []

    def add_product(self, product, producer_id):
        """
        Adds a product to the cart, tracking its original producer.

        Args:
            product: The product object to add.
            producer_id: The ID of the producer who supplied the product.
        """
        self.products.append({"product": product, "producer_id": producer_id})

    def remove_product(self, product):
        """
        Removes a product from the cart.

        Args:
            product: The product object to remove.

        Returns:
            The producer_id of the removed product, or None if not found.
        """
        for item in self.products:
            if item["product"] == product:
                self.products.remove(item)
                return item["producer_id"]
        return None

    def get_products(self):
        """
        Returns a list of all product objects currently in the cart.
        """
        return map(lambda item: item["product"], self.products)


class Marketplace:
    """
    The central marketplace that manages producers, inventories, and customer carts.

    This class is the core of the simulation, orchestrating all interactions
    between producers and consumers in a thread-safe way.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have in their inventory queue.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Manages inventories for each producer. Key: producer_id, Value: SafeList of products.
        self.producer_queues = {}
        self.producer_id_generator = 0
        self.producer_id_generator_lock = Lock()

        # Manages active shopping carts. Key: cart_id, Value: Cart object.
        self.carts = {}
        self.cart_id_generator = 0
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer, giving them a unique ID and an inventory queue.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.producer_id_generator_lock:
            current_prod_id = self.producer_id_generator
            self.producer_queues[current_prod_id] = SafeList(maxsize=self.queue_size_per_producer)
            self.producer_id_generator += 1
            return current_prod_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to their inventory queue.

        Args:
            producer_id (int): The ID of the publishing producer.
            product: The product to publish.

        Returns:
            bool: True if publishing was successful, False if the producer's queue was full.
        """
        try:
            self.producer_queues[producer_id].put(product)
            return True
        except Full:
            return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.cart_id_generator_lock:
            current_cart_id = self.cart_id_generator
            self.carts[current_cart_id] = Cart()
            self.cart_id_generator += 1
            return current_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified cart by taking it from any available producer.

        Args:
            cart_id (int): The ID of the cart to add to.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        producers_num = 0
        with self.producer_id_generator_lock:
            producers_num = self.producer_id_generator

        # Block-Logic: Iterate through all registered producers to find the desired product.
        for producer_id in range(producers_num):
            # Attempt to remove the product from a producer's queue. If successful...
            if self.producer_queues[producer_id].remove(product):
                # ...add it to the consumer's cart and return True.
                self.carts[cart_id].add_product(product, producer_id)
                return True
        # If the loop completes without finding the product, return False.
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer's inventory.
        """
        producer_id = self.carts[cart_id].remove_product(product)
        # Return the product to the inventory, bypassing capacity checks.
        self.producer_queues[producer_id].put_anyway(product)

    def place_order(self, cart_id):
        """
        Finalizes an order by retrieving all products from the cart.

        Returns:
            A list of product objects in the cart.
        """
        return self.carts[cart_id].get_products()


class Producer(Thread):
    """
    Represents a producer thread that creates products and publishes them.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains
                             (product_object, quantity, production_time).
            marketplace (Marketplace): The central marketplace object.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the inventory queue is full.
            **kwargs: Catches 'daemon' and 'name' for the thread.
        """
        Thread.__init__(self, daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and then enters an infinite loop to
        produce and publish products.
        """
        producer_id = self.marketplace.register_producer()

        # Invariant: The producer will continuously cycle through its product list.
        while True:
            for (product, quantity, production_time) in self.products:
                # Simulate the time it takes to produce an item.
                sleep(production_time)
                # Produce the specified quantity of the item.
                for _ in range(quantity):
                    # Block-Logic: Attempt to publish the product. If the inventory is full,
                    # wait and retry until successful.
                    while not self.marketplace.publish(producer_id, product):
                        sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A dataclass representing a generic product with a name and a price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, a specific type of Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, a specific type of Product."""
    acidity: str
    roast_level: str
