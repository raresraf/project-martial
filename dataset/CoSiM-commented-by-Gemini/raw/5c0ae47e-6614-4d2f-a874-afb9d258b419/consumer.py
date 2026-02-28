# -*- coding: utf-8 -*-
"""
Models a Producer-Consumer scenario simulating a simple e-commerce marketplace.

This script implements a marketplace with Producers and Consumers, each running in its
own thread. It attempts to manage concurrent access using a series of fine-grained locks
for different operations within the Marketplace. This design, while attempting to increase
concurrency, introduces significant complexity and potential for race conditions.
"""

import time
from threading import Thread
from threading import Lock


class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace to buy products.

    Each consumer thread processes a list of shopping tasks, creating a cart,
    adding or removing items, and finally placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts," where each cart is a list of actions (dicts).
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): The time in seconds to wait before retrying an action.
            **kwargs: Keyword arguments passed to the Thread constructor (e.g., name).
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # A lock to ensure that print statements from different threads are not interleaved.
        self.print_locked = Lock()

    def run(self):
        """
        The main execution loop for the consumer thread.

        Processes each cart, executes the add/remove operations, places the order,
        and prints the items bought.
        """
        for task_cart in self.carts:
            # Get a new cart from the marketplace for this set of tasks.
            current_cart = self.marketplace.new_cart()
            for task in task_cart:
                looper = task['quantity']
                # Invariant: Loop until the desired quantity of the product is processed.
                while looper > 0:
                    # Pre-condition: Check if the product is available in the market stock.
                    # Note: This check is not atomic with the subsequent removal, which can
                    # lead to a Time-of-Check-to-Time-of-Use (TOCTTOU) race condition.
                    if task['product'] in self.marketplace.market_stock:
                        self.execute_task(task['type'], current_cart, task['product'])
                        looper -= 1
                    else:
                        # If the product is not available, wait and retry.
                        time.sleep(self.retry_wait_time)

            # Finalize the transaction by placing the order.
            order = self.marketplace.place_order(current_cart)
            # Use a lock to prevent garbled output from concurrent print calls.
            with self.print_locked:
                for product in order:
                    print(self.getName(), "bought", product)

    def execute_task(self, task_type, cart_id, product):
        """
        A helper method to call the appropriate marketplace action.

        Args:
            task_type (str): The type of action ('add' or 'remove').
            cart_id (int): The ID of the current cart.
            product (str): The product to act upon.
        """
        if task_type == 'remove':
            self.marketplace.remove_from_cart(cart_id, product)
        elif task_type == 'add':
            self.marketplace.add_to_cart(cart_id, product)

from threading import Lock


class Marketplace:
    """
    A central marketplace that manages producers, products, and carts.

    This implementation uses a set of fine-grained locks to try and manage
    concurrent access to its different components.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): Max products a producer can have in the market.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.no_of_producers = -1
        self.no_of_carts = -1
        # Maps a product to the ID of the producer who created it.
        self.product_creator = {}
        # A single list containing all products currently available for sale.
        self.market_stock = []
        # Tracks the number of products each producer has in the market.
        self.product_counter = []
        # A list of lists, where each inner list represents a cart's contents.
        self.cart = [[]]

        # A series of locks for different operations to attempt fine-grained concurrency.
        self.register_locked = Lock()
        self.cart_locked = Lock()
        self.add_locked = Lock()
        self.remove_locked = Lock()
        self.publish_locked = Lock()
        self.market_locked = Lock()

    def register_producer(self):
        """Registers a new producer and returns a unique ID."""
        with self.register_locked:
            self.no_of_producers += 1
            new_prod_id = self.no_of_producers
        
        self.product_counter.append(0)
        return new_prod_id

    def publish(self, producer_id, product):
        """Publishes a product from a producer to the marketplace."""
        # Pre-condition: Check if the producer is already at its queue capacity.
        if self.product_counter[producer_id] >= self.queue_size_per_producer:
            return False

        self.market_stock.append(product)

        with self.publish_locked:
            self.product_counter[producer_id] += 1
            self.product_creator[product] = producer_id

        return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its ID."""
        with self.cart_locked:
            self.no_of_carts += 1
            new_cart_id = self.no_of_carts
        
        self.cart.append([])
        return new_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product from the market to a specific cart.

        Note: This method is subject to race conditions. The check for product
        existence is not atomic with its removal from `market_stock`.
        """
        # Non-atomic check: another thread could remove the product after this check passes.
        if product not in self.market_stock:
            return False
        self.cart[cart_id].append(product)
        with self.add_locked:
            # Decrement the original producer's product counter.
            self.product_counter[self.product_creator[product]] -= 1
        with self.market_locked:
            # This lock only protects the remove call itself, not the whole operation.
            if product in self.market_stock:
                self.market_stock.remove(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the market stock."""
        if product in self.cart[cart_id]:
            with self.cart_locked:
                # Increment the original producer's product counter.
                self.product_counter[self.product_creator[product]] += 1
            self.cart[cart_id].remove(product)
            self.market_stock.append(product)

    def place_order(self, cart_id):
        """Returns the contents of a cart to finalize an order."""
        return self.cart[cart_id]

import time
from threading import Thread


class Producer(Thread):
    """
    Represents a producer that creates products and publishes them to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """The main execution loop for the producer thread."""
        while True:
            # Invariant: Continuously cycle through the product creation tasks.
            for (product, quantity, wait_time) in self.products:
                looper = quantity
                while looper > 0:
                    response = self.marketplace.publish(self.prod_id, product)
                    if response:
                        # If publish is successful, simulate production time and decrement counter.
                        time.sleep(wait_time)
                        looper -= 1
                    else:
                        # If publish fails (queue full), wait and retry.
                        time.sleep(self.republish_wait_time)


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
