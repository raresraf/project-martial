# -*- coding: utf-8 -*-
"""
This module contains a self-contained, multi-threaded simulation of a
producer-consumer marketplace.

It defines the following key components:
- Consumer: A thread that simulates a customer's shopping process, from adding
  items to a cart to placing an order.
- Producer: A thread that simulates a producer adding a stock of products to
  the marketplace.
- Marketplace: The central, thread-safe class that manages the inventory and
  coordinates the actions of producers and consumers using locks.
- Product, Tea, Coffee: Dataclasses representing the items being traded.
"""
from threading import Thread, Lock
from time import sleep


class Consumer(Thread):
    """
    A consumer thread that attempts to acquire a list of products.

    This thread simulates a consumer who has a shopping list (`products_to_buy`)
    and attempts to add each item to their cart. If an item is not available,
    it waits and retries. Once all items are in the cart, it places an order.
    """

    def __init__(self, products_to_buy, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            products_to_buy (list): A list of products the consumer wants to buy.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying to get a product.
            **kwargs: Keyword arguments for the `Thread` constructor.
        """
        super().__init__(**kwargs)
        self.products_to_buy = products_to_buy
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution method for the consumer.

        Creates a new cart, then iterates through the shopping list, attempting
        to add each product. After successfully adding all products, it places
        the order.
        """
        cart_id = self.marketplace.new_cart()
        for product in self.products_to_buy:
            # Block Logic: This busy-wait loop attempts to acquire a product
            # until it succeeds.
            # Invariant: The loop continues until `add_to_cart` returns True.
            while True:
                if self.marketplace.add_to_cart(cart_id, product):
                    break
                sleep(self.retry_wait_time)
        self.marketplace.place_order(cart_id)


class Marketplace:
    """
    A thread-safe marketplace that manages inventory and transactions.

    This class serves as the shared resource between Producer and Consumer
    threads. It uses a single `Lock` to ensure that all operations on the
    shared inventory (`products`) and carts are atomic, preventing race
    conditions. The inventory is managed as a simple list of `Product` objects.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                                           single producer can publish.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.products = []
        self.lock = Lock()
        self.producers_count = {}
        self.carts = {}
        self.next_cart_id = 0
        self.next_producer_id = 0

    def register_producer(self):
        """
        Registers a new producer, returning a unique ID.

        Returns:
            int: The ID for the new producer.
        """
        with self.lock:
            producer_id = self.next_producer_id
            self.producers_count[producer_id] = 0
            self.next_producer_id += 1
            return producer_id

    def publish(self, producer_id, product):
        """
        Adds a product from a producer to the marketplace inventory.

        Args:
            producer_id (int): The ID of the producer.
            product (Product): The product to add.

        Returns:
            bool: True if the product was added, False if the producer's
                  queue is full.
        """
        with self.lock:
            # Pre-condition: Check if the producer is at their capacity limit.
            if self.producers_count[producer_id] >= self.queue_size_per_producer:
                return False
            self.products.append(product)
            self.producers_count[producer_id] += 1
            return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.lock:
            cart_id = self.next_cart_id
            self.carts[cart_id] = []
            self.next_cart_id += 1
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from the marketplace inventory to a consumer's cart.

        Args:
            cart_id (int): The ID of the target cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and moved, False otherwise.
        """
        with self.lock:
            # Pre-condition: Check if the desired product is available in the main inventory.
            if product not in self.products:
                return False
            self.products.remove(product)
            self.carts[cart_id].append(product)
            return True

    def remove_from_cart(self, cart_id, product):
        """
        Moves a product from a consumer's cart back to the marketplace inventory.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product (Product): The product to move back.

        Returns:
            bool: True if the product was in the cart and moved, False otherwise.
        """
        with self.lock:
            # Pre-condition: Check if the product is actually in the specified cart.
            if product not in self.carts[cart_id]:
                return False
            self.carts[cart_id].remove(product)
            self.products.append(product)
            return True

    def place_order(self, cart_id):
        """
        Finalizes an order by consuming the items in the cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: The list of products that were in the cart.
        """
        with self.lock:
            # This implementation doesn't track producers for purchased items,
            # so it just returns the items and clears the cart.
            items = self.carts[cart_id]
            self.carts[cart_id] = []
            for item in items:
                print(f"{self.name} bought {item}")
            return items


class Producer(Thread):
    """
    A producer thread that publishes a given number of products to the marketplace.
    """

    def __init__(self, products_to_publish, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products_to_publish (list): A list of tuples, each containing
                                        (product, quantity_to_produce).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait if publishing fails.
            **kwargs: Keyword arguments for the `Thread` constructor.
        """
        super().__init__(**kwargs)
        self.products_to_publish = products_to_publish
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution method for the producer.

        Loops through its list of products, attempting to publish the specified
        quantity of each. It will busy-wait if its queue is full.
        """
        while True:
            for product, quantity in self.products_to_publish:
                # Invariant: Loop until the target quantity for the product is published.
                while quantity > 0:
                    if self.marketplace.publish(self.producer_id, product):
                        quantity -= 1
                    else:
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, frozen=True)
class Product:
    """An immutable data class representing a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, frozen=True)
class Tea(Product):
    """A data class for Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, frozen=True)
class Coffee(Product):
    """A data class for Coffee, inheriting from Product."""
    acidity: float
    roast_level: str
