"""
@file consumer.py
@brief A multi-threaded, multi-file simulation of an online marketplace.

This file contains all the components for a marketplace simulation, including
producers who publish products, consumers who purchase them, and the central
marketplace that facilitates these interactions. The simulation uses a
producer-consumer pattern with fine-grained locking and a retry-on-failure
mechanism for concurrent operations.
"""

from threading import Thread, Lock
from time import sleep
from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    A base dataclass representing a generic product.
    It is immutable (`frozen=True`) to ensure thread safety when passed around.
    """
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


class Marketplace:
    """
    The central hub of the simulation, managing all interactions between
    producers and consumers.

    This class acts as the shared resource, controlling access to product lists
    and shopping carts. It employs a fine-grained locking strategy, using a
    separate lock for each type of operation to manage concurrency.
    """
    
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace state and its various locks."""
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_index = -1
        self.consumer_cart_index = -1
        self.producer_queue_size_list = []
        self.products_list = []
        self.producers_to_products_map = {}
        self.carts = []
        # A fine-grained locking approach where each operation type has its own lock.
        self.lock_register = Lock()
        self.lock_publish = Lock()
        self.lock_new_cart = Lock()
        self.lock_add_cart = Lock()
        self.lock_remove_cart = Lock()

    def register_producer(self):
        """
        Atomically registers a new producer, providing a unique ID.
        @return The new producer's unique ID.
        """
        self.lock_register.acquire()
        self.producer_index += 1
        self.lock_register.release()
        self.producer_queue_size_list.append(0)
        return self.producer_index


    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        It fails if the producer's personal queue size limit is exceeded.
        @return True on success, False on failure.
        """
        if self.producer_queue_size_list[producer_id] <= self.queue_size_per_producer:
            self.producer_queue_size_list[producer_id] += 1
            self.products_list.append(product)
            self.lock_publish.acquire()
            self.producers_to_products_map[product] = producer_id
            self.lock_publish.release()
            return True
        else:
            return False

    def new_cart(self):
        """
        Atomically creates a new empty cart for a consumer.
        @return The new cart's unique ID.
        """
        self.lock_new_cart.acquire()
        self.consumer_cart_index += 1
        self.lock_new_cart.release()
        self.carts.append([])
        return self.consumer_cart_index

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified cart if the product is available.
        This involves moving the product from the public product list to the cart.
        @return True on success, False if the product is not available.
        """
        if product in self.products_list:
            self.products_list.remove(product)

            self.lock_add_cart.acquire()
            self.producer_queue_size_list[self.producers_to_products_map[product]] -= 1
            self.lock_add_cart.release()
            self.carts[cart_id].append(product)
            return True
        else:
            return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the public product list.
        """
        self.products_list.append(product)

        self.carts[cart_id].remove(product)
        self.lock_remove_cart.acquire()
        self.producer_queue_size_list[self.producers_to_products_map[product]] += 1
        self.lock_remove_cart.release()

    def place_order(self, cart_id):
        """Finalizes an order by returning the contents of the cart."""
        return self.carts[cart_id]


class Producer(Thread):
    """
    A worker thread that simulates a producer publishing products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer with its products and marketplace connection."""
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution loop for the producer.

        It registers with the marketplace and then enters an infinite loop, continuously
        trying to publish its products. It uses a retry-with-sleep mechanism if the
        marketplace's queue for this producer is full.
        """
        producer_id = self.marketplace.register_producer()
        while True:
            for (product, quantity, wait_time) in self.products:
                while quantity > 0:
                    status = self.marketplace.publish(producer_id, product)
                    # Block Logic: If publishing fails (e.g., queue is full), wait and retry.
                    while not status:
                        sleep(self.republish_wait_time)
                        status = self.marketplace.publish(producer_id, product)
                    if status:
                        quantity -= 1
                        sleep(wait_time)


class Consumer(Thread):
    """
    A worker thread that simulates a consumer purchasing products.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the consumer with a set of shopping carts to process."""
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution loop for the consumer.

        It processes each cart's operations (add/remove). For 'add' operations, it
        uses a retry-with-sleep mechanism if a product is not immediately available.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                if operation.get("type") == "add":
                    for _ in range(operation.get("quantity")):
                        status = self.marketplace.add_to_cart(cart_id, operation.get("product"))
                        # Block Logic: If adding fails (product unavailable), wait and retry.
                        while not status:
                            sleep(self.retry_wait_time)
                            status = self.marketplace.add_to_cart(cart_id, operation.get("product"))
                elif operation.get("type") == "remove":
                    for _ in range(operation.get("quantity")):
                        self.marketplace.remove_from_cart(cart_id, operation.get("product"))
            for product in self.marketplace.place_order(cart_id):
                print(self.kwargs.get("name") + " bought " + product.__str__())
