# -*- coding: utf-8 -*-
"""
A multi-threaded simulation of a producer-consumer marketplace.

This script models a simple e-commerce system with Producers creating products
and Consumers purchasing them through a central Marketplace.

WARNING: This implementation contains several concurrency issues. Shared resources in
the Marketplace class are not consistently protected by locks, leading to race
conditions and unpredictable behavior under concurrent access. It should not be
used as a model for a production system.
"""

from threading import Thread, Lock
from time import sleep
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a customer shopping.

    Each consumer processes a list of shopping operations (add/remove) for a
    series of carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping sessions, each with operations.
            marketplace (Marketplace): The central marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed 'add'.
            **kwargs: Thread-related arguments, including 'name'.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution logic for the consumer thread.
        Simulates creating a cart, adding/removing items, and placing an order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            # 'opp' is short for 'operation'
            for opp in cart:
                for i in range(0, opp["quantity"]):
                    if opp["type"] == "add":
                        # Block-Logic: Continuously try to add a product until successful.
                        # This is a busy-wait loop if the product is unavailable.
                        while self.marketplace.add_to_cart(cart_id, opp["product"]) == False:
                            sleep(self.retry_wait_time)
                    elif opp["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, opp["product"])

            prod_list = self.marketplace.place_order(cart_id)

            for product in prod_list:
                print(str(self.name) + " bought " + str(product))


class Marketplace:
    """
    The central marketplace hub. Manages producers, carts, and product flow.

    WARNING: The implementation of this class is not thread-safe. It uses
    insufficient and incorrect locking, which will lead to race conditions.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The max number of items per producer.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.consumer_id = 0
        # Inventory storage: dict mapping producer_id to a list of products.
        self.prod_dict = {}
        # Cart storage: dict mapping cart_id to a list of [product, producer_id] pairs.
        self.cart_dict = {}
        # These locks are used inconsistently and are insufficient to prevent race conditions.
        self.lock_add_cart = Lock()
        self.lock_publish = Lock()
        pass

    def register_producer(self):
        """
        Registers a new producer, returning a new producer ID.

        @warning Not thread-safe. Two producers calling this simultaneously
        could receive the same ID due to a race condition on `self.producer_id`.
        """
        self.producer_id += 1
        self.prod_dict[self.producer_id] = []
        return self.producer_id
        pass

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product.

        @warning The single lock `lock_publish` creates a global bottleneck for all
        producers, even though they are publishing to different inventory lists.
        """
        self.lock_publish.acquire()
        if len(self.prod_dict[producer_id]) < self.queue_size_per_producer:
            self.prod_dict[producer_id].append(product)
            self.lock_publish.release()
            return True
        self.lock_publish.release()
        return False
        pass

    def new_cart(self):
        """
        Creates a new cart for a consumer.

        @warning Not thread-safe. Two consumers calling this simultaneously
        could receive the same cart ID due to a race condition on `self.consumer_id`.
        """
        self.consumer_id += 1
        self.cart_dict[self.consumer_id] = []
        return self.consumer_id
        pass

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by searching all producer inventories.

        @warning This method is highly inefficient (O(N*M) search) and uses a single
        global lock, creating a major bottleneck for all consumers.
        """
        self.lock_add_cart.acquire()

        for prod_id in self.prod_dict.keys():
            # This inner loop is problematic; modifying a list while iterating
            # can lead to undefined behavior, although it may appear to work here.
            for p in self.prod_dict[prod_id]:
                if p == product:
                    self.prod_dict[prod_id].remove(product)
                    self.cart_dict[cart_id].append([product, prod_id])
                    self.lock_add_cart.release()
                    return True
        self.lock_add_cart.release()
        return False
        pass

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the producer's inventory.

        @warning CRITICAL: This method is not thread-safe. It reads from and writes to
        shared dictionaries and lists without acquiring any locks, which will
        corrupt the application's state under concurrent access.
        """
        for prod in self.cart_dict[cart_id]:
            if prod[0] == product:
                self.cart_dict[cart_id].remove(prod)
                self.prod_dict[prod[1]].append(prod[0])
                break

    def place_order(self, cart_id):
        """
        Finalizes an order by collecting all products from the cart.
        """
        prod_list = []
        for prod in self.cart_dict[cart_id]:
           prod_list.append(prod[0])
        return prod_list


class Producer(Thread):
    """
    Represents a producer thread that creates products and publishes them.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to produce.
            marketplace (Marketplace): The central marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Thread-related arguments.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Bug: self.name is assigned the entire kwargs dict, not the name string.
        self.name = kwargs

    def run(self):
        """
        The main execution logic for the producer thread.
        Produces items with a simulated delay and publishes them.
        """
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                # Simulate production time.
                sleep(product[2])
                for i in range(0, product[1]):
                    # Block-Logic: Keep trying to publish until successful.
                    while self.marketplace.publish(producer_id, product[0]) == False:
                        sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea as a specific type of Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee as a specific type of Product."""
    acidity: str
    roast_level: str
