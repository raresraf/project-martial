# -*- coding: utf-8 -*-
"""
This module implements a producer-consumer simulation for an e-commerce marketplace.

It models a system with multiple producers (vendors) who publish products, and
multiple consumers (customers) who add products to carts and place orders. The
Marketplace class acts as the central, thread-safe intermediary for all
interactions.

Classes:
    Consumer: A thread representing a customer.
    Marketplace: The central marketplace that manages all operations.
    Producer: A thread representing a product vendor.
    Product, Tea, Coffee: Dataclasses for representing products.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of items to process.
            marketplace (Marketplace): The marketplace to interact with.
            retry_wait_time (float): The time to wait before retrying a failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main loop for the consumer.

        It processes each cart, adds or removes items, and finally places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for item in cart:
                quantity = 0
                # Retry logic for adding/removing items until the desired quantity is reached.
                while quantity < item["quantity"]:
                    if item["type"] == "add":
                        ver = self.marketplace.add_to_cart(cart_id, item["product"])
                    else:
                        ver = self.marketplace.remove_from_cart(cart_id, item["product"])

                    if ver:
                        quantity += 1
                    else:
                        # If the operation fails, wait before retrying.
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread

class Marketplace:
    """
    The central marketplace that manages producers, consumers, and products.

    This class is designed to be thread-safe, using locks to protect shared data.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        self.max_products_per_size = queue_size_per_producer
        self.carts = {}
        self.producers = {}
        self.reserved = {}

        self.id_cart = 0
        self.id_producer = 0

        self.lock_id_cart = Lock()
        self.lock_id_producer = Lock()
        self.lock_print = Lock()

    def register_producer(self):
        """
        Registers a new producer and returns a unique producer ID.
        """
        with self.lock_id_producer:
            self.id_producer += 1
            prod_id = self.id_producer

        self.producers[prod_id] = []
        return prod_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.
        """
        prod_id = int(producer_id)
        # Enforces a limit on the number of products per producer.
        if len(self.producers[prod_id]) >= self.max_products_per_size:
            return False

        self.producers[prod_id].append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty cart and returns its ID.
        """
        with self.lock_id_cart:
            self.id_cart += 1
            cart_id = self.id_cart
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified cart.

        This method moves the product from the producer's available stock
        to a reserved state, associated with the cart.
        """
        ver = False
        # Find the product in the producers' stock.
        for _ in self.producers:
            if product in self.producers[_]:
                ver = True
                key = _
                break

        if not ver:
            return False

        # Move the product from available to reserved.
        self.producers[key].remove(product)
        if key in self.reserved.keys():
            self.reserved[key].append(product)
        else:
            self.reserved[key] = []
            self.reserved[key].append(product)

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified cart.

        This method moves the product from the reserved state back to the
        producer's available stock.
        """
        ver = True
        # Find the product in the reserved items.
        for key in self.reserved:
            for cnt in self.reserved[key]:
                if cnt == product:
                    ver = False
                    rem = key
                    break
            if not ver:
                break

        self.carts[cart_id].remove(product)

        # Move the product from reserved back to available.
        self.producers[rem].append(product)
        self.reserved[rem].remove(product)
        return True

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This method prints the items bought and clears the cart.
        """
        res = []
        res.extend(self.carts[cart_id])
        del self.carts[cart_id]

        # Print the purchased items in a thread-safe manner.
        for cnt in res:
            with self.lock_print:
                print("{} bought {}".format(currentThread().getName(), cnt))
        return res


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer that creates and publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to be produced.
            marketplace (Marketplace): The marketplace to publish to.
            republish_wait_time (float): Time to wait before retrying to publish.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        The main loop for the producer.

        It continuously produces and publishes products to the marketplace.
        """
        while True:
            for(product, num_prod, wait_time) in self.products:
                for quantity in range(num_prod):
                    # Attempt to publish a product.
                    if self.marketplace.publish(str(self.prod_id), product):
                        time.sleep(wait_time)
                    else:
                        # If publishing fails (e.g., queue is full), wait and retry.
                        time.sleep(self.republish_wait_time)
                        quantity -= 1


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
