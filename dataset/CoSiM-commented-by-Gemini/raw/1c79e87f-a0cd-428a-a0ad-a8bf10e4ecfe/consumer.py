"""
This module implements a producer-consumer simulation of an e-commerce marketplace.

It defines `Producer` and `Consumer` threads that interact with a central `Marketplace`
class. Producers publish products, and Consumers add products to carts and place orders.
The simulation uses threading to model concurrent access to the marketplace.
"""

from threading import Thread, Semaphore
import time
from collections import defaultdict
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.

    Each consumer runs in its own thread, processing a list of shopping carts. Each
    cart contains a series of actions (add/remove products).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of actions.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a
                                     failed action (e.g., adding a product).
            **kwargs: Arguments for the parent Thread class, including 'name'.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self.name = kwargs['name']

    def run(self):
        """
        The main execution loop for the consumer thread.

        Processes each cart, performs the add/remove actions, and places the order.
        """
        # Block Logic: Process each shopping cart assigned to this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            # Block Logic: Execute the sequence of actions for the current cart.
            for param in cart:
                if param["type"] == "add":
                    # Attempt to add the specified quantity of a product to the cart.
                    i = 0
                    while i < param["quantity"]:
                        response = self.marketplace.add_to_cart(cart_id, param["product"])

                        # Invariant: If adding to the cart fails (e.g., product is out of
                        # stock or taken by another consumer), wait and retry.
                        while not response:
                            time.sleep(self.retry_wait_time)
                            response = self.marketplace.add_to_cart(cart_id, param["product"])
                        i += 1
                elif param["type"] == "remove":
                    # Remove the specified quantity of a product from the cart.
                    for i in range(param["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, param["product"])
            
            # Finalize the transaction for the current cart.
            checkout = self.marketplace.place_order(cart_id)
            for i in checkout:
                print(self.name + " bought " + str(i))


class Marketplace:
    """
    The central marketplace where producers and consumers interact.

    It manages product inventory from multiple producers and processes consumer
    shopping carts. Note: This class has significant thread-safety issues, as
    access to the core `products` and `consumers` dictionaries is not
    consistently protected by locks, leading to potential race conditions.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                                           single producer can have listed.
        """
        # Maps producer_id to a list of their published products.
        self.products = defaultdict(list)
        # Maps cart_id to a list of products in a consumer's cart.
        self.consumers = defaultdict(list)
        self.producer_id = 0
        self.consumer_id = 0
        # Semaphore to protect the producer_id counter during registration.
        self.producer_lock = Semaphore(1)
        # Semaphore to protect the consumer_id counter during cart creation.
        self.consumer_lock = Semaphore(1)
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        Assigns a unique ID to a new producer.

        Returns:
            int: The unique producer ID.
        """
        self.producer_lock.acquire()
        id_producer = self.producer_id
        self.products[id_producer] = []
        self.producer_id += 1
        self.producer_lock.release()
        return id_producer

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product in the marketplace.

        Args:
            producer_id (int): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if publishing was successful, False if the producer's
                  queue was full.
        """
        products_list = self.products.get(producer_id)
        length = len(products_list)
        # Enforces the per-producer queue size limit.
        if length < self.queue_size_per_producer:
            self.products[producer_id].append(product)
            return True
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its ID.

        Returns:
            int: The unique cart ID.
        """
        self.consumer_lock.acquire()
        cart_id = self.consumer_id
        self.consumers[cart_id] = []
        self.consumer_id += 1
        self.consumer_lock.release()
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart by taking it from a producer's inventory.

        This method is not thread-safe. It iterates through producer inventories and
        modifies them without a lock, creating a race condition.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        # Block Logic: Search all producer inventories for the desired product.
        for key, value in self.products.items():
            if product in value:
                # Operation is not atomic: another thread could modify `value`
                # between the `if` check and the `remove` call.
                self.consumers[cart_id].append((product, key))
                product_list = self.products.get(key)
                product_list.remove(product)
                self.products[key] = product_list
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the producer's inventory.

        This method is not thread-safe and is vulnerable to race conditions.
        """
        # Block Logic: Find the specified product in the cart.
        for key, value in self.consumers.items():
            if key == cart_id:
                for prod in self.consumers.get(key):
                    if product == prod[0]:
                        # Operation is not atomic.
                        self.consumers[cart_id].remove(prod)
                        # Return the product to the original producer's inventory.
                        self.products[prod[1]].append(product)
                        return

    def place_order(self, cart_id):
        """
        Finalizes an order by preparing a list of items and clearing the cart.
        """
        order_list = []
        for key, value in self.consumers.items():
            if key == cart_id:
                for prod in value:
                    order_list.append(prod[0])

        # Remove the cart from the marketplace. This operation is not thread-safe.
        new_dict = {key: val for key, val in self.consumers.items() if key != cart_id}
        self.consumers = new_dict
        return order_list


class Producer(Thread):
    """
    Represents a producer that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains a product,
                             quantity, and publishing interval.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the queue is full.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.producer_id = 0

    def run(self):
        """
        The main execution loop for the producer thread.
        """
        self.producer_id = self.marketplace.register_producer()
        # This thread runs indefinitely, continuously publishing products.
        while True:
            for entry in self.products:
                # Publish the specified quantity of a product.
                for i in range(entry[1]):
                    response = self.marketplace.publish(self.producer_id, entry[0])

                    # Invariant: If the producer's queue in the marketplace is full,
                    # wait and retry until publishing succeeds.
                    while not response:
                        time.sleep(self.republish_wait_time)
                        response = self.marketplace.publish(self.producer_id, entry[0])

                    # Wait for a product-specific interval before the next publish.
                    time.sleep(entry[2])


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base data class for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class for a Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class for a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
