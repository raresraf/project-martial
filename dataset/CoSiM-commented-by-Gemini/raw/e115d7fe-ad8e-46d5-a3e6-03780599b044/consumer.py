"""
consumer.py

@brief A multithreaded simulation of a producer-consumer marketplace.
@description This module defines the core components for a marketplace simulation,
including Producer threads that publish products, Consumer threads that purchase them,
and a central Marketplace to facilitate the exchange. It uses threading to simulate
concurrent producers and consumers.
"""

from threading import Thread, Lock
import time
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer processes a list of carts, where each cart contains a series of
    actions (add or remove products).
    """

    name = None

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of
                          product operations.
            marketplace (Marketplace): The central marketplace object.
            retry_wait_time (float): Time in seconds to wait before retrying to
                                     add a product if it's not available.
            **kwargs: Keyword arguments, including 'name' for the consumer.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through its assigned carts, simulates adding/removing products,
        and finally places the order.
        """
        # Block Logic: Process each shopping cart assigned to this consumer.
        for cart in self.carts:
            # Pre-condition: A new, empty cart is created in the marketplace for the current shopping session.
            cart_id = self.marketplace.new_cart()
            for prod in cart:
                # Block Logic: Process either an 'add' or 'remove' operation for a product.
                if prod["type"] == "add":
                    # Invariant: Loop continues until the desired quantity of the product has been successfully added to the cart.
                    while prod["quantity"] > 0:
                        # Attempt to add one unit of the product to the cart.
                        check = self.marketplace.add_to_cart(cart_id, prod["product"])

                        if check:
                            # If successful, decrement the remaining quantity.
                            prod["quantity"] -= 1
                        else:
                            # If the product is not available, wait before retrying.
                            time.sleep(self.retry_wait_time)
                else: # type is 'remove'
                    # Invariant: Loop continues until the desired quantity has been removed.
                    while prod["quantity"] > 0:
                        self.marketplace.remove_from_cart(cart_id, prod["product"])
                        prod["quantity"] -= 1
            
            # Finalize the transaction for the current cart.
            cart_list = self.marketplace.place_order(cart_id)
            # Post-condition: Print the items bought in reverse order of how they were placed.
            cart_list.reverse()
            for elem in cart_list:
                print(self.name + " bought " + str(elem))


class Marketplace:
    """
    Manages the interactions between producers and consumers.

    This class holds the state of all products available for sale and all
    active consumer shopping carts. It provides methods for producers to
    publish products and for consumers to create carts and purchase items.
    """
    
    # Static-like counters for unique producer and cart IDs.
    prod_id = 0
    cart_id = 0

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                                           single producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.prod_dict = {}  # Maps producer_id to a list of their products.
        self.cart_dict = {}  # Maps cart_id to a list of products in the cart.
        self.lock = Lock()   # A lock for synchronizing access, though its usage is flawed.

    def register_producer(self):
        """
        Assigns a new unique ID to a producer.

        Returns:
            int: The new producer ID.
        """
        self.prod_id += 1
        self.prod_dict[self.prod_id] = []
        return self.prod_id

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product for sale.

        The operation succeeds only if the producer's product queue is not full.

        Args:
            producer_id (int): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        if len(self.prod_dict[producer_id]) < self.queue_size_per_producer:
            self.prod_dict[producer_id].append(product)
            return True
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The new unique cart ID.
        """
        self.cart_id += 1
        self.cart_dict[self.cart_id] = []
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's shopping cart if it's available.

        Scans through all producers' inventories to find the requested product.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        # Block Logic: Search all producer inventories for the desired product.
        for key in self.prod_dict:
            # Inline: A lock is acquired here to attempt thread-safe reading of the producer's inventory.
            self.lock.acquire()
            for prod in self.prod_dict[key]:
                if product == prod:
                    self.cart_dict[cart_id].append(product)
                    # NOTE: The implementation has a critical flaw where it returns while holding the lock.
                    self.lock.release()
                    return True
            # Inline: The lock is released if the product is not found in the current producer's inventory.
            self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's shopping cart.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Product): The product to remove.
        """
        self.cart_dict[cart_id].remove(product)   

    def place_order(self, cart_id):
        """
        Finalizes an order, removing purchased items from producers' inventories.
        
        Note: This method lacks synchronization and is subject to race conditions in a
        multithreaded context.

        Args:
            cart_id (int): The ID of the cart to be processed.

        Returns:
            list: The list of products that were in the cart.
        """
        # Block Logic: For each item in the consumer's cart, find and remove it from the corresponding producer's inventory.
        for prod in self.cart_dict[cart_id]:
            for key in self.prod_dict:
                for product in self.prod_dict[key]:
                    if product == prod:
                        self.prod_dict[key].remove(product)

        return self.cart_dict[cart_id]


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of tuples, where each tuple contains a product,
                             its quantity, and the time to wait after publishing it.
            marketplace (Marketplace): The central marketplace object.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the marketplace queue is full.
            **kwargs: Keyword arguments, including 'daemon' flag.
        """
        Thread.__init__(self, daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and then enters an infinite loop to
        continuously publish its products.
        """
        prod_id = self.marketplace.register_producer()
        # Invariant: The producer will continuously try to keep its products in stock.
        while True:
            # Block Logic: Iterate through all products this producer is responsible for.
            for prod in self.products:
                quantity = prod[1]
                # Invariant: Publish the specified quantity of the current product.
                while quantity > 0:
                    # Attempt to publish one unit of the product.
                    check = self.marketplace.publish(prod_id, prod[0])

                    if check:
                        # If successful, decrement quantity and wait before next action.
                        quantity -= 1
                        time.sleep(prod[2])
                    else:
                        # If the producer's queue is full, wait before retrying.
                        time.sleep(self.republish_wait_time)


# --- Data classes for products ---

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A product of type Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A product of type Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
