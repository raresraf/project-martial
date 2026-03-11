"""
Models a marketplace simulation with producers and consumers using multithreading.

This script implements a producer-consumer pattern where 'Producer' threads generate
and publish products to a central 'Marketplace', and 'Consumer' threads browse
and purchase these products. The simulation handles concurrency for creating carts,
registering producers, and managing product inventory.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace to purchase products.

    Each consumer thread processes a list of shopping carts, where each cart contains
    a series of operations (add/remove) for different products.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of product operations.
            marketplace (Marketplace): The central marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying to add a product.
            **kwargs: Keyword arguments, including 'name' for the consumer.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution loop for the consumer thread.

        Iterates through its assigned carts, processes the operations for each,
        and places the final order.
        """
        # Invariant: The loop processes each cart sequentially for this consumer.
        for cart in self.carts:
            # Pre-condition: A new cart must be created in the marketplace for each shopping session.
            cart_id = self.marketplace.new_cart()

            # Invariant: Processes all product operations within a single cart.
            for product_info in cart:
                op = product_info['type']
                product = product_info['product']
                quantity = product_info['quantity']

                # Invariant: Ensures the specified quantity of a product is processed.
                while quantity > 0:
                    # Block Logic: Handles the addition of a product to the cart.
                    if op == "add":
                        # Attempt to add the product to the marketplace cart.
                        added = self.marketplace.add_to_cart(cart_id, product)

                        if added:
                            quantity = quantity - 1
                        else:
                            # Inline: If adding fails (e.g., product out of stock), wait before retrying.
                            time.sleep(self.retry_wait_time)

                    # Block Logic: Handles the removal of a product from the cart.
                    if op == "remove":
                        self.marketplace.remove_from_cart(cart_id, product)
                        quantity = quantity - 1
            
            # Finalizes the transaction for the current cart.
            products = self.marketplace.place_order(cart_id)
        
            # Post-condition: Prints out the products successfully purchased by the consumer.
            for prod in products:
                print(self.name + " bought " + str(prod))

import random
from threading import Lock

class Marketplace:
    """
    Manages the inventory, producers, and customer carts in a thread-safe manner.
    
    This class acts as the shared resource between Producer and Consumer threads,
    coordinating all interactions.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single producer can have in the marketplace at once.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.prod_seed = 0
        self.cart_seed = 0

        # Stores the list of registered producer IDs.
        self.producers = []

        # A dictionary mapping producer IDs to their list of available products.
        self.items_by_producers = {}

        # Stores all active shopping carts.
        self.carts = []

        # Locks to ensure thread-safe generation of unique producer and cart IDs.
        self.p_seed = Lock()
        self.c_seed = Lock()

    def register_producer(self):
        """
        Registers a new producer, providing a unique ID.

        Returns:
            str: A unique identifier for the new producer.
        """
        # Pre-condition: Acquire lock to ensure atomic producer ID generation.
        self.p_seed.acquire()

        random.seed(self.prod_seed)
        producer_id = random.randint(10000, 99999)
        self.prod_seed = self.prod_seed + 1

        self.p_seed.release()
        # Post-condition: Lock is released.

        # Initializes the data structures for the new producer.
        products = []
        self.items_by_producers[str(producer_id)] = products
        self.producers.append(str(producer_id)) 

        return str(producer_id)

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Args:
            producer_id (str): The ID of the producer publishing the item.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise (e.g., queue is full).
        """
        products = self.items_by_producers[producer_id]

        # Pre-condition: Checks if the producer's product queue has space.
        if len(self.items_by_producers[producer_id]) >= self.queue_size_per_producer:
            return False

        products.append(product)
        self.items_by_producers[producer_id] = products

        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its ID.

        Returns:
            int: The unique identifier for the newly created cart.
        """
        # Pre-condition: Acquire lock for thread-safe cart ID generation.
        self.c_seed.acquire()

        cart_id = self.cart_seed
        self.cart_seed = self.cart_seed + 1

        self.c_seed.release()
        # Post-condition: Lock is released.

        new_cart = []
        self.carts.append(new_cart)

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart if it's available.

        This involves finding the product in any producer's inventory and moving it to the cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was added, False if it was not found.
        """
        # Invariant: Scans through all producers to find the requested product.
        for producer_id in self.producers:
            for item in self.items_by_producers[producer_id]:
                if item == product:
                    # Block Logic: Moves the item from producer inventory to the consumer's cart.
                    self.items_by_producers[producer_id].remove(item)
                    self.carts[cart_id].append([product, producer_id])
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the original producer's inventory.

        Args:
            cart_id (int): The cart ID.
            product (Product): The product to remove.

        Returns:
            bool: True if the product was found and removed, False otherwise.
        """
        found = False
        # Invariant: Searches for the specified product within the cart.
        for prod in self.carts[cart_id]:
            if prod[0] == product:
                # Block Logic: If found, moves the product back to the producer's inventory.
                found = True
                put_back = prod[0]
                producer_id = prod[1]
                
                # Removes the product from the consumer's cart.
                self.carts[cart_id].remove(prod)

                # Restocks the product in the original producer's inventory list.
                self.items_by_producers[producer_id].append(put_back)

                break

        if not found:
            return False 
        return True

    def place_order(self, cart_id):
        """
        Finalizes the order for a cart, returning the list of purchased products.

        Args:
            cart_id (int): The cart to be finalized.

        Returns:
            list: A list of products that were in the cart.
        """
        list_of_products = []
        
        # Invariant: Iterates through the cart to compile the final list of products.
        for prod in self.carts[cart_id]:
            list_of_products.append(prod[0])

        return list_of_products


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
            products (list): A list of tuples (product, quantity, waiting_time) to be produced.
            marketplace (Marketplace): The central marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish if the queue is full.
            **kwargs: Additional keyword arguments.
        """
        Thread.__init__(self)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution loop for the producer thread.

        Registers itself with the marketplace and continuously produces and publishes products.
        """
        producer_id = self.marketplace.register_producer()

        # Invariant: The producer runs indefinitely, attempting to publish its products.
        while True:
            for (product, quantity, waiting_time) in self.products:
                # Invariant: Ensures the specified quantity of each product is published.
                while quantity > 0:
                    published = self.marketplace.publish(producer_id, product)

                    if published:
                        quantity = quantity - 1
                        # Inline: Simulates time taken to produce the next item.
                        time.sleep(waiting_time)
                    else:
                        # Inline: If publishing fails (queue full), waits before retrying.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A generic product with a name and a price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A specialization of Product representing Tea, with a 'type' attribute."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A specialization of Product representing Coffee, with acidity and roast level."""
    acidity: str
    roast_level: str
