"""
A simulation of a marketplace with producers and consumers running in concurrent threads.

This script models a basic e-commerce system where Producer threads generate and
publish products, and Consumer threads browse and purchase these products. The
Marketplace class acts as the central, thread-safe intermediary for all operations,
using locks to manage concurrent access to shared inventory and carts.
"""

from threading import Thread, Lock, currentThread
import time
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer that processes a list of shopping carts.

    Each consumer runs in its own thread, sequentially processing a predefined
    list of carts. For each cart, it adds or removes products as specified,
    with a retry mechanism for when products are unavailable.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of product operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (int): Seconds to wait before retrying a failed product operation.
            **kwargs: Keyword arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts 
        self.marketplace = marketplace 
        self.retry_wait_time = retry_wait_time


    def run(self):
        """
        The main execution loop for the consumer thread.

        Iterates through each assigned cart, creates it in the marketplace,
        processes all product additions/removals, and finally places the order.
        """
        # Invariant: Processes one cart from the consumer's list per iteration.
        for cart in self.carts:
            # Pre-condition: A new, empty cart is created in the marketplace for the current order.
            cart_id = self.marketplace.new_cart()
            # Invariant: Processes one product operation (add/remove) for the current cart.
            for products in cart:
                now_quantity = 0
                # Pre-condition: Loop until the desired quantity of a product is successfully added or removed.
                while now_quantity < products["quantity"]:
                    if products["type"] == "add":
                        # Attempt to add the product to the cart.
                        check = self.marketplace.add_to_cart(cart_id, products["product"])
                    if products["type"] == "remove":
                        # Attempt to remove the product from the cart.
                        check = self.marketplace.remove_from_cart(cart_id, products["product"])
                    
                    # If the operation failed (e.g., product unavailable), wait and retry.
                    if check is False:
                        time.sleep(self.retry_wait_time)
                    else:
                        now_quantity += 1
            # Once all operations for the cart are complete, place the order.
            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    A thread-safe marketplace that manages producers, products, and customer carts.

    This class is the central hub for all interactions. It uses locks to ensure
    that concurrent operations from multiple producers and consumers do not lead
    to race conditions or inconsistent state.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.no_of_producers = 0
        self.producers = {}  # Tracks products counts per producer_id
        self.no_of_carts = 0
        self.carts = {}  # Stores products in each active cart
        self.producers_products = {}  # Maps a product to its producer_id
        self.available_products = []  # List of all products currently for sale
        
        # Lock for safely registering new producers.
        self.lock_reg_producers = Lock() 
        # Lock for safely creating new carts.
        self.lock_carts = Lock() 
        # Lock for all product inventory operations (publishing, adding/removing from cart).
        self.lock_producers = Lock() 
        
    def register_producer(self):
        """
        Allocates a unique ID for a new producer.

        Uses a lock to prevent race conditions when multiple producers register simultaneously.

        Returns:
            int: The unique ID assigned to the producer.
        """
        self.lock_reg_producers.acquire()
        self.no_of_producers += 1
        producer_id = self.no_of_producers

        # Initialize the producer's published product count to zero.
        self.producers[producer_id] = 0
        self.lock_reg_producers.release()
        return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.

        The operation is rejected if the producer is already at its maximum
        product capacity.

        Args:
            producer_id (str): The ID of the publishing producer.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Pre-condition: Check if the producer is allowed to publish more products.
        if self.producers[int(producer_id)] >= self.queue_size_per_producer:
            return False

        # This section should ideally be locked if multiple threads could publish
        # for the same producer_id, though the design implies one thread per producer.
        self.producers[int(producer_id)] += 1
        self.producers_products[product] = int(producer_id)
        self.available_products.append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Uses a lock to prevent race conditions when multiple consumers request carts.

        Returns:
            int: The unique ID for the new cart.
        """
        self.lock_carts.acquire()
        self.no_of_carts += 1
        cart_id = self.no_of_carts
        self.carts[cart_id] = []
        self.lock_carts.release()
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.

        This is a critical section that locks the producer/product inventory. It atomically
        checks for product availability, removes the product from the public pool,
        and adds it to the consumer's private cart.

        Args:
            cart_id (int): The target cart ID.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False if it was not available.
        """
        self.lock_producers.acquire()
        # Pre-condition: Check if the desired product is in the available list.
        if product not in self.available_products:
            self.lock_producers.release()
            return False

        # Identify the producer of the product.
        prod_id = self.producers_products[product]

        # Decrement the producer's active product count and move the product.
        self.producers[prod_id] -= 1
        self.available_products.remove(product)
        self.carts[cart_id].append(product)
        self.lock_producers.release()
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart, returning it to the available product pool.

        Args:
            cart_id (int): The cart from which to remove the product.
            product (Product): The product to remove.
        """
        # This operation is not fully thread-safe if cart_id could be accessed
        # by another thread simultaneously (e.g., in place_order).
        self.carts[cart_id].remove(product)
        self.available_products.append(product)
        
        # Lock to safely update the producer's product count.
        self.lock_producers.acquire()
        self.producers[self.producers_products[product]] += 1
        self.lock_producers.release()

    def place_order(self, cart_id):
        """
        Finalizes an order, printing the items bought.

        Removes the cart from the active list.

        Args:
            cart_id (int): The ID of the cart to be processed.
        """
        # The `pop` operation on a dictionary is atomic.
        prod_list = self.carts.pop(cart_id)
        for product in prod_list:
            print(f"{currentThread().getName()} bought {product}")


class Producer(Thread):
    """
    Represents a producer that generates and publishes products to the marketplace.

    Each producer runs in its own thread, continuously attempting to publish
    products from a given list according to specified quantities and frequencies.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, each containing (product, quantity, publish_interval).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (int): Seconds to wait before retrying to publish
                                       if the marketplace queue is full.
            **kwargs: Keyword arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products 
        self.marketplace = marketplace 
        self.republish_wait_time = republish_wait_time
        # Register with the marketplace to get a unique ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread.

        Continuously loops through its product list, attempting to publish each
        one until its specified quantity is met.
        """
        while True:
            # Invariant: Iterates through the list of products this producer is responsible for.
            for sublist in self.products:
                count = 0
                # Pre-condition: Loop until the target quantity for the sublist's product is published.
                while count < sublist[1]:
                    # Attempt to publish one item.
                    check = self.marketplace.publish(str(self.producer_id), sublist[0])
                    if check:
                        # Success: wait for the specified interval before publishing the next item.
                        time.sleep(sublist[2])
                        count += 1
                    else:
                        # Failure (marketplace queue full): wait before retrying.
                        time.sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product and adding a 'type'."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    A dataclass representing Coffee, inheriting from Product and adding
    acidity and roast level attributes.
    """
    acidity: str
    roast_level: str
