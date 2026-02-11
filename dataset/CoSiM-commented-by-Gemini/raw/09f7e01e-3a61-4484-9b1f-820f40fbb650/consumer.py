"""
This module simulates a multi-threaded producer-consumer marketplace.

- Producers publish products to the marketplace.
- Consumers add/remove products from carts and place orders.
- The Marketplace manages inventory, carts, and synchronization.

NOTE: This implementation contains several critical thread-safety issues (race conditions)
in the Marketplace class due to missing lock acquisitions.
"""

from __future__ import division
from threading import Thread, Lock, currentThread
import time
from dataclasses import dataclass


# --- Data Classes for Products ---

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A Tea product, inheriting from Product and adding a 'type'."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A Coffee product, inheriting from Product with acidity and roast level."""
    acidity: str
    roast_level: str


# --- Main Simulation Classes ---

class Consumer(Thread):
    """
    A thread that simulates a consumer creating carts and placing orders.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """Processes each assigned cart and its operations."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]
                # Add items to the cart, busy-waiting if a product is unavailable.
                if operation["type"] == "add":
                    for _ in range(quantity):
                        while not self.marketplace.add_to_cart(cart_id, operation["product"]):
                            time.sleep(self.retry_wait_time)
                # Remove items from the cart.
                if operation["type"] == "remove":
                    for _ in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, operation["product"])

            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    The central marketplace that manages producers, products, and carts.
    
    Acts as the shared resource between Producer and Consumer threads.
    Synchronization is attempted with a single coarse-grained lock, but it is
    not applied consistently, leading to race conditions.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.lock = Lock()
        self.cid = 0 # Counter for unique cart IDs.
        self.producer_items = []  # Tracks items per producer.
        self.products = []  # Global list of available products.
        self.carts = {}  # Stores contents of active carts.
        self.producers = {} # Maps a product back to its producer ID.

    def register_producer(self):
        """Registers a new producer, returning a unique producer ID."""
        with self.lock:
            prod_id = len(self.producer_items)
            self.producer_items.append(0)
            return prod_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.
        
        CRITICAL FLAW: This method is NOT thread-safe. It reads and writes
        shared lists and dictionaries without acquiring the lock, which will
        cause race conditions if multiple producers publish concurrently.
        """
        producer_id = int(producer_id)
        if self.producer_items[producer_id] >= self.queue_size_per_producer:
            return False

        self.producer_items[producer_id] += 1
        self.products.append(product)
        self.producers[product] = producer_id
        return True

    def new_cart(self):
        """Creates a new, empty cart and returns its unique ID."""
        with self.lock:
            self.cid += 1
            cart_id = self.cid
            self.carts[cart_id] = []
            return cart_id

    def add_to_cart(self, cart_id, product):
        """Moves a product from the marketplace inventory to a user's cart."""
        with self.lock:
            if product not in self.products:
                return False # Product is not available.

            self.producer_items[self.producers[product]] -= 1
            self.products.remove(product)
            self.carts[cart_id].append(product)
            return True

    def remove_from_cart(self, cart_id, product):
        """
        Moves a product from a cart back to the marketplace inventory.
        
        CRITICAL FLAW: This method is NOT fully thread-safe. It appends to the
        shared `self.products` list outside of the lock, creating a race condition.
        """
        self.carts[cart_id].remove(product)
        self.products.append(product) # <-- Race condition here.
        with self.lock:
            self.producer_items[self.producers[product]] += 1

    def place_order(self, cart_id):
        """Simulates placing an order by printing the cart's contents."""
        products_list = self.carts.get(cart_id)
        for product in products_list:
            with self.lock:
                print("{} bought {}".format(currentThread().getName(), product))
        return products_list


class Producer(Thread):
    """
    A thread that simulates a producer publishing a list of products.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        Continuously loops, attempting to publish its products to the marketplace.
        """
        while True:
            for product, quantity, wait_time in self.products:
                for _ in range(quantity):
                    # Busy-waits until the product can be published.
                    while not self.marketplace.publish(str(self.prod_id), product):
                        time.sleep(self.republish_wait_time)
                    time.sleep(wait_time)
