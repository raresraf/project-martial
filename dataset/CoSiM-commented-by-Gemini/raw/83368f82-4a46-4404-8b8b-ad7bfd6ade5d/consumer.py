"""
This module simulates a marketplace with producers and consumers. The code is a
concatenation of multiple files: consumer.py, marketplace.py, producer.py, and
product.py.

The architecture features a central `Marketplace` that holds a single global list
of all available products.

WARNING: This implementation has several significant concurrency flaws and bugs:
- Race Condition: The `publish` method is not protected by a lock, allowing
  multiple producers to modify shared data structures concurrently, which can
  lead to a corrupt state.
- Race Condition: The `add_to_cart` method releases its lock before the operation
  is fully complete (before adding the item to the cart), breaking atomicity.
- Bug: The `remove_from_cart` method does not correctly return a product to the
  original producer's queue count, leading to incorrect accounting of producer
  inventory limits.
"""

from threading import Thread, currentThread, Lock
from time import sleep
from dataclasses import dataclass

# --- Product Definitions ---

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """An immutable base class for a product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A tea-specific product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A coffee-specific product."""
    acidity: str
    roast_level: str

# --- Marketplace Class ---

class Marketplace:
    """
    The central marketplace, managing producers, consumers, and product inventory.
    This implementation uses a single global list for all products.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.all_carts = {}
        self.carts_id_lock = Lock()
        self.cart_id = -1
        self.producer_id = -1
        self.producer_id_lock = Lock()
        # A single flat list containing all products from all producers.
        self.products_in_marketplace = []
        # Dictionaries to track which products belong to which producer and their queue counts.
        self.producers_queues = {}
        self.producers_products = {}
        self.add_remove_lock = Lock()

    def register_producer(self):
        """Thread-safely registers a new producer and returns a unique ID."""
        with self.producer_id_lock:
            self.producer_id += 1
            new_id = self.producer_id
        
        self.producers_products[new_id] = []
        self.producers_queues[new_id] = 0
        return new_id

    def publish(self, producer_id, product):
        """
        Adds a product to the marketplace.
        WARNING: This method is NOT thread-safe. Multiple producers calling this
        concurrently will lead to race conditions when modifying the shared lists
        and dictionaries.
        """
        if self.producers_queues[int(producer_id)] < self.queue_size_per_producer:
            self.producers_queues[int(producer_id)] += 1
            self.products_in_marketplace.append(product)
            self.producers_products[int(producer_id)].append(product)
            return True
        return False

    def new_cart(self):
        """Thread-safely creates a new, empty cart for a consumer."""
        with self.carts_id_lock:
            self.cart_id += 1
            new_id = self.cart_id
        self.all_carts[new_id] = []
        return new_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.
        WARNING: This method is not fully atomic, releasing its lock before
        the operation is complete.
        """
        with self.add_remove_lock:
            if product in self.products_in_marketplace:
                self.products_in_marketplace.remove(product)

                # Inefficiently find which producer owned the product to decrement their queue count.
                for producer in self.producers_products:
                    if product in self.producers_products[producer]:
                        self.producers_queues[producer] -= 1
                        self.producers_products[producer].remove(product)
                        break
            else:
                return False
        
        # RACE CONDITION: The lock is released before this final modification.
        self.all_carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart.
        BUG: This method is flawed. It adds the product back to the global pool
        but fails to credit it back to the original producer's queue count.
        """
        self.all_carts[cart_id].remove(product)
        self.products_in_marketplace.append(product)

    def place_order(self, cart_id):
        """Finalizes the order and returns the products in the cart."""
        return self.all_carts[cart_id]

# --- Producer and Consumer Threads ---

class Producer(Thread):
    """A thread that simulates a producer, continuously trying to publish products."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main loop for the producer. It runs indefinitely, attempting to
        publish its products.
        """
        while True:
            for product_info in self.products:
                product_obj, quantity, production_time = product_info
                published_count = 0
                while published_count < quantity:
                    if self.marketplace.publish(str(self.producer_id), product_obj):
                        sleep(production_time)
                        published_count += 1
                    else:
                        # If the producer's queue is full, wait and retry.
                        sleep(self.republish_wait_time)

class Consumer(Thread):
    """A thread that simulates a consumer executing a list of shopping tasks."""
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts # A list of shopping lists (list of tasks).
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """Processes each shopping list, adding/removing items from the marketplace."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for action in cart:
                count = 0
                while count < action['quantity']:
                    if action['type'] == 'add':
                        # If adding to the cart fails, wait and retry.
                        if not self.marketplace.add_to_cart(cart_id, action['product']):
                            sleep(self.retry_wait_time)
                        else:
                            count += 1
                    elif action['type'] == 'remove':
                        self.marketplace.remove_from_cart(cart_id, action['product'])
                        count += 1

            products_in_cart = self.marketplace.place_order(cart_id)
            for product in products_in_cart:
                print(f"{currentThread().getName()} bought {product}")
