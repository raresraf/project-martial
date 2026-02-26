"""
This module implements a thread-safe simulation of an e-commerce marketplace.

This is a well-structured producer-consumer model that correctly implements
synchronization to handle concurrent access to shared resources.

The main components are:
- `Marketplace`: A thread-safe central broker that manages all shared state,
  including product inventories and shopping carts. It uses a combination of
  global and fine-grained locks to ensure data integrity while allowing for
  concurrency.
- `Cart`: A helper class to represent a shopping cart, which correctly tracks
  not only the products but also their original producers.
- `Producer`: A thread that produces and publishes items to the marketplace.
- `Consumer`: A thread that simulates a customer's shopping activity.
"""


from __future__ import print_function
from threading import Thread, Lock
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that shops in the marketplace.

    The consumer is given a list of "carts", each representing a shopping
    session with a list of actions ('add' or 'remove'). It delegates all
    concurrent operations to the thread-safe `Marketplace`.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self)
        if 'name' in kwargs:
            self.name = kwargs['name']
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_to_cart(self, product, cart_id):
        """
        A helper method that persistently tries to add a product to a cart.
        
        If the marketplace cannot add the item (e.g., it's out of stock), this
        method will sleep and retry indefinitely until it succeeds.
        """
        if not self.marketplace.add_to_cart(cart_id, product):
            while True:
                sleep(self.retry_wait_time)
                if self.marketplace.add_to_cart(cart_id, product):
                    break

    def run(self):
        """The main execution loop for the consumer."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for action in cart:
                if action['type'] == 'add':
                    for _ in range(action['quantity']):
                        self.add_to_cart(action['product'], cart_id)
                else: # action['type'] == 'remove'
                    for _ in range(action['quantity']):
                        self.marketplace.remove_from_cart(cart_id, action['product'])

            # After all operations, place the order and print the items.
            for item in self.marketplace.place_order(cart_id):
                print("{} bought {}".format(self.name, item))


from threading import Lock


class Cart:
    """
    Represents a single shopping cart.

    This class is not thread-safe on its own but is designed to be managed
    exclusively by the thread-safe `Marketplace`. It importantly tracks the
    originating producer for each product, which is necessary for returning
    items to the correct inventory.
    """
    def __init__(self):
        self.products = []
        self.producer_ids = []

    def add_product(self, product, producer_id):
        """Adds a product and its producer's ID to the cart."""
        self.products.append(product)
        self.producer_ids.append(producer_id)

    def remove_product(self, product):
        """Removes a product and returns the ID of the producer it came from."""
        idx = self.products.index(product)
        producer_id = self.producer_ids[idx]
        del self.products[idx]
        del self.producer_ids[idx]
        return producer_id


class Marketplace:
    """
    A thread-safe marketplace that acts as a broker for producers and consumers.

    It uses a fine-grained locking strategy to protect shared data structures
    while allowing for high concurrency.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.carts = []
        self.carts_lock = Lock() # Lock for the global list of carts.

        self.product_queues = []   # List of product lists, one for each producer.
        self.products_locks = []   # List of locks, one for each product list.
        self.queues_lock = Lock()  # Lock for the global list of producers.

    def register_producer(self):
        """
        Thread-safely registers a new producer, creating their inventory list and lock.
        """
        with self.queues_lock:
            producer_id = len(self.product_queues)
            self.product_queues.append([])
            self.products_locks.append(Lock())
        return producer_id

    def publish(self, producer_id, product):
        """Thread-safely publishes a product for a specific producer."""
        producer_id = int(producer_id)
        # Acquire the specific lock for this producer's inventory.
        with self.products_locks[producer_id]:
            if len(self.product_queues[producer_id]) < self.queue_size_per_producer:
                self.product_queues[producer_id].append(product)
                return True
        return False

    def new_cart(self):
        """Thread-safely creates a new cart."""
        with self.carts_lock:
            cart_id = len(self.carts)
            self.carts.append(Cart())
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Thread-safely searches all producer inventories for a product and adds
        it to the cart if found.
        """
        added = False
        with self.queues_lock:
            q_num = len(self.product_queues)

        # Iterate through each producer to find the product.
        for i in range(q_num):
            with self.products_locks[i]: # Lock this producer's inventory.
                if product not in self.product_queues[i]:
                    continue
                # If found, move it from inventory to the cart.
                added = True
                self.carts[cart_id].add_product(product, i)
                self.product_queues[i].remove(product)
                break # Stop searching once found.
        return added

    def remove_from_cart(self, cart_id, product):
        """
        Thread-safely removes a product from a cart and returns it to the
        correct producer's inventory.
        """
        producer_id = self.carts[cart_id].remove_product(product)
        with self.products_locks[producer_id]:
            self.product_queues[producer_id].append(product)

    def place_order(self, cart_id):
        """Returns the final list of products in a cart."""
        return self.carts[cart_id].products


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that publishes a list of products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self)
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'daemon' in kwargs:
            self.daemon = kwargs['daemon']

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main execution loop for the producer."""
        while True:
            for prod in self.products:
                for _ in range(prod[1]):  # Publish `quantity` of the product.
                    # Persistently try to publish the product.
                    if not self.marketplace.publish(self.producer_id, prod[0]):
                        while True:
                            sleep(self.republish_wait_time)
                            if self.marketplace.publish(self.producer_id, prod):
                                break
                    sleep(prod[2])  # Wait for a specific time after each publication.