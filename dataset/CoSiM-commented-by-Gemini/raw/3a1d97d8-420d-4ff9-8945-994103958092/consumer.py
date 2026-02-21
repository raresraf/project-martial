"""A producer-consumer marketplace simulation with per-producer queues.

This module models an e-commerce marketplace where multiple Producer threads
and Consumer threads interact concurrently. The Marketplace is designed with a
separate, locked queue for each producer, which is a key architectural choice.

NOTE: The file contains multiple class definitions and local package imports
(e.g., `from tema.marketplace import Marketplace`), suggesting it was intended
to be part of a larger `tema` package. It is documented here as a single file.
"""

"""
Implements a multi-threaded producer-consumer simulation for a marketplace.

Note: This file is named `consumer.py` but contains the full simulation
logic, including the Marketplace and Producer classes.

This module defines a system with a fine-grained locking strategy where each
producer has its own dedicated product queue and lock within the marketplace.
This minimizes contention between producers but increases the search time for
consumers.
"""

import time
from threading import Thread, Lock
from tema.marketplace import Marketplace
from tema.product import Product


class Consumer(Thread):
    """A thread simulating a consumer that buys products."""

    def __init__(self,
                 carts: list,
                 marketplace: Marketplace,
                 retry_wait_time: int,
                 **kwargs) \
            :
        """Initializes a Consumer thread."""
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """Processes a list of shopping carts sequentially.

        For each cart, it performs the specified 'add' or 'remove' operations.
        'add' is a blocking operation that retries until the product is found.
        Finally, it places the order and prints the items.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for action in cart:
                type_ = action['type']
                product = action['product']
                qty = action['quantity']

                for _ in range(qty):
                    if type_ == 'add':
                        # Retry adding to cart until successful.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif type_ == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)

            order = self.marketplace.place_order(cart_id)

            for product in order:
                print(f'{self.name} bought {product}')


class Marketplace:
    """A thread-safe marketplace with per-producer product queues.

    This marketplace gives each producer a dedicated queue and lock. This allows
    producers to publish products without contending with each other, but it
    requires consumers to search through all producer queues to find an item.
    """

    def __init__(self, queue_size_per_producer: int):
        """Initializes the Marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        # Each producer gets a tuple of (product_list, lock).
        self.producer_queues = []
        self.consumer_carts = []
        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()

    def register_producer(self) -> int:
        """Registers a new producer and allocates a dedicated queue and lock."""
        with self.register_producer_lock:
            producer_id = len(self.producer_queues)
            self.producer_queues.append(([], Lock()))
        return producer_id

    def publish(self, producer_id: int, product: Product) -> bool:
        """Publishes a product to a specific producer's queue."""
        queue, lock = self.producer_queues[producer_id]
        with lock:
            if len(queue) >= self.queue_size_per_producer:
                return False  # Queue is full.
            queue.append(product)
        return True

    def new_cart(self) -> int:
        """Creates a new, empty cart for a consumer."""
        with self.new_cart_lock:
            cart_id = len(self.consumer_carts)
            self.consumer_carts.append([])
        return cart_id

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """Adds a product to a cart by searching all producer queues.

        This operation can be slow as it iterates through every producer's
        queue, acquiring and releasing each lock, until the product is found.
        """
        cart = self.consumer_carts[cart_id]

        for producer_id, (queue, lock) in enumerate(self.producer_queues):
            with lock:
                try:
                    queue.remove(product)
                except ValueError:
                    continue  # Product not in this queue.

            # Store the product and its original producer ID in the cart.
            cart.append((product, producer_id))
            return True

        return False  # Product not found in any producer queue.

    def remove_from_cart(self, cart_id: int, product: Product) -> bool:
        """Removes a product from a cart and returns it to its original producer."""
        cart = self.consumer_carts[cart_id]

        for i, (prod, producer_id) in enumerate(cart):
            if prod == product:
                del cart[i]
                # Return the product to the correct producer's queue.
                queue, lock = self.producer_queues[producer_id]
                with lock:
                    queue.append(prod)
                return True
        return False

    def place_order(self, cart_id) -> list:
        """Finalizes an order by returning the products in the cart."""
        cart = self.consumer_carts[cart_id]
        return [product for product, producer_id in cart]


class Producer(Thread):
    """A thread that produces products and publishes them to the marketplace."""

    def __init__(self,
                 products: list,
                 marketplace: Marketplace,
                 republish_wait_time: int,
                 **kwargs) \
            :
        """Initializes a Producer thread."""
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_ = self.marketplace.register_producer()

    def run(self):
        """Continuously produces items and tries to publish them.

        If publishing fails because the producer's queue is full, it waits
        and retries.
        """
        while True:
            for product, qty, wait_time in self.products:
                for _ in range(qty):
                    time.sleep(wait_time)
                    # Retry publishing until successful.
                    while not self.marketplace.publish(self.id_, product):
                        time.sleep(self.republish_wait_time)
