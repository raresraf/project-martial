# -*- coding: utf-8 -*-
"""
A multi-threaded simulation of a producer-consumer marketplace.

This module defines three classes:
- Marketplace: The central shared resource where products are published and purchased.
- Producer: A thread that creates products and adds them to the marketplace.
- Consumer: A thread that simulates a user adding products to a cart and buying them.

The simulation uses locks to ensure thread-safe operations on the shared
marketplace data.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    A thread that simulates a consumer's shopping activity.

    Each consumer thread processes a list of "carts", where each cart contains
    a series of operations ('add' or 'remove') to be performed.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of cart data structures. Each cart is a list
                of dictionaries, with each dictionary representing an operation.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Seconds to wait before retrying a failed
                operation (e.g., adding an unavailable product).
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

        # Map operation names to marketplace methods for easy dispatch.
        self.actions = {
            'add': self.marketplace.add_to_cart,
            'remove': self.marketplace.remove_from_cart
        }

    def run(self):
        """The main execution logic for the consumer thread."""
        # Process each shopping journey (cart) assigned to this consumer.
        for cart in self.carts:
            # Each consumer gets a new, unique cart from the marketplace.
            cart_id = self.marketplace.new_cart()

            # Execute all operations within the current cart.
            for operation in cart:
                iters = 0

                # Perform the operation 'quantity' times (e.g., add 3 apples).
                while iters < operation['quantity']:
                    # Dynamically call the appropriate marketplace action ('add' or 'remove').
                    ret = self.actions[operation['type']](
                        cart_id, operation['product'])

                    # If the operation was successful, move to the next iteration.
                    # `ret is None` handles the 'remove' case which doesn't return a bool.
                    if ret or ret is None:
                        iters += 1
                    else:
                        # If adding failed (e.g., product unavailable), wait and retry.
                        time.sleep(self.retry_wait_time)

            # After all operations, finalize the purchase.
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock


class Marketplace:
    """
    Represents the shared marketplace resource.

    Manages product inventory, producer registration, and customer carts in a
    thread-safe manner using locks.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                producer can have listed in the marketplace at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_queues = []  # Tracks current publication count for each producer.
        self.all_products = []  # A simple list acting as the central product inventory.
        self.producted_by = dict()  # Maps a product to its producer's ID.
        self.no_carts = 0  # A counter to generate unique cart IDs.
        self.carts = dict()  # Stores the contents of active shopping carts.

        # Locks to prevent race conditions on shared data structures.
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.cart_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.producer_lock:
            # The ID is the producer's index in the queues list.
            producer_id = len(self.producer_queues)
            self.producer_queues.append(0)
            return producer_id

    def publish(self, producer_id, product):
        """
        Adds a product from a given producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (any): The product to be added.

        Returns:
            bool: True if publishing was successful, False otherwise (e.g., queue is full).
        """
        # A producer cannot exceed their publication quota.
        if self.producer_queues[producer_id] >= self.queue_size_per_producer:
            return False

        # Update tracking information for the new product.
        self.producer_queues[producer_id] += 1
        self.producted_by[product] = producer_id
        self.all_products.append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its ID.

        Returns:
            int: The unique ID for the newly created cart.
        """
        with self.consumer_lock:
            # Generate a new cart ID and increment the global counter.
            cart_id = self.no_carts
            self.no_carts += 1

        self.carts.setdefault(cart_id, [])
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart if available.

        This is a critical section that modifies both product and cart inventories.

        Args:
            cart_id (int): The ID of the cart to add to.
            product (any): The product to add.

        Returns:
            bool: True if the product was added successfully, False if not available.
        """
        with self.cart_lock:
            # The product must exist in the marketplace to be added to a cart.
            if product not in self.all_products:
                return False

            # The product is now "claimed" by the cart.
            # Decrement the producer's queue count and remove from general inventory.
            self.producer_queues[self.producted_by[product]] -= 1
            self.all_products.remove(product)

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart, returning it to the marketplace.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product (any): The product to remove.
        """
        self.carts[cart_id].remove(product)

        # Make the product available again in the marketplace.
        self.all_products.append(product)
        self.producer_queues[self.producted_by[product]] += 1

    def place_order(self, cart_id):
        """
        Finalizes an order by "checking out" a cart.

        This removes the cart from the system and prints the items purchased.

        Args:
            cart_id (int): The ID of the cart to be checked out.

        Returns:
            list: The list of products that were in the cart, or None if cart was empty.
        """
        # Atomically remove the cart to prevent further modifications.
        products = self.carts.pop(cart_id, None)

        if products:
            for product in products:
                print(f'{currentThread().getName()} bought {product}')

        return products


from threading import Thread
import time


class Producer(Thread):
    """A thread that produces and publishes items to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains
                (product_name, quantity_to_produce, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Seconds to wait before retrying to
                publish a product if the queue is full.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        # Each producer must register to get a unique ID.
        self.own_id = marketplace.register_producer()

    def run(self):
        """The main execution logic for the producer thread."""
        # The producer runs indefinitely, trying to keep the marketplace stocked.
        while True:
            # Iterate through the catalog of products this producer can make.
            for (product, no_products, wait_time) in self.products:
                i = 0
                # Produce the specified quantity of the current product.
                while i < no_products:
                    # Attempt to publish the product to the marketplace.
                    if self.marketplace.publish(self.own_id, product):
                        # If successful, wait for the "production time" and move to the next.
                        time.sleep(wait_time)
                        i += 1
                    else:
                        # If publishing fails (queue full), wait before retrying.
                        time.sleep(self.republish_wait_time)
