# -*- coding: utf-8 -*-
"""
@brief A simulation of a Producer-Consumer model using a central Marketplace.
@file consumer.py

This module simulates an e-commerce marketplace where Producer threads create
and publish products, and Consumer threads add products to carts and place orders.

WARNING:
This implementation is flawed and contains several critical concurrency issues and bugs,
making it unsuitable for production use.
1.  **Race Conditions:** The `Marketplace` class does not consistently use locks
    to protect shared resources like the `products` list. The `publish` method
    is not thread-safe at all, and other methods have partial or incorrect locking.
2.  **Logic Bug:** The `place_order` method contains a bug where it attempts to
    pop the same cart twice, resulting in lost data.
3.  **Inefficient Polling:** Consumers and Producers use `time.sleep` in a
    busy-wait loop to handle failed operations, which is inefficient. A proper
    synchronization mechanism like condition variables would be more suitable.
"""

from threading import Thread, Lock, currentThread
import time


class Consumer(Thread):
    """
    Represents a consumer that processes a list of shopping carts.

    Each consumer runs in its own thread, requests a new cart from the
    marketplace, performs a series of 'add' or 'remove' operations as defined
    in its workload, and finally places the order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of cart workloads. Each workload is a list of
                          operations (add/remove).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Seconds to wait before retrying a failed
                                     cart operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution loop for the consumer.

        Processes each cart in its assigned workload sequentially.
        """
        # Invariant: Process all assigned cart workloads.
        for cart in self.carts:
            # For each workload, a new cart is created in the marketplace.
            cart_id = self.marketplace.new_cart()

            # Pre-condition: Iterate through all operations for the current cart.
            for operation in cart:
                cart_operations = 0
                quantity = operation["quantity"]

                # Invariant: Execute the operation 'quantity' times.
                while cart_operations < quantity:
                    operation_name = operation["type"]
                    product = operation["product"]

                    # Block Logic: Dispatch to the appropriate marketplace method
                    # based on the operation type.
                    if operation_name == "add":
                        ret = self.marketplace.add_to_cart(cart_id, product)
                    elif operation_name == "remove":
                        ret = self.marketplace.remove_from_cart(cart_id, product)

                    # Pre-condition: Check if the operation was successful.
                    if ret is None or ret:
                        # If successful, increment the counter.
                        cart_operations += 1
                    else:
                        # If failed (e.g., product not available), wait and retry.
                        time.sleep(self.retry_wait_time)

            # After all operations, place the final order.
            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    A central marketplace that manages products and shopping carts.

    This class is the hub for all interactions between Producers and Consumers.
    It is responsible for tracking available products, managing active carts,
    and processing orders.

    NOTE: This class is not thread-safe.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have in the marketplace
                                           at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # Shared list of available products. Access is not properly synchronized.
        self.products = []
        # Dictionary to hold active shopping carts.
        self.carts = {}
        # Tracks the number of products each producer has published.
        self.nr_prod_in_queue = []
        # Maps a product to its original producer.
        self.products_owners = {}
        self.nr_carts = 0
        # Lock for the product list, but it is not used consistently.
        self.lock_products_queue = Lock()
        # Lock for the cart counter, used correctly.
        self.lock_nr_carts = Lock()

    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        # This part is not thread-safe. If multiple producers register at the
        # same time, it could lead to a race condition.
        producer_id = len(self.nr_prod_in_queue)
        self.nr_prod_in_queue.append(0)
        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        WARNING: This method is not thread-safe. Multiple producers calling
        this concurrently can lead to race conditions when modifying `self.products`
        and `self.nr_prod_in_queue`.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise
                  (e.g., if the producer's queue is full).
        """
        # Pre-condition: Check if the producer is below its publication limit.
        if self.nr_prod_in_queue[producer_id] < self.queue_size_per_producer:
            # RACE CONDITION: These operations should be protected by a lock.
            self.products.append(product)
            self.nr_prod_in_queue[producer_id] += 1
            self.products_owners[product] = producer_id
            return True
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its ID.

        This method is thread-safe.

        Returns:
            int: The ID of the newly created cart.
        """
        with self.lock_nr_carts:
            cart_id = self.nr_carts
            self.nr_carts += 1
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product from the marketplace to a specified shopping cart.

        This method is partially thread-safe but contributes to overall race
        conditions due to inconsistent locking strategy with other methods.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            bool: True if the product was successfully added, False if the
                  product was not available.
        """
        with self.lock_products_queue:
            # Pre-condition: Check if the product is available in the marketplace.
            if product in self.products:
                # Remove product from public inventory.
                self.products.remove(product)
                # Decrement the original producer's product count.
                producer_id = self.products_owners[product]
                self.nr_prod_in_queue[producer_id] -= 1
                # Add product to the consumer's private cart.
                self.carts[cart_id].append(product)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the marketplace.

        WARNING: This method has race conditions. The product is returned to the
        marketplace inventory outside of the lock that protects the producer count.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product: The product to remove.
        """
        # This operation is not thread-safe if a cart could be modified concurrently.
        self.carts[cart_id].remove(product)
        # RACE CONDITION: Product is added back to the list outside the lock.
        self.products.append(product)

        with self.lock_products_queue:
            # The producer's count is incremented under lock, but the list
            # append above was not, leading to an inconsistent state.
            producer_id = self.products_owners[product]
            self.nr_prod_in_queue[producer_id] += 1

    def place_order(self, cart_id):
        """
        Finalizes an order, printing the items "bought".

        BUG: This method pops the cart from `self.carts` twice. The second
        `pop` will always return None, so `order` is never correctly assigned.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            Should return the list of products, but returns None due to a bug.
        """
        products_list = self.carts.pop(cart_id, None)
        # BUG: This second pop will always find nothing, as the cart was just popped.
        order = self.carts.pop(cart_id, None)
        if products_list:
            for product in products_list:
                print(currentThread().getName(), "bought", product)
        return order


class Producer(Thread):
    """
    Represents a producer that creates products and publishes them.

    Each producer runs in its own thread and continuously tries to publish its
    assigned products to the marketplace, respecting the per-producer queue limits.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of products to be produced, where each
                             item is a tuple of (product, quantity, wait_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Seconds to wait before retrying to
                                         publish a product if the queue is full.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Registers itself with the marketplace to get a unique producer ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer.

        Continuously cycles through its product list and attempts to publish
        them to the marketplace.
        """
        # Invariant: The producer runs indefinitely.
        while True:
            # Block Logic: Iterate through all product types the producer is responsible for.
            for (product, quantity, wait_time) in self.products:
                added_products = 0
                # Invariant: Publish the specified 'quantity' of the current product.
                while added_products < quantity:
                    ret = self.marketplace.publish(self.producer_id, product)
                    # Pre-condition: Check if publication was successful.
                    if ret:
                        time.sleep(wait_time)
                        added_products += 1
                    else:
                        # If marketplace queue is full, wait before retrying.
                        time.sleep(self.republish_wait_time)