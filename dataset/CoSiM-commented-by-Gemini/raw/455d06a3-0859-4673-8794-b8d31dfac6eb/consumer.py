"""
This module provides another implementation of a multi-threaded producer-consumer
simulation for an e-commerce marketplace.

This version features a more complex internal implementation of the Marketplace,
with different data structures and a more fine-grained locking strategy
compared to other similar simulations. It models producers publishing products
and consumers adding them to carts.
"""

import sys
import time
from threading import Lock, Thread, currentThread


class Consumer(Thread):
    """A thread that simulates a consumer placing orders in the marketplace."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists, where each is a sequence of
                          add/remove operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed action.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        remove_from_cart = "remove"
        add_to_cart = "add"

        # Maps operation types to the corresponding marketplace methods.
        self.cart_actions = {remove_from_cart: self.marketplace.remove_from_cart,
                             add_to_cart: self.marketplace.add_to_cart}

    def run(self):
        """The main loop for the consumer.

        Processes each shopping list ('cart') assigned to it, executing the
        add/remove operations and then placing the final order.
        """
        for cart in self.carts:
            id_of_cart = self.marketplace.new_cart()
            for action in cart:
                index = 0
                action_quantity = action["quantity"]
                while index < action_quantity:
                    action_type = action["type"]
                    action_product = action["product"]
                    # Attempt the action (e.g., add or remove from cart).
                    result = self.cart_actions[action_type](id_of_cart, action_product)

                    if result is False:
                        # If the action fails, wait before retrying.
                        time.sleep(self.retry_wait_time)
                    elif result is True or result is None:
                        # If successful, move to the next item quantity.
                        index += 1

            self.marketplace.place_order(id_of_cart)


class Marketplace:
    """A thread-safe marketplace with a complex internal state.

    This implementation uses multiple locks and a nested list structure for
    products, which makes its state management intricate. It has several
    inefficiencies and potential race conditions.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): Max items a producer can publish.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # `products` is a list of lists, indexed by producer_id.
        self.products = []
        # `carts` is a dict mapping cart_id to a list of products.
        self.carts = {}
        # `map_products_to_producer` seems to map a product to its producer.
        # This could be problematic if products are not unique across producers.
        self.map_products_to_producer = {}
        self.register_lock = Lock()
        self.new_cart_lock = Lock()
        self.products_lock = Lock()
        self.final_lock = Lock()
        self.cart_id = 0

    def register_producer(self):
        """Registers a new producer, giving them a slot in the products list."""
        with self.register_lock:
            producer_id = len(self.products)
            self.products.append([])
        return producer_id

    def publish(self, producer_id, product):
        """Publishes a product from a specific producer."""
        with self.products_lock:
            if len(self.products[producer_id]) >= self.queue_size_per_producer:
                return False  # Capacity full for this producer.
            self.products[producer_id].append(product)
        # This mapping is not thread-safe and assumes product names are unique.
        self.map_products_to_producer[product] = producer_id
        return True

    def new_cart(self):
        """Creates a new cart ID and entry."""
        with self.new_cart_lock:
            self.cart_id += 1
        self.carts[self.cart_id] = []
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """Adds a product to a cart.

        Note: This method is highly inefficient due to the list comprehension
        used to check for product existence, which runs on every call. It also
        contains race conditions due to unsafe modification of the carts list.
        """
        with self.products_lock:
            # This check is O(N*M) where N is producers and M is products per producer.
            if product not in [j for i in self.products for j in i]:
                return False
            if product in self.map_products_to_producer.keys():
                producer_id = self.map_products_to_producer[product]
                if product in self.products[producer_id]:
                    self.products[producer_id].remove(product)
        # This modification is outside the lock, which is unsafe.
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace."""
        with self.products_lock:
            # These modifications are not safe if multiple consumers access the same cart.
            self.carts[cart_id].remove(product)
            producer_id = self.map_products_to_producer[product]
            self.products[producer_id].append(product)

    def place_order(self, cart_id):
        """Finalizes and prints an order.

        Note: Locking per-product for printing can lead to interleaved output
        if multiple orders are placed at the same time.
        """
        for product in self.carts[cart_id]:
            with self.final_lock:
                print(currentThread().getName(), "bought", product)
        return self.carts[cart_id]


class Producer(Thread):
    """A thread that simulates a producer publishing products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread.

        Args:
            products (list): A list of (product, quantity, wait_time) tuples.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main loop for the producer, which runs indefinitely."""
        while True:
            for product_data in self.products:
                product_type = product_data[0]
                count_prod = product_data[1]
                wait_time = product_data[2]
                index = 0
                while index < count_prod:
                    # Attempt to publish the product.
                    result = self.marketplace.publish(self.producer_id, product_type)

                    if result is False:
                        # If marketplace is full for this producer, wait.
                        time.sleep(self.republish_wait_time)
                    else:
                        # On success, wait for the specified time and move to the next item.
                        time.sleep(wait_time)
                        index += 1
