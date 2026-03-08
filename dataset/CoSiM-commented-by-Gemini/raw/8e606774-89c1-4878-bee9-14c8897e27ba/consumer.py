# -*- coding: utf-8 -*-
"""
This file contains a multi-threaded simulation of an e-commerce marketplace,
combining logic for consumers, producers, and the marketplace itself.
The code appears to be concatenated from multiple source files.
"""

# ==============================================================================
# Inferred file: consumer.py
# Contains the Consumer thread and the central Marketplace logic.
# ==============================================================================

from threading import Thread, currentThread, Lock
import time
import unittest
# The following modules are part of the other concatenated files.
# from producer import Producer
# from product import Product, Tea, Coffee


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a customer shopping in the marketplace.

    Each consumer processes a predefined list of shopping carts, where each cart
    contains a series of 'add' or 'remove' operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts to process. Each cart is a list of
                          operation dictionaries.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a
                                     failed operation.
            **kwargs: Arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def get_name(self):
        """Returns the name of the consumer thread."""
        return self.name

    def run(self):
        """
        The main execution loop for the consumer.

        Processes each cart sequentially, performing add/remove operations and
        finally placing the order.
        """
        for cart in self.carts:
            # Each shopping journey starts with a new cart in the marketplace.
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                quantity = 0
                # Retry operations until the desired quantity is met.
                while quantity < operation['quantity']:
                    if operation['type'] == 'add':
                        result = self.marketplace.add_to_cart(cart_id, operation['product'])
                    elif operation['type'] == 'remove':
                        result = self.marketplace.remove_from_cart(cart_id, operation['product'])

                    if result:
                        quantity += 1
                    else:
                        # If adding/removing fails (e.g., product unavailable), wait and retry.
                        time.sleep(self.retry_wait_time)

            # After filling the cart, place the order.
            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    The central marketplace that synchronizes producers and consumers.

    This class manages the inventory of available products, active shopping carts,
    and the producers. It uses a single global lock for most operations, which
    ensures thread safety but can be a performance bottleneck.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.num_producers = 0
        self.num_carts = 0
        self.num_items_producer = []  # Tracks item count for each producer.

        # NOTE: The design of `self.producers` is a potential flaw. As a dictionary,
        # it implies only one unique instance of a product can exist in the market
        # at any time, overwriting previous entries.
        self.producers = {}
        self.carts = {}

        self.lock = Lock()

    def register_producer(self):
        """Assigns a unique ID to a new producer."""
        with self.lock:
            producer_id = self.num_producers
            self.num_producers += 1
            self.num_items_producer.insert(producer_id, 0)
        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product in the marketplace.

        Returns:
            bool: True if publishing was successful, False if the producer's
                  personal queue limit has been reached.
        """
        casted_producer_int = int(producer_id)
        # Enforce the per-producer queue limit.
        if self.num_items_producer[casted_producer_int] >= self.queue_size_per_producer:
            return False

        # This logic is protected by the calling producer's own retry loop, not a lock.
        self.num_items_producer[casted_producer_int] += 1
        self.producers[product] = casted_producer_int
        return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its ID."""
        with self.lock:
            cart_id = self.num_carts
            self.num_carts += 1
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from the marketplace inventory to a consumer's cart.

        This is the primary producer-consumer handoff point.
        """
        with self.lock:
            if product not in self.producers:
                return False
            
            # Decrement the producer's item count and remove the product from public inventory.
            producer_id = self.producers.pop(product)
            self.num_items_producer[producer_id] -= 1

        # NOTE: The cart stores products and producer IDs in an interleaved list,
        # which is unconventional. A list of tuples would be clearer.
        self.carts[cart_id].extend([product, producer_id])
        return True

    def remove_from_cart(self, cart_id, product):
        """Moves a product from a consumer's cart back to the marketplace."""
        if product in self.carts[cart_id]:
            index = self.carts[cart_id].index(product)
            self.carts[cart_id].pop(index)  # Remove product
            producer_id = self.carts[cart_id].pop(index)  # Remove corresponding producer_id

            # Return the product to the producer's inventory count.
            with self.lock:
                self.producers[product] = producer_id
                self.num_items_producer[int(producer_id)] += 1
        return True # Returns True even if item was not in cart.

    def place_order(self, cart_id):
        """Finalizes an order, printing the purchased items."""
        product_list = self.carts.pop(cart_id)

        # The loop iterates through the (product, producer_id) pairs.
        for i in range(0, len(product_list), 2):
            print(f"{currentThread().get_name()} bought {product_list[i]}")
            # The corresponding producer's slot is now free, but this is not protected
            # by a lock, which could be a race condition.
            # self.num_items_producer[product_list[i + 1]] is implicitly assumed to be safe.
        return product_list


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    product = "Tea(name='Linden', price=9, type='Herbal')"
    product2 = "Tea(name='Linden', price=10, type='Herbal')"

    def setUp(self):
        self.marketplace = Marketplace(13)

    def test_publish_limit_fail(self):
        producer_id = self.marketplace.queue_size_per_producer - 1
        self.marketplace.num_items_producer = [0] * self.marketplace.queue_size_per_producer
        self.marketplace.num_items_producer[producer_id] = self.marketplace.queue_size_per_producer
        self.assertFalse(self.marketplace.publish(str(producer_id), self.product))

    def test_publish_limit_success(self):
        producer_id = self.marketplace.queue_size_per_producer - 1
        self.marketplace.num_items_producer = [0] * self.marketplace.queue_size_per_producer
        self.marketplace.num_items_producer[producer_id] = self.marketplace.queue_size_per_producer - 1
        self.assertTrue(self.marketplace.publish(str(producer_id), self.product))

    def test_add_to_cart_fail(self):
        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.product))

    def test_add_to_cart_success(self):
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(str(producer_id), self.product)
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.product))

    def test_new_cart(self):
        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, self.marketplace.num_carts - 1)
        self.assertIsInstance(self.marketplace.carts[cart_id], list)

    def test_remove_from_cart(self):
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(str(producer_id), self.product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.product)
        self.marketplace.remove_from_cart(cart_id, self.product)
        self.assertNotIn(self.product, self.marketplace.carts[cart_id])

# ==============================================================================
# Inferred file: producer.py
# ==============================================================================

class Producer(Thread):
    """
    Represents a producer thread that continuously creates products and
    publishes them to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products the producer can create. Each
                             entry is a tuple of (product_object, quantity, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the marketplace queue is full.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = -1

    def run(self):
        """
        The main execution loop for the producer.

        Registers with the marketplace and then enters an infinite loop to
        publish products.
        """
        self.producer_id = self.marketplace.register_producer()

        while True:
            for product_info in self.products:
                product, total_quantity, production_time = product_info
                produced_count = 0
                while produced_count < total_quantity:
                    # Attempt to publish the product.
                    if self.marketplace.publish(str(self.producer_id), product):
                        # If successful, wait for the "production time" and increment count.
                        time.sleep(production_time)
                        produced_count += 1
                    else:
                        # If unsuccessful (queue full), wait before retrying.
                        time.sleep(self.republish_wait_time)

# ==============================================================================
# Inferred file: product.py
# Contains dataclass definitions for products.
# ==============================================================================

from dataclasses import dataclass

@dataclass(frozen=True, order=False)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int

@dataclass(frozen=True, order=False)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str

@dataclass(frozen=True, order=False)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
