# -*- coding: utf-8 -*-
"""
This module provides a simulation of a multi-threaded producer-consumer
marketplace.

It defines the core classes for the simulation:
- Consumer: A thread that simulates a customer adding items to a shopping cart
  and placing an order.
- Producer: A thread that simulates a producer publishing products to the
  marketplace.
- Marketplace: The central, thread-safe class that orchestrates all
  interactions, managing inventory, producers, and carts using locks to ensure
  data consistency under concurrent access.
- Product, Tea, Coffee: Dataclasses for representing items sold in the
  marketplace.
- TestMarketplace: Unit tests to verify the functionality of the Marketplace.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that performs shopping operations in the marketplace.

    A consumer is initialized with a list of "carts", where each cart is a sequence
    of 'add' or 'remove' operations. The consumer will attempt to execute these
    operations, retrying indefinitely if a product is unavailable.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists. Each list contains
                          dictionaries specifying an operation ('type'), a
                          'product', and a 'quantity'.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a
                                     failed operation.
            **kwargs: Keyword arguments passed to the `Thread` constructor.
        """
        super().__init__(**kwargs)
        self.marketplace = marketplace
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """
        The main execution loop for the consumer thread.

        Processes each shopping list, executing all operations within it. It uses
        a retry-on-failure approach for adding/removing items. Once all
        operations for a cart are complete, it places the order.
        """
        for cart in self.carts:
            for operation in cart:
                op_type = operation['type']
                product = operation['product']
                quantity = operation['quantity']

                # Block Logic: This loop ensures that the specified quantity of an
                # operation is completed, retrying on failure.
                # Invariant: The loop continues until `quantity` reaches zero.
                while True:
                    # Attempt to perform the add or remove operation.
                    op_res = self.marketplace.add_to_cart(self.cart_id, product) 
                                if op_type == 'add' 
                                else self.marketplace.remove_from_cart(self.cart_id, product)

                    if op_res:
                        # If the operation was successful, decrement the remaining quantity.
                        quantity -= 1
                    else:
                        # If it failed (e.g., product unavailable), wait and retry.
                        sleep(self.retry_wait_time)

                    if quantity == 0:
                        break

            # After processing all operations, finalize the purchase.
            items_bought = self.marketplace.place_order(self.cart_id)
            if len(items_bought) > 0:
                with self.marketplace.print_lock:
                    print('
'.join(items_bought))


import time
from threading import Lock
from unittest import TestCase
import logging
import logging.handlers

from tema.product import Tea, Coffee


class Marketplace:
    """
    A thread-safe marketplace simulation.

    Manages the interactions between producer and consumer threads. It uses a
    single master lock (`market_lock`) to protect all shared state, including
    product inventory and cart contents, ensuring atomic operations. It also
    maintains a reservation system where items added to a cart are "reserved"
    to prevent other consumers from taking them before the order is placed.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items a single
                                           producer can have for sale at one time.
        """
        self.logger = logging.getLogger('marketplace_logger')
        self.logger.setLevel(logging.INFO)
        rotating_handler = logging.handlers.RotatingFileHandler('marketplace.log',
                                                                maxBytes=10000,
                                                                backupCount=10)

        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%dT%H:%M:%S')
        formatter.converter = time.gmtime
        rotating_handler.setFormatter(formatter)
        self.logger.addHandler(rotating_handler)

        # A single lock to protect all shared data structures.
        self.market_lock = Lock()
        # A separate lock to synchronize print statements from multiple consumers.
        self.print_lock = Lock()

        self.queue_size = queue_size_per_producer
        self.producer_items_count = {}
        self.consumer_id_count = 1
        self.producer_id_count = 1

        # {producer_id: {product_name: (total_count, reserved_count)}}
        self.products = {}
        # {cart_id: {product_name: {producer_id: count}}}
        self.carts = {}
        # {product_name: Product}
        self.all_products = {}

    def register_producer(self):
        """
        Registers a new producer and initializes its inventory tracking.

        Returns:
            str: The unique ID assigned to the new producer.
        """
        with self.market_lock:
            self.logger.info('entering register_producer')
            producer_id = f'producer{self.producer_id_count}'
            self.producer_items_count[producer_id] = 0
            self.products[producer_id] = {}
            self.producer_id_count += 1

            self.logger.info('leaving register_producer')
            return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id (str): The ID of the publishing producer.
            product (Product): The product to publish.

        Returns:
            bool: True if successful, False if the producer's queue is full.
        """
        self.logger.info('entering publish with args: {%s}, {%s}', producer_id, str(product))
        # Pre-condition: Check if the producer has reached their item limit.
        if self.producer_items_count[producer_id] == self.queue_size:
            self.logger.info('leaving publish')
            return False

        if product.name not in self.all_products:
            self.all_products[product.name] = product

        with self.market_lock:
            self.producer_items_count[producer_id] += 1
            if product.name not in self.products[producer_id]:
                # (total_items, reserved_items)
                self.products[producer_id][product.name] = (1, 0)
            else:
                num_items, reserved_items = self.products[producer_id][product.name]
                self.products[producer_id][product.name] = num_items + 1, reserved_items

        self.logger.info('leaving publish')
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            str: The unique ID assigned to the new cart.
        """
        with self.market_lock:
            self.logger.info('entering new_cart')
            cart_id = f'cons{self.consumer_id_count}'
            self.carts[cart_id] = {}
            self.consumer_id_count += 1

            self.logger.info('leaving new_cart')
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart, reserving it from the marketplace inventory.

        Algorithm: Iterates through producers to find one who has the desired
        product available (i.e., total items > reserved items). If found, it
        atomically increments the producer's reserved count and adds the item to
        the consumer's cart.

        Returns:
            bool: True if the item was successfully reserved and added, False otherwise.
        """
        self.logger.info('entering add_to_cart with args: {%s}, {%s}', cart_id, str(product))
        for producer_id, producer_products in self.products.items():
            if product.name in producer_products:
                num_items, reserved_items = producer_products[product.name]
                with self.market_lock:
                    # Pre-condition: Check if there's an unreserved item available.
                    if reserved_items < num_items:
                        # Reserve the item.
                        producer_products[product.name] = (num_items, reserved_items + 1)

                        if product.name not in self.carts[cart_id]:
                            self.carts[cart_id][product.name] = {}

                        if producer_id not in self.carts[cart_id][product.name]:
                            self.carts[cart_id][product.name][producer_id] = 1
                        else:
                            self.carts[cart_id][product.name][producer_id] += 1

                        self.logger.info('leaving add_to_cart')
                        return True

        self.logger.info('leaving add_to_cart')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart, releasing its reservation.

        Returns:
            bool: True if an item was successfully removed, False otherwise.
        """
        self.logger.info('entering remove_from_cart with args: {%s}, {%s}',
                         cart_id, str(product))
        deleted_producer_id = None
        for producer_id in self.carts[cart_id].get(product.name, {}):
            with self.market_lock:
                if self.carts[cart_id][product.name][producer_id] > 0:
                    deleted_producer_id = producer_id
                    # Decrement item count in cart.
                    self.carts[cart_id][product.name][producer_id] -= 1
                    if self.carts[cart_id][product.name][producer_id] == 0:
                        del self.carts[cart_id][product.name][producer_id]
                    if len(self.carts[cart_id][product.name]) == 0:
                        del self.carts[cart_id][product.name]

                    break

        if deleted_producer_id is None:
            self.logger.info('leaving remove_from_cart')
            return False

        # Un-reserve the item from the producer's inventory.
        with self.market_lock:
            num_items, reserved_items = self.products[deleted_producer_id][product.name]
            self.products[deleted_producer_id][product.name] = num_items, reserved_items - 1

        self.logger.info('leaving remove_from_cart')
        return True

    def place_order(self, cart_id):
        """
        Finalizes an order, consuming all items in a cart.

        This action adjusts the producer's inventory to reflect the sale,
        decrementing both the total and reserved counts.

        Returns:
            list: A list of strings describing the items bought.
        """
        self.logger.info('entering place_order with args: {%s}', cart_id)
        items_bought = []
        for product_name in self.carts[cart_id]:
            for producer_id, num_reserved in self.carts[cart_id][product_name].items():
                with self.market_lock:
                    num_items, reserved_items = self.products[producer_id][product_name]
                    # Update producer inventory to reflect the sale.
                    self.products[producer_id][product_name] = 
                        (num_items - num_reserved, reserved_items - num_reserved)
                    self.producer_items_count[producer_id] -= num_reserved
                    for _ in range(num_reserved):
                        items_bought.append(f'{cart_id} bought {self.all_products[product_name]}')

        # Clear the cart after the order is placed.
        self.carts[cart_id] = {}

        self.logger.info('leaving place_order')
        return items_bought


class TestMarketplace(TestCase):
    """Unit tests for the Marketplace class."""

    def setUp(self):
        """Set up a fresh marketplace for each test."""
        self.marketplace = Marketplace(2)
        self.first_prd_id = self.marketplace.register_producer()
        self.second_prd_id = self.marketplace.register_producer()

        self.first_cart_id = self.marketplace.new_cart()
        self.second_cart_id = self.marketplace.new_cart()

        self.fake_products = {'first_tea': Tea('Green', 2, 'Good'),
                              'second_tea': Tea('Black', 3, 'Bad'),
                              'first_coffee': Coffee('Brazilian', 5, 'high', 'high')}

    def test_register_producer(self):
        """Tests that producers are registered with unique sequential IDs."""
        first_producer_id = self.marketplace.register_producer()
        self.assertEqual(first_producer_id, 'producer3')
        self.assertEqual(self.marketplace.producer_items_count[first_producer_id], 0)
        self.assertTrue(first_producer_id in self.marketplace.products)
        self.assertEqual(self.marketplace.producer_id_count, 4)

        second_producer_id = self.marketplace.register_producer()
        self.assertEqual(second_producer_id, 'producer4')
        self.assertEqual(self.marketplace.producer_id_count, 5)

    def test_publish(self):
        """Tests product publishing, including queue size limits."""
        first_product = self.fake_products['first_tea']
        result = self.marketplace.publish(self.first_prd_id, first_product)
        self.assertEqual(result, True)
        self.assertTrue(first_product.name in self.marketplace.all_products)
        self.assertEqual(self.marketplace.products[self.first_prd_id][first_product.name], (1, 0))

        result = self.marketplace.publish(self.first_prd_id, first_product)
        self.assertEqual(result, True)
        self.assertEqual(self.marketplace.products[self.first_prd_id][first_product.name], (2, 0))

        # Test failure when queue is full.
        result = self.marketplace.publish(self.first_prd_id, first_product)
        self.assertEqual(result, False)

        second_product = self.fake_products['first_coffee']
        result = self.marketplace.publish(self.second_prd_id, second_product)
        self.assertEqual(result, True)
        self.assertEqual(self.marketplace.products[self.second_prd_id][second_product.name],
                         (1, 0))

    def test_new_cart(self):
        """Tests that new carts are created with unique sequential IDs."""
        first_cart_id = self.marketplace.new_cart()
        self.assertEqual(first_cart_id, 'cons3')
        self.assertTrue(first_cart_id in self.marketplace.carts)
        self.assertEqual(self.marketplace.consumer_id_count, 4)

        second_producer_id = self.marketplace.new_cart()
        self.assertEqual(second_producer_id, 'cons4')
        self.assertEqual(self.marketplace.consumer_id_count, 5)

    def test_add_to_cart(self):
        """Tests the item reservation logic when adding to a cart."""
        self.marketplace.publish(self.first_prd_id, self.fake_products['first_tea'])
        found_res = self.marketplace.add_to_cart(self.first_cart_id,
                                                 self.fake_products['first_tea'])
        self.assertTrue(found_res)
        self.assertEqual(self.marketplace.products[self.first_prd_id]['Green'],
                         (1, 1)) # Check reservation
        self.assertTrue('Green' in self.marketplace.carts[self.first_cart_id])
        self.assertEqual(self.marketplace.carts[self.first_cart_id]['Green'][self.first_prd_id],
                         1)

        # Test that adding again fails because item is reserved.
        add_again_res = self.marketplace.add_to_cart(self.first_cart_id,
                                                     self.fake_products['first_tea'])
        self.assertFalse(add_again_res)

        self.marketplace.publish(self.first_prd_id, self.fake_products['first_tea'])
        found_again = self.marketplace.add_to_cart(self.first_cart_id,
                                                   self.fake_products['first_tea'])
        self.assertTrue(found_again)
        self.assertEqual(self.marketplace.products[self.first_prd_id]['Green'],
                         (2, 2))
        self.assertEqual(self.marketplace.carts[self.first_cart_id]['Green'][self.first_prd_id],
                         2)

    def test_remove_from_cart(self):
        """Tests that removing from a cart correctly releases the reservation."""
        self.marketplace.publish(self.first_prd_id, self.fake_products['first_tea'])
        self.marketplace.add_to_cart(self.first_cart_id, self.fake_products['first_tea'])

        result = self.marketplace.remove_from_cart(self.first_cart_id,
                                                   self.fake_products['first_tea'])
        self.assertTrue(result)
        self.assertEqual(self.marketplace.products[self.first_prd_id]['Green'],
                         (1, 0)) # Check reservation is released
        self.assertTrue('Green' not in self.marketplace.carts[self.first_cart_id])

    def test_place_order(self):
        """Tests that placing an order correctly finalizes the sale."""
        self.marketplace.publish(self.first_prd_id, self.fake_products['first_tea'])
        self.marketplace.publish(self.first_prd_id, self.fake_products['first_coffee'])
        self.marketplace.add_to_cart(self.first_cart_id, self.fake_products['first_tea'])
        self.marketplace.add_to_cart(self.first_cart_id, self.fake_products['second_tea'])

        items_bought = self.marketplace.place_order(self.first_cart_id)
        self.assertTrue(str(self.fake_products['first_tea']) in items_bought[0])


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.

    The producer runs in a loop, attempting to publish a specified quantity
    of various products, sleeping between attempts.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, each containing
                             (product, quantity_to_produce, sleep_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait if publishing fails.
            **kwargs: Keyword arguments for the `Thread` constructor.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer.

        Continuously attempts to publish its assigned products until the
        target quantity for each is met.
        """
        while True:
            for product, quantity, sleep_time in self.products:
                # This inner loop is problematic; it will block on one product type.
                # A better implementation would iterate through all products.
                produce_res = self.marketplace.publish(self.producer_id, product)
                if produce_res:
                    quantity -= 1
                    sleep(sleep_time)
                else:
                    sleep(self.republish_wait_time)

                if quantity == 0:
                    break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """An immutable data class for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class for Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class for Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
