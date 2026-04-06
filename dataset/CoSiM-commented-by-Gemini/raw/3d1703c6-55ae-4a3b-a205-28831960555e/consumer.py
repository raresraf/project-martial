"""
This module simulates a producer-consumer model for an e-commerce marketplace.

It defines the core entities of the simulation:
- Marketplace: A central, thread-safe hub that manages product inventories from
  producers and processes shopping carts for consumers.
- Producer: A thread that continuously creates products and publishes them to the
  marketplace.
- Consumer: A thread that simulates a customer, adding items to a cart and
  placing orders.
- Product: Dataclasses representing the items being sold.
- TestMarketplace: Unit tests to verify the functionality of the Marketplace.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that processes a list of shopping carts.

    Each consumer simulates a customer's journey: creating a cart, adding and
    removing items, and finally placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts", where each cart is a list of
                          operations (add/remove).
            marketplace (Marketplace): A reference to the central marketplace object.
            retry_wait_time (int): Time in seconds to wait before retrying to
                                   add a product if it's not available.
            **kwargs: Keyword arguments for the Thread parent class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through its assigned carts, executes the add/remove operations
        for each, and places the final order.
        """

        # Invariant: Process all assigned shopping carts.
        for cart in self.carts:
            # Step 1: Get a new cart ID from the marketplace for this shopping session.
            cart_id = self.marketplace.new_cart()
            # Invariant: Process all operations within a single cart.
            for operation in cart:
                # Block Logic: Handle 'add' operations.
                if operation['type'] == 'add':
                    for _ in range(operation['quantity']):
                        success = self.marketplace.add_to_cart(cart_id, operation['product'])
                        # Pre-condition: If a product is not immediately available, retry.
                        # This loop simulates a user waiting for an item to be restocked.
                        while not success:
                            time.sleep(self.retry_wait_time)


                            success = self.marketplace.add_to_cart(cart_id, operation['product'])
                # Block Logic: Handle 'remove' operations.
                else:
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            # Step 2: Finalize the cart by placing the order.
            orders = self.marketplace.place_order(cart_id)
            for order in orders:
                print(self.kwargs['name'], 'bought', order)

import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
import unittest
from tema.product import Tea
from tema.product import Coffee

class Marketplace:
    """
    A thread-safe marketplace that coordinates producers and consumers.

    This class manages product inventories from multiple registered producers and
    handles the lifecycle of customer shopping carts, from creation to order placement.
    Locks are used extensively to ensure safe concurrent access from multiple
    producer and consumer threads.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers_slots = {}
        self.producers_queues = {}
        self.carts = {}
        self.next_producer_id = 0
        self.next_cart_id = 0

        # --- Synchronization Primitives ---
        self.slots_locks = {}
        self.queues_locks = {}
        self.producer_id_lock = Lock()
        self.cart_id_lock = Lock()

        # --- Logging Setup ---
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=20000, backupCount=5)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.

        Initializes the data structures to track the producer's inventory.

        Returns:
            str: The unique ID assigned to the new producer.
        """

        # Atomically generate a new producer ID.
        self.producer_id_lock.acquire()
        producer_id = str(self.next_producer_id)
        self.next_producer_id += 1
        self.producer_id_lock.release()

        # Initialize inventory slots, product queue, and corresponding locks.
        self.producers_slots[producer_id] = self.queue_size_per_producer
        self.producers_queues[producer_id] = []
        self.slots_locks[producer_id] = Lock()
        self.queues_locks[producer_id] = Lock()

        self.logger.info('Register producer: producer_id = %s', producer_id)
        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Args:
            producer_id (str): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's inventory queue is full.
        """

        # Pre-condition: Check if the producer has available inventory slots.
        self.slots_locks[producer_id].acquire()
        if self.producers_slots[producer_id] == 0:
            self.logger.info('Publish product: producer_id = %s, product = %s, 
                            return = False', producer_id, product)
            self.slots_locks[producer_id].release()
            return False
        self.slots_locks[producer_id].release()

        # Add the product to the producer's queue.
        self.queues_locks[producer_id].acquire()
        self.producers_queues[producer_id].append(product)
        self.queues_locks[producer_id].release()

        # Decrement the number of available slots.
        self.slots_locks[producer_id].acquire()
        self.producers_slots[producer_id] -= 1
        self.slots_locks[producer_id].release()

        self.logger.info('Publish product: producer_id = %s, product = %s, 
                        return = True ', producer_id, product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart.

        Returns:
            int: A unique ID for the newly created cart.
        """

        # Atomically generate a new cart ID.
        self.cart_id_lock.acquire()
        cart_id = self.next_cart_id
        self.next_cart_id = self.next_cart_id + 1
        self.cart_id_lock.release()

        self.carts[cart_id] = {}

        self.logger.info('New cart: cart_id = %d', cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's shopping cart.

        This method searches through all producers' inventories to find the
        product. If found, it moves the product from the producer's queue to
        the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """

        # Invariant: Search all producers for the requested product.
        for producer in self.producers_queues:
            # Lock the producer's queue to safely check for the product.
            self.queues_locks[producer].acquire()
            if product in self.producers_queues[producer]:
                # If found, remove it from the producer's inventory.
                self.producers_queues[producer].remove(product)
                self.queues_locks[producer].release()

                # Add the product to the consumer's cart.
                if not producer in self.carts[cart_id].keys():
                    self.carts[cart_id][producer] = []
                self.carts[cart_id][producer].append(product)

                self.logger.info('Add to cart: cart_id = %d, product = %s, 
                                return = True', cart_id, product)
                return True
            self.queues_locks[producer].release()

        # If the product was not found in any producer's queue.
        self.logger.info('Add to cart: cart_id = %d, product = %s, 
                        return = False', cart_id, product)
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the producer.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to remove.
        """
        for producer in self.carts[cart_id]:
            if product in self.carts[cart_id][producer]:
                # Remove from cart.
                self.carts[cart_id][producer].remove(product)
                
                # Return the product to the producer's inventory queue.
                self.queues_locks[producer].acquire()
                self.producers_queues[producer].append(product)
                self.queues_locks[producer].release()

                self.logger.info('Remove from cart: cart_id = %d, product = %s', cart_id, product)
                break

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This method moves all products from the cart into a final order list and
        replenishes the inventory slots for the respective producers, allowing them
        to publish new products.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of all products that were in the cart.
        """
        returned_list = []

        # Invariant: Process all products from all producers within the cart.
        for producer in self.carts[cart_id]:
            for product in self.carts[cart_id][producer]:
                returned_list.append(product)
                
                # Replenish the producer's available slot count now that the item is sold.
                self.slots_locks[producer].acquire()
                self.producers_slots[producer] += 1
                self.slots_locks[producer].release()

        # Delete the cart after the order is placed.
        del self.carts[cart_id]

        self.logger.info('Place order: cart_id = %d,returned list = %s', cart_id, list)
        return returned_list


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""

    def setUp(self):
        """Sets up a new Marketplace and some sample products for each test."""
        self.marketplace = Marketplace(3)
        self.products = []
        self.products.append(Tea('Musetel', 1, 'Herbal'))
        self.products.append(Tea('Coada Soricelului', 3, 'Herbal'))
        self.products.append(Coffee('Espresso', 2, '10.0', 'HIGH'))
        self.products.append(Tea('Urechea boului', 4, 'Non-herbal'))

    def test_register_producer(self):
        """Tests the producer registration logic."""
        for i in range(0, 10):
            self.assertEqual(self.marketplace.register_producer(), str(i))

    def test_new_cart(self):
        """Tests the cart creation logic."""
        for i in range(0, 10):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_publish(self):
        """Tests successful product publishing."""
        for _ in range(0, 4):
            self.marketplace.register_producer()

        self.marketplace.publish('0', self.products[0])
        self.marketplace.publish('0', self.products[1])
        self.marketplace.publish('2', self.products[0])
        self.marketplace.publish('0', self.products[0])
        self.marketplace.publish('1', self.products[3])

        self.assertEqual(self.marketplace.producers_queues['0'],
                         [self.products[0], self.products[1], self.products[0]])
        self.assertEqual(self.marketplace.producers_queues['1'], [self.products[3]])
        self.assertEqual(self.marketplace.producers_queues['2'], [self.products[0]])
        self.assertEqual(self.marketplace.producers_queues['3'], [])

    def test_publish_fails(self):
        """Tests that publishing fails when the producer's queue is full."""
        self.marketplace.register_producer()
        for i in range(0, 3):
            self.assertTrue(self.marketplace.publish(str(0), self.products[i]))
        self.assertFalse(self.marketplace.publish(str(0), self.products[0]))

    def test_add_to_cart(self):
        """Tests the add_to_cart functionality."""

        # Setup: Create carts, producers, and publish products.
        for i in range(0, 3):
            self.marketplace.new_cart()
            self.marketplace.register_producer()
        for i in range(0, 3):
            for _ in range(0, 3):
                self.marketplace.publish(str(i), self.products[i])

        # Test: Successfully add items to the cart.
        for i in range(0, 3):
            for _ in range(0, 3):
                self.assertTrue(self.marketplace.add_to_cart(i, self.products[i]))

        # Test: Fail to add more items than are available.
        for i in range(0, 3):
            self.assertFalse(self.marketplace.add_to_cart(i, self.products[i]))
        
        # Verify final cart contents.
        for i in range(0, 3):
            self.assertEqual(self.marketplace.carts[i][str(i)],
                             [self.products[i], self.products[i], self.products[i]])

    def test_remove_from_cart(self):
        """Tests removing items from a cart."""
        id_cart = self.marketplace.new_cart()
        id_producer = self.marketplace.register_producer()
        for i in range(0, 3):
            self.marketplace.publish('0', self.products[i])
            self.marketplace.add_to_cart(0, self.products[i])

        self.marketplace.remove_from_cart(id_cart, self.products[1])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer],
                         [self.products[0], self.products[2]])
        self.marketplace.remove_from_cart(id_cart, self.products[0])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer], [self.products[2]])
        self.marketplace.remove_from_cart(id_cart, self.products[2])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer], [])

    def test_place_order(self):
        """Tests the order placement logic."""

        # Setup: Create a cart, a producer, and add items.
        cart_id = self.marketplace.new_cart()
        producer_id = self.marketplace.register_producer()
        for i in range(0, 3):
            self.marketplace.publish(producer_id, self.products[i])
            self.marketplace.add_to_cart(cart_id, self.products[i])

        returned_list = self.marketplace.place_order(cart_id)
        
        # Verify the returned order contents.
        self.assertEqual(returned_list, [self.products[0], self.products[1], self.products[2]])
        
        # Verify that the producer's inventory slots have been replenished.
        self.assertEqual(self.marketplace.producers_slots[producer_id], 3)


import time
from threading import Thread



class Producer(Thread):
    """
    Represents a producer thread that continuously creates products.

    The producer publishes a list of products to the marketplace at specified
    intervals, simulating a real-world inventory supplier.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains a
                             product, quantity, and production delay.
            marketplace (Marketplace): A reference to the central marketplace.
            republish_wait_time (int): Time to wait before retrying to publish
                                       if the marketplace queue is full.
            **kwargs: Keyword arguments for the Thread parent class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and enters an infinite loop to publish
        products.
        """

        producer_id = self.marketplace.register_producer()
        
        # Invariant: Continuously produce and publish products.
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    time.sleep(product[2])
                    success = self.marketplace.publish(producer_id, product[0])
                    # Pre-condition: If publishing fails, wait and retry.
                    # This simulates waiting for inventory space to become available.
                    while not success:
                        time.sleep(self.republish_wait_time)
                        success = self.marketplace.publish(producer_id, product[0])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Represents a Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Represents a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
