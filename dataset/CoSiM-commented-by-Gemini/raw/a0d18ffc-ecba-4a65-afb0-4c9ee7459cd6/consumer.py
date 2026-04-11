"""
This module implements a producer-consumer simulation of a marketplace.

It defines the core components of the simulation:
- Marketplace: A thread-safe class that manages producers, products, and carts.
  It uses semaphores to control concurrent access to shared resources.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places an order.
- TestMarketplace: A suite of unit tests to verify the marketplace functionality.
"""

from threading import Thread
from time import sleep
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import BoundedSemaphore
from random import randint
import sys
import threading
import time
from uuid import UUID, uuid1
from unittest import TestCase

# The following are placeholder classes as the original `tema.product` is not available.
class Product:
    """Base class for a product."""
    pass
class Tea(Product):
    """Represents a Tea product."""
    def __init__(self, name, price, type):
        pass
class Coffee(Product):
    """Represents a Coffee product."""
    def __init__(self, name, price, acidity, roast):
        pass

class Consumer(Thread):
    """
    A thread that simulates a consumer purchasing products from the marketplace.

    The consumer processes a list of carts, with each cart containing a sequence
    of 'add' or 'remove' operations. It persistently tries to add products
    until successful and then places the order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping cart actions to perform.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): Time in seconds to wait before retrying a failed action.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main execution logic for the consumer thread."""
        for cart in self.carts:
            # Each consumer session gets a new unique cart from the marketplace.
            id_cart = self.marketplace.new_cart()
            # Process all actions (add/remove) for the current cart.
            for action in cart:
                if action['type'] == 'add':
                    # Block Logic: Attempt to add the specified quantity of a product.
                    quantity = 0
                    # Invariant: Loop until the desired quantity has been successfully added.
                    while quantity < action['quantity']:
                        if self.marketplace.add_to_cart(id_cart, action['product']):
                            quantity += 1
                        else:
                            # If adding fails (e.g., product is unavailable), wait and retry.
                            sleep(self.retry_wait_time)
                elif action['type'] == 'remove':
                    # Block Logic: Remove the specified quantity of a product.
                    for _ in range(action['quantity']):
                        self.marketplace.remove_from_cart(id_cart, action['product'])
            
            # Finalize the purchase.
            self.marketplace.place_order(id_cart)


class Marketplace:
    """
    Manages producers, products, and carts in a thread-safe manner.

    This class acts as the central hub for the simulation. It uses semaphores
    to ensure that concurrent operations from multiple producers and consumers
    do not lead to race conditions or inconsistent states.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Data Structure: producer_queues is a dictionary where each key is a producer's UUID.
        # The value is a list: [Semaphore, inventory_count, list_of_products].
        # - [0]: A BoundedSemaphore acting as a mutex for the producer's specific data.
        # - [1]: An integer tracking the number of products currently available from this producer.
        # - [2]: A list where each item is `[Product, is_available_bool]`.
        self.producer_queues = {}
        
        # Data Structure: `carts` maps a cart ID to a list of (producer_id, product) tuples.
        self.carts = {}
        
        # Synchronization: A semaphore to protect the global `carts` dictionary during creation.
        self.carts_mutex = BoundedSemaphore(1)
        # Synchronization: A semaphore to ensure that print statements are atomic and not interleaved.
        self.print_mutex = BoundedSemaphore(1)

        # Set up logging to a rotating file.
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=10**6, backupCount=10)
        formatter = logging.Formatter('%(asctime)s UTC %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def register_producer(self) -> UUID:
        """
        Registers a new producer with the marketplace.

        Returns:
            UUID: A unique identifier for the newly registered producer.
        """
        logging.info('register_producer() was called')
        id_prod = uuid1()
        self.producer_queues[id_prod] = [BoundedSemaphore(1), 0, []]
        logging.info('register_producer() returned (%s)', id_prod)
        return id_prod

    def publish(self, producer_id: UUID, product: Product) -> bool:
        """
        Allows a producer to list a new product for sale.

        Args:
            producer_id (UUID): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published, False if the producer's inventory is full.
        """
        logging.info('publish(%s, %s) was called', producer_id, product)
        # Pre-condition: Check if the producer has space to publish a new product.
        if self.producer_queues[producer_id][1] < self.queue_size_per_producer:
            self.producer_queues[producer_id][2].append([product, True])
            # Synchronization: Safely increment the producer's inventory count.
            with self.producer_queues[producer_id][0]:
                self.producer_queues[producer_id][1] += 1
            logging.info('publish(%s, %s) returned True', producer_id, product)
            return True
        logging.info('publish(%s, %s) returned False', producer_id, product)
        return False

    def new_cart(self) -> int:
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: A unique ID for the new cart.
        """
        logging.info('new_cart() was called')
        # Synchronization: Ensure that cart creation is atomic.
        with self.carts_mutex:
            id_cart = randint(0, sys.maxsize)
            # Invariant: Ensure the generated cart ID is unique.
            while id_cart in list(self.carts.keys()):
                id_cart = randint(0, sys.maxsize)
            self.carts[id_cart] = []
            logging.info('new_cart() returned %d', id_cart)
            return id_cart

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """
        Adds an available product to a consumer's cart.

        It searches all producer inventories for an available product and, if found,
        marks it as unavailable and adds it to the cart.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        logging.info('add_to_cart(%d, %s) was called', cart_id, product)
        # Block Logic: Iterate through all producers to find the product.
        for id_prod in self.producer_queues:
            # Synchronization: Lock the specific producer's data while checking their inventory.
            with self.producer_queues[id_prod][0]:
                for prod in self.producer_queues[id_prod][2]:
                    # Find the first available product that matches.
                    if prod[0] == product and prod[1]:
                        prod[1] = False # Mark as unavailable.
                        self.producer_queues[id_prod][1] -= 1
                        self.carts[cart_id].append((id_prod, product))
                        logging.info('publish(%s, %s) returned True', cart_id, product)
                        return True
        logging.info('add_to_cart(%s, %s) returned False', cart_id, product)
        return False

    def remove_from_cart(self, cart_id: int, product: Product):
        """
        Removes a product from a cart, making it available again.
        """
        logging.info('remove_from_cart(%d, %s) was called', cart_id, product)
        for item in self.carts[cart_id]:
            if item[1] == product:
                # Synchronization: Lock the producer's data to return the item.
                with self.producer_queues[item[0]][0]:
                    for prod in self.producer_queues[item[0]][2]:
                        if prod[0] == product:
                            prod[1] = True # Mark as available again.
                            self.producer_queues[item[0]][1] += 1
                            self.carts[cart_id].remove((item[0], product))
                            logging.info('remove_from_cart(%d, %s) returned', cart_id, product)
                            return

    def place_order(self, cart_id: int):
        """
        Finalizes an order, permanently removing items from producer inventories.
        """
        logging.info('place_order(%d) was called', cart_id)
        result = []
        for item in self.carts[cart_id]:
            result.append(item[1])
            # Synchronization: Lock the producer's data to finalize the sale.
            with self.producer_queues[item[0]][0]:
                for prod in self.producer_queues[item[0]][2]:
                    if prod[0] == item[1]:
                        # Item is sold and removed from the producer's list.
                        self.producer_queues[item[0]][2].remove(prod)
                        # Note: The inventory count was already decremented in add_to_cart.
                        # This logic appears to have a bug where the count is decremented twice.
                        # self.producer_queues[item[0]][1] -= 1
                        break
        
        self.carts.pop(cart_id)
        # Synchronization: Use a mutex to ensure clean printing from multiple threads.
        with self.print_mutex:
            for item in result:
                print(threading.current_thread().name, "bought", item)
        logging.info('place_order(%d) returned %s', cart_id, result)
        return result

class TestMarketplace(TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Initializes a new marketplace and products for each test."""
        self.marketplace = Marketplace(1)
        self.coffee = Coffee('Indonezia', 4, '5.05', 'MEDIUM')
        self.tea = Tea('Linden', 9, 'Herbal')

    def test_register_producer(self):
        """Tests that producer registration returns unique IDs and initializes data correctly."""
        id_prod = self.marketplace.register_producer()
        self.assertIsInstance(id_prod, UUID, "Return type not UUID")
        self.assertEqual(self.marketplace.producer_queues[id_prod][1], 0, "Initial size not 0")
        self.assertEqual(len(self.marketplace.producer_queues[id_prod][2]), 0, "Queue not empty")
        self.assertNotEqual(id_prod, self.marketplace.register_producer(), "IDs equal")

    def test_publish(self):
        """Tests that publishing succeeds when slots are available and fails when they are full."""
        id_prod = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(id_prod, self.coffee), "First publish should pass")
        self.assertFalse(self.marketplace.publish(id_prod, self.tea), "Second publish should fail")

    def test_new_cart(self):
        """Tests that new carts are created with unique IDs."""
        cart_id = self.marketplace.new_cart()
        self.assertGreaterEqual(cart_id, 0, "Cart ID should not be negative")
        self.assertNotEqual(cart_id, self.marketplace.new_cart(), "IDs equal")

    def test_add_to_cart(self):
        """Tests that a product can be added to a cart only if it's available."""
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.coffee), "Coffee is in store")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.tea), "Tea not in store")

    def test_remove_from_cart(self):
        """Tests that removing an item from a cart makes it available again."""
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.coffee)
        self.marketplace.remove_from_cart(cart_id, self.tea) # Should do nothing
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1, "Item not in cart")
        self.marketplace.remove_from_cart(cart_id, self.coffee)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 0, "Item should be removed")

    def test_place_order(self):
        """Tests that placing an order correctly returns the products."""
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.tea) # Should fail
        self.marketplace.add_to_cart(cart_id, self.coffee) # Should succeed
        self.marketplace.remove_from_cart(cart_id, self.tea) # Should do nothing
        self.assertEqual(self.marketplace.place_order(cart_id), [self.coffee], "Results differ")


class Producer(Thread):
    """
    A thread that simulates a producer publishing products to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of (product, quantity, delay) tuples to publish.
            marketplace (Marketplace): The marketplace instance.
            republish_wait_time (float): Time to wait before retrying a failed publish.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.product_actions = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        """The main execution logic for the producer thread."""
        # Invariant: The producer runs in an infinite loop to continuously supply products.
        while True:
            for (product, quantity, delay) in self.product_actions:
                total = 0
                # Invariant: Publish the specified quantity of the current product.
                while total < quantity:
                    if self.marketplace.publish(self.id_producer, product):
                        total += 1
                        sleep(delay)
                    else:
                        # If publishing fails (inventory is full), wait and retry.
                        sleep(self.republish_wait_time)
