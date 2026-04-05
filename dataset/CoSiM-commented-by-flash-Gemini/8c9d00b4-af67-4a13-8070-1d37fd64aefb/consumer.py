

"""
@8c9d00b4-af67-4a13-8070-1d37fd64aefb/consumer.py
@brief Defines a multi-threaded simulation of a marketplace where Consumer and Producer entities interact with a central Marketplace to manage product inventory and transactions. It also includes product data structures and unit tests for the marketplace.
Functional Utility: This module provides a comprehensive simulation environment for studying concurrent operations in an e-commerce context, focusing on resource management and synchronization.
Domain: Concurrency, Producer-Consumer Problem, E-commerce Simulation, Data Structures.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Represents a consumer that interacts with the marketplace to add and remove products from a cart and place orders.
    Functional Utility: Manages the lifecycle of a consumer's shopping activities, from creating a cart to placing an order, including handling product availability.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts: A list of shopping carts, where each cart is a list of operations (add/remove product).
        @param marketplace: Reference to the shared `Marketplace` instance.
        @param retry_wait_time: Time to wait before retrying an `add_to_cart` operation.
        @param **kwargs: Additional keyword arguments, including the thread name.
        Functional Utility: Sets up the consumer with its assigned shopping intentions, access to the marketplace, and retry logic.
        """
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.carts = carts

    def run(self):
        """
        @brief Executes the consumer's shopping logic.
        Functional Utility: Orchestrates the consumer's interaction with the marketplace, processing each cart's operations and finalizing orders.
        """
        for cart in self.carts:
            # Functional Utility: Requests a new, unique shopping cart from the marketplace.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Processes each command (add or remove) within the current cart.
            for command in cart:
                # Block Logic: Handles adding a product to the cart.
                if command["type"] == "add":
                    for _ in range(command["quantity"]):
                        # Invariant: Product not successfully added to cart.
                        # Functional Utility: Continuously attempts to add the product to the cart until successful.
                        while not self.marketplace.add_to_cart(cart_id, command["product"]):
                            # Functional Utility: Pauses execution before retrying to prevent busy-waiting.
                            sleep(self.retry_wait_time)
                # Functional Utility: Handles removing a product from the cart.
                elif command["type"] == "remove":
                    for _ in range(command["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, command["product"])

            # Functional Utility: Finalizes the order for the current cart.
            items = self.marketplace.place_order(cart_id)
            # Functional Utility: Acquires a lock to ensure exclusive access to shared console output, preventing interleaved print statements from different consumers.
            for item in items:
                with self.marketplace.print_lock:
                    # Functional Utility: Prints the list of products successfully purchased by the consumer.
                    print(self.name, "bought", item[0])

import time
import unittest
from threading import Lock
from tema.product import *
import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    handlers=[RotatingFileHandler('tema/marketplace.log', maxBytes=1000, backupCount=10)],
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.Formatter.converter = time.gmtime


class Marketplace:
    """
    @brief Manages products, producers, and consumer carts in a thread-safe manner.
    Functional Utility: Provides the central logic for product flow within the e-commerce simulation, ensuring data consistency and handling interactions between producers and consumers.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace.
        @param queue_size_per_producer: The maximum number of products a producer can have in the marketplace at any given time.
        Functional Utility: Sets up internal data structures for managing products, carts, producer and consumer IDs, and associated locks for thread safety.
        """
        self.queue = []
        self.queue_size_per_producer = queue_size_per_producer


        self.carts = []
        self.producer_ids = 0
        self.cart_ids = 0

        self.producer_lock = Lock()
        self.cart_lock = Lock()
        self.print_lock = Lock()
        self.queue_lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace, assigning it a unique ID.
        Functional Utility: Atomically increments the producer ID and initializes an empty product queue for the new producer.
        Synchronization: Uses `self.producer_lock` to protect shared state during registration.
        @return: The newly registered producer's ID.
        """
        logging.info('Method called: register_producer')
        self.producer_lock.acquire()

        curr_id = self.producer_ids
        self.queue.append([])
        self.producer_ids += 1

        self.producer_lock.release()


        logging.info('Method register_producer returned ' + str(curr_id))
        return curr_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to publish a product to the marketplace.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product to be published.
        Functional Utility: Adds the product to the producer's queue within the marketplace, if the producer's queue size limit is not exceeded.
        Pre-condition: `len(self.queue[producer_id]) < self.queue_size_per_producer`.
        @return: `True` if the product was successfully published, `False` otherwise.
        """
        logging.info('Method called: publish; params: producer_id=' + str(producer_id) + ' product=' + str(product))

        # Block Logic: Checks if the producer has reached its maximum allowed products in the marketplace.
        if len(self.queue[producer_id]) >= self.queue_size_per_producer:
            logging.info('Method publish returned False')
            return False

        self.queue[producer_id].append(product)
        logging.info('Method publish returned True')
        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart for a consumer, assigning it a unique ID.
        Functional Utility: Atomically increments the cart ID and initializes an empty cart for the consumer.
        Synchronization: Uses `self.cart_lock` to protect shared state during cart creation.
        @return: The ID of the newly created cart.
        """
        logging.info('Method called: new_cart')
        self.cart_lock.acquire()

        cart_id = self.cart_ids
        self.carts.append([])
        self.cart_ids += 1

        self.cart_lock.release()
        logging.info('Method new_cart returned ' + str(cart_id))
        return cart_id

    def find_product(self, product):
        """
        @brief Searches for a specific product across all producer queues in the marketplace.
        @param product: The product to find.
        Functional Utility: Locates the first occurrence of the product and returns its producer ID and index within that producer's queue.
        @return: A tuple `(producer_idx, item_idx)` if found, or `(-1, -1)` if not found.
        """
        logging.info('Method called: find_product; params: product=' + str(product))
        for producer_idx, producer_queue in enumerate(self.queue):
            for item_idx, item in enumerate(producer_queue):
                if item == product:
                    logging.info('Method find_product returned ' + str(producer_idx) + ', ' + str(item_idx))
                    return producer_idx, item_idx

        logging.info('Method find_product returned -1, -1')
        return -1, -1

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to add a product to a consumer's cart.
        @param cart_id: The ID of the cart to which the product should be added.
        @param product: The product to add.
        Functional Utility: Moves a product from a producer's queue to the specified cart, ensuring atomicity and checking for product availability.
        Synchronization: Uses `self.queue_lock` to ensure thread-safe access to product queues and carts during the operation.
        Pre-condition: The `product` must be available in any producer's queue.
        @return: `True` if the product was successfully added, `False` otherwise.
        """
        logging.info('Method called: add_to_cart; params: cart_id=' + str(cart_id) + ' product=' + str(product))

        self.queue_lock.acquire()

        producer_idx, item_idx = self.find_product(product)

        # Block Logic: If the product is not found, releases the lock and returns False.
        if producer_idx == -1 and item_idx == -1:
            self.queue_lock.release()
            logging.info('Method add_to_cart returned False')
            return False

        # Functional Utility: Transfers the product from the producer's queue to the consumer's cart.
        self.carts[cart_id].append(
            [self.queue[producer_idx].pop(item_idx), producer_idx]
        )

        self.queue_lock.release()
        logging.info('Method add_to_cart returned True')
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a consumer's cart and returns it to the marketplace.
        @param cart_id: The ID of the cart from which to remove the product.
        @param product: The product to remove.
        Functional Utility: Finds and removes a product from the specified cart, then re-publishes it to the original producer's queue.
        Synchronization: This method does *not* explicitly use `self.queue_lock` for the `publish` call, which might lead to a race condition if `publish` is not internally thread-safe.
        """
        logging.info('Method called: remove_from_cart; params: cart_id=' + str(cart_id) + ' product=' + str(product))

        for idx, prod in enumerate(self.carts[cart_id]):
            # Block Logic: Searches for the product in the cart.
            if prod[0] == product:
                # Functional Utility: Re-publishes the product to its original producer and removes it from the cart.
                self.publish(prod[1], product)
                self.carts[cart_id].pop(idx)
                break
        logging.info('Method remove_from_cart returned void')

    def place_order(self, cart_id):
        """
        @brief Finalizes a consumer's order for a given cart.
        @param cart_id: The ID of the cart for which to place the order.
        Functional Utility: Returns the contents of the cart as the placed order. Note that products are *not* removed from the cart at this stage, nor are producer counts decremented, unlike the previous `Marketplace` implementation.
        @return: A list of products (with their original producer IDs) in the placed order.
        """
        logging.info('Method called: place_order; params: cart_id=' + str(cart_id))
        logging.info('Method place_order returned ' + str(self.carts[cart_id]))
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    """
    @brief Provides unit tests for the `Marketplace` class to ensure its functionality.
    Functional Utility: Verifies the correct behavior of the Marketplace's core methods under various scenarios.
    """
    def setUp(self):
        """
        @brief Initializes a `Marketplace` instance and populates its internal queues and carts for testing.
        Functional Utility: Creates a controlled environment for testing specific marketplace operations, ensuring reproducibility.
        """
        self.marketplace = Marketplace(15)
        self.marketplace.queue = [
            [Tea(name='Linden', price=9, type='Herbal'), Tea(name='Linden', price=9, type='Herbal'),
             Tea(name='Test', price=9, type='Herbal'), Tea(name='Linden', price=9, type='Herbal'),
             Tea(name='Linden', price=9, type='Herbal'),
             Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')]]
        self.marketplace.carts = [[[Tea(name='Test1', price=9, type='Herbal'), 0],
                                   [Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM'), 0],
                                   [Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM'), 0]]]

    def test_register_producer(self):
        """
        @brief Tests the `register_producer` method.
        Functional Utility: Verifies that a producer can be successfully registered and assigned a unique ID.
        """
        self.assertEqual(self.marketplace.register_producer(), 0)

    def test_publish(self):
        """
        @brief Tests the `publish` method.
        Functional Utility: Confirms that a producer can publish a product and it is correctly added to the marketplace.
        """
        prod_id = self.marketplace.register_producer()
        prod = Coffee(name='Test_Publish', price=1, acidity=5.05, roast_level='MEDIUM')
        self.marketplace.publish(prod_id, prod)
        self.assertTrue(prod in self.marketplace.queue[prod_id])

    def test_new_cart(self):
        """
        @brief Tests the `new_cart` method.
        Functional Utility: Verifies that a new shopping cart can be successfully created and added to the marketplace's cart list.
        """
        first_length = len(self.marketplace.carts)
        self.marketplace.new_cart()
        self.assertEqual(first_length + 1, len(self.marketplace.carts))

    def test_find_product(self):
        """
        @brief Tests the `find_product` method.
        Functional Utility: Confirms that the marketplace can correctly locate a specific product within its inventory.
        """
        prod = Tea(name='Test', price=9, type='Herbal')
        prod_idx, item_idx = self.marketplace.find_product(prod)
        self.assertEqual(prod_idx, 0)
        self.assertEqual(item_idx, 2)

    def test_add_to_cart(self):
        """
        @brief Tests the `add_to_cart` method.
        Functional Utility: Verifies that a product can be successfully added to a cart and removed from the marketplace's available products.
        """
        first_len = len(self.marketplace.queue[0])
        prod = Tea(name='Test', price=9, type='Herbal')
        self.marketplace.add_to_cart(0, prod)
        self.assertEqual(len(self.marketplace.queue[0]), first_len - 1)
        prod_idx, item_idx = self.marketplace.find_product(prod)
        self.assertEqual(prod_idx, -1)
        self.assertEqual(item_idx, -1)

    def test_remove_from_cart(self):
        """
        @brief Tests the `remove_from_cart` method.
        Functional Utility: Ensures that a product can be correctly removed from a cart and is made available again in the marketplace.
        """
        prod = Tea(name='Test1', price=9, type='Herbal')
        first_len = len(self.marketplace.carts[0])
        self.marketplace.remove_from_cart(0, prod)
        self.assertEqual(len(self.marketplace.carts[0]), first_len - 1)

    def test_place_order(self):
        """
        @brief Tests the `place_order` method.
        Functional Utility: Verifies that placing an order correctly returns the items that were in the cart.
        """
        self.assertEqual(self.marketplace.place_order(0), self.marketplace.carts[0])


class Producer(Thread):
    """
    @brief Represents a producer that continuously publishes products to the marketplace.
    Functional Utility: Manages the product supply side of the e-commerce simulation, ensuring products are made available in the marketplace according to a defined schedule.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products: A list of products to publish, including product ID, quantity, and a sleep time after publishing.
        @param marketplace: Reference to the shared `Marketplace` instance.
        @param republish_wait_time: Time to wait before retrying to publish if the marketplace is full for this producer.
        @param **kwargs: Additional keyword arguments, including the thread name.
        Functional Utility: Sets up the producer with its inventory, access to the marketplace, and retry logic.
        """
        Thread.__init__(self, **kwargs)
        self.id = None
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Executes the producer's publishing logic.
        Functional Utility: Continuously registers with the marketplace and attempts to publish its products, handling potential delays due to marketplace capacity.
        """
        # Functional Utility: Registers the producer with the marketplace and obtains a unique ID.
        self.id = self.marketplace.register_producer()

        # Block Logic: Enters an infinite loop to continuously publish products.
        while True:
            # Block Logic: Iterates through the list of products this producer can supply.
            for product in self.products:
                for _ in range(product[1]):
                    # Invariant: Product not successfully published.
                    # Functional Utility: Continuously attempts to publish the product until successful, pausing if the marketplace capacity for this producer is reached.
                    while not self.marketplace.publish(self.id, product[0]):
                        # Functional Utility: Pauses execution before retrying to publish.
                        sleep(self.republish_wait_time)

                    # Functional Utility: Pauses after publishing a product, simulating production time.
                    sleep(product[2])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base class for all products in the marketplace.
    Functional Utility: Provides a common structure for product identification and pricing.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a Tea product, inheriting from the base Product class.
    Functional Utility: Extends the generic product definition with a specific type attribute for tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a Coffee product, inheriting from the base Product class.
    Functional Utility: Extends the generic product definition with attributes specific to coffee characteristics like acidity and roast level.
    """
    acidity: str
    roast_level: str
