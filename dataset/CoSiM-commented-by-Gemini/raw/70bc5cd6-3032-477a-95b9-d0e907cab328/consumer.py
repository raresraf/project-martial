"""
This module simulates a multi-threaded producer-consumer marketplace.

It defines classes for a `Marketplace`, which acts as the central shared
resource, `Consumer` threads that acquire products, and `Producer` threads
that supply products. The simulation uses a coarse-grained locking strategy
with a single `threading.Lock` to ensure data consistency across all
concurrent operations.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer thread that simulates a shopping process.

    Each consumer is given a list of shopping lists ('carts' of actions) and
    interacts with the marketplace to fulfill them. It operates on a single
    persistent cart for its entire lifetime.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping action lists. Each inner list
                contains dictionaries specifying 'type', 'product', 'quantity'.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying to
                add a product to the cart.
            **kwargs: Additional keyword arguments for the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # A single cart is created for the lifetime of this consumer instance.
        self.cart_id = marketplace.new_cart()

    def run(self):
        """The main execution logic for the consumer thread."""
        # Invariant: The consumer processes each list of shopping actions.
        for k in range(len(self.carts)):
            for elem in self.carts[k]:
                if elem['type'] == 'add':
                    # Block Logic: Attempt to add the specified quantity of a product.
                    for i in range(elem['quantity']):
                        # Pre-condition: If adding fails (e.g., product unavailable),
                        # block and retry until successful.
                        while not self.marketplace.add_to_cart(self.cart_id, elem['product']):
                            time.sleep(self.retry_wait_time)

                if elem['type'] == 'remove':
                    # Block Logic: Remove the specified quantity of a product.
                    for i in range(elem['quantity']):
                        self.marketplace.remove_from_cart(self.cart_id, elem['product'])
        
        # After all actions, place the final order.
        self.marketplace.place_order(self.cart_id)

# --- Start of concatenated Marketplace, Testing, and Producer modules ---

from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import unittest
from product import Coffee
from product import Tea

class Marketplace:
    """
    A thread-safe marketplace managing inventory from producers and carts for consumers.

    This class uses a single, coarse-grained lock to protect all shared data,
    ensuring consistency but potentially limiting concurrency.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): Max number of products a producer can have in stock.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.cart_id = 0
        self.products_stock = []  # List of lists, inventory per producer.
        self.carts = [] # List of lists, items per cart.

        # A single lock for all marketplace operations to ensure thread safety.
        self.lock = Lock()
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler("marketplace.log", maxBytes=200000, backupCount=10)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Gives a unique ID to a new producer and prepares their inventory slot.

        Returns:
            int: The new producer's unique ID.
        """
        self.logger.info("START: register_producer")
        with self.lock:
            self.id_producer += 1
            self.products_stock.append([])
        self.logger.info("END: register_producer")
        return self.id_producer

    def publish(self, producer_id, product):
        """
        Allows a producer to add a product to their stock.

        Args:
            producer_id (int): The ID of the publishing producer.
            product: The product to add.

        Returns:
            bool: True if successful, False if the producer's stock is full.
        """
        self.logger.info("START: publish")
        self.logger.info("Params-> producer_id: {}, product: {}".format(producer_id, product))
        with self.lock:
            # Pre-condition: Check if the producer has space for more products.
            if len(self.products_stock[producer_id - 1]) >= self.queue_size_per_producer:
                self.logger.info("END: publish")
                return False

            self.products_stock[producer_id - 1].append(product)

        self.logger.info("END: publish")
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID for the new cart.
        """
        self.logger.info("START: new_cart")
        with self.lock:
            self.cart_id += 1
            self.carts.append([])
        self.logger.info("END: new_cart")
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from a producer's stock to a consumer's cart.

        It searches all producer inventories for the product. If found, the product
        is transferred from the stock to the cart.

        Args:
            cart_id (int): The consumer's cart ID.
            product: The product to add.

        Returns:
            bool: True if the product was found and moved, False otherwise.
        """
        self.logger.info("START: add_to_cart")
        self.logger.info("Params-> cart_id: {}, product: {}".format(cart_id, product))

        for i in range(len(self.products_stock)):
            with self.lock:
                if self.products_stock[i].count(product) > 0:
                    # Block Logic: Move item from producer stock to consumer cart.
                    # The cart stores which producer it came from.
                    self.carts[cart_id - 1].append((i, product))
                    self.products_stock[i].remove(product)
                    self.logger.info("START: add_to_cart")
                    return True
        self.logger.info("END: add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Moves a product from a consumer's cart back to the original producer's stock.

        Args:
            cart_id (int): The consumer's cart ID.
            product: The product to remove from the cart.
        """
        self.logger.info("START: remove_from_cart")
        self.logger.info("Params-> cart_id: {}, product: {}".format(cart_id, product))
        for i in range(len(self.carts[cart_id - 1])):
            with self.lock:
                # Pre-condition: Find the product in the cart.
                if self.carts[cart_id - 1][i][1] == product:
                    # Block Logic: Identify the original producer and return the product to their stock.
                    producer_index = self.carts[cart_id - 1][i][0]
                    self.products_stock[producer_index].append(product)
                    self.carts[cart_id - 1].remove((producer_index, product))
                    self.logger.info("END: remove_from_cart")
                    return
        self.logger.info("END: remove_from_cart")

    def place_order(self, cart_id):
        """
        Finalizes the purchase for the items in the cart.

        Note: In this implementation, items are already removed from stock when added
        to the cart. This method effectively confirms the transaction and logs the
        purchased items. The cart itself is not cleared.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of the products that were purchased.
        """
        self.logger.info("START: place_order")
        self.logger.info("Params-> cart_id: {}".format(cart_id))
        for product in self.carts[cart_id - 1]:
            with self.lock:
                print("cons{} bought {}".format(cart_id, product[1]))
        self.logger.info("END: place_order")
        return [elem[1] for elem in self.carts[cart_id - 1]]

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace logic."""
    def setUp(self):
        """Initializes a marketplace and products for each test."""
        self.marketplace = Marketplace(4)
        self.coffee = Coffee('Arabic', 2, '0.25', 'MEDIUM')
        self.tea = Tea('CrazyLove', 2, 'Herbal')

    def test_register_producer(self):
        """Tests if producer registration provides sequential IDs."""
        r_c = self.marketplace.register_producer()
        self.assertEqual(r_c, 1)
        r_c = self.marketplace.register_producer()
        self.assertEqual(r_c, 2)
        r_c = self.marketplace.register_producer()
        self.assertEqual(r_c, 3)

    def test_new_cart(self):
        """Tests if new cart creation provides sequential IDs."""
        r_c = self.marketplace.new_cart()
        self.assertEqual(r_c, 1)
        r_c = self.marketplace.new_cart()
        self.assertEqual(r_c, 2)
        r_c = self.marketplace.new_cart()
        self.assertEqual(r_c, 3)
        r_c = self.marketplace.new_cart()
        self.assertEqual(r_c, 4)

    def test_publish(self):
        """Tests that publishing respects the producer's queue size limit."""
        id_ret = self.marketplace.register_producer()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)
        r_c = self.marketplace.publish(id_ret, self.tea)
        self.assertEqual(r_c, True)
        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)
        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)
        # This one should fail as the queue size is 4.
        r_c = self.marketplace.publish(id_ret, self.tea)
        self.assertEqual(r_c, False)

    def test_add_to_cart(self):
        """Tests that a product can be successfully added to a cart if available."""
        id_ret = self.marketplace.register_producer()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        id_ret = self.marketplace.new_cart()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.add_to_cart(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        # This should fail as no tea has been published.
        r_c = self.marketplace.add_to_cart(id_ret, self.tea)
        self.assertEqual(r_c, False)

    def test_remove_from_cart(self):
        """Tests that removing a product makes it available again."""
        id_ret = self.marketplace.register_producer()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.publish(id_ret, self.tea)
        self.assertEqual(r_c, True)

        id_ret = self.marketplace.new_cart()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.add_to_cart(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.add_to_cart(id_ret, self.tea)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.add_to_cart(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        self.marketplace.remove_from_cart(id_ret, self.tea)

        r_c = self.marketplace.place_order(id_ret)
        self.assertEqual(r_c, [self.coffee, self.coffee])

    def test_place_order(self):
        """Tests the end-to-end process of adding and placing an order."""
        id_ret = self.marketplace.register_producer()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.publish(id_ret, self.tea)
        self.assertEqual(r_c, True)

        id_ret = self.marketplace.new_cart()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.add_to_cart(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.place_order(id_ret)
        self.assertEqual(r_c, [self.coffee])

        r_c = self.marketplace.add_to_cart(id_ret, self.tea)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.add_to_cart(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.place_order(id_ret)
        self.assertEqual(r_c, [self.coffee, self.tea, self.coffee])


class Producer(Thread):
    """
    Represents a producer thread that continuously adds products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread.

        Args:
            products (list): A list of (product, quantity, sleep_time) tuples.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish.
            **kwargs: Additional keyword arguments for the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.id_prod = marketplace.register_producer()
        self.republish_wait_time = republish_wait_time

    def run(self):
        """The main execution loop for the producer.

        Continuously loops through its product list, publishing each one the
        specified number of times. Retries with a delay if the marketplace is full.
        """
        while True:
            for product in self.products:
                i = 0
                while i < product[1]:
                    # Attempt to publish until successful.
                    if self.marketplace.publish(self.id_prod, product[0]):
                        i = i + 1
                    else:
                        # If queue is full, wait and retry.
                        time.sleep(self.republish_wait_time)
                time.sleep(product[2])
