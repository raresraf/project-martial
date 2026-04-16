"""
This module implements a multi-threaded producer-consumer simulation of a marketplace.

It defines the following main classes:
- Marketplace: A thread-safe marketplace where producers can publish products and
  consumers can purchase them. It manages inventory, producers, and consumer carts.
- Producer: A thread that generates and publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places orders from the marketplace.

The simulation uses threading primitives like Locks to ensure data consistency
in a concurrent environment. It also includes a suite of unit tests to verify the
marketplace's functionality.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing products from the marketplace.

    This thread processes a list of shopping carts, where each cart contains a series
    of 'add' or 'remove' operations. It interacts with the `Marketplace` to perform
    these actions in a concurrent environment.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of cart operations to be performed. Each operation
                          is a dictionary specifying the 'type' ('add' or 'remove'),
                          'product', and 'quantity'.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying an
                                     'add' operation if the product is unavailable.
            **kwargs: Keyword arguments for the `Thread` constructor, including the
                      consumer's 'name'.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution logic for the consumer thread.

        For each cart provided during initialization, it creates a new cart in the
        marketplace, processes all 'add' and 'remove' operations, and finally places
        the order. If an 'add' operation fails because a product is unavailable, it
        will wait and retry until it succeeds.
        """
        
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()
            
            for operation in cart:
                
                if operation["type"] == "add":
                    for _ in range(operation["quantity"]):
                        while not self.marketplace.add_to_cart(cart_id, operation["product"]):
                            
                            sleep(self.retry_wait_time)
                elif operation["type"] == "remove":
                    
                    for _ in range(operation["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
            
            order = self.marketplace.place_order(cart_id)
            
            for product in order:
                print("{0} bought {1}".format(self.name, product))

import time
from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea


class Marketplace:
    """
    Implements a thread-safe marketplace for producers and consumers.

    This class manages product inventories, producer registrations, and consumer shopping
    carts. It uses a system of locks to ensure that concurrent operations from multiple
    producer and consumer threads do not lead to race conditions or inconsistent state.
    It maintains separate queues for each producer and tracks product availability globally.
    """
    

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed in the marketplace
                                           at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        
        self.producers_queue = {}
        
        self.carts = {}
        
        self.producer_id = 0
        
        self.cart_id = 0
        
        self.producer_id_lock = Lock()
        
        self.cart_id_lock = Lock()
        
        
        
        self.producers_locks = {}
        
        
        self.products_producers = {}
        
        
        
        
        
        self.products_locks = {}
        
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.INFO)
        self.handler = RotatingFileHandler("marketplace.log", maxBytes=1024 * 512, backupCount=20)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.formatter.converter = time.gmtime
        self.logger.addHandler(self.handler)

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        This method assigns a unique ID to the new producer, initializes their product
        queue, and creates a dedicated lock for them. The operation is thread-safe.

        Returns:
            str: The unique ID string for the newly registered producer (e.g., 'prod0').
        """
        self.logger.info("Entered register_producer()!")
        
        self.producer_id_lock.acquire()
        
        producer_id_string = "prod{0}".format(self.producer_id)
        
        self.producers_queue[producer_id_string] = 0
        
        self.producers_locks[producer_id_string] = Lock()
        
        self.producer_id += 1
        
        self.producer_id_lock.release()
        self.logger.info("Finished register_producer(): returned producer_id: %s!",
                         producer_id_string)
        return producer_id_string

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        If the producer's queue is already full, the publication fails. Otherwise, the
        product is added to a global list of available products, and the producer's
        queue count is incremented. Access to both producer and product data
        structures is protected by locks.

        Args:
            producer_id (str): The ID of the producer.
            product: The product object to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise.
        """
        self.logger.info("Entered publish(%s, %s)!", producer_id, product)
        
        self.producers_locks[producer_id].acquire()
        
        queue_size = self.producers_queue[producer_id]
        
        if queue_size == self.queue_size_per_producer:
            
            self.producers_locks[producer_id].release()
            self.logger.info("Finished publish(%s, %s): Queue is Full!",
                             producer_id, product)
            return False
        
        if product not in self.products_producers:
            self.products_locks[product] = Lock()
            self.products_locks[product].acquire()
            self.products_producers[product] = []
        else:
            self.products_locks[product].acquire()
        self.products_producers[product].append(producer_id)
        self.products_locks[product].release()
        
        self.producers_queue[producer_id] += 1
        


        self.producers_locks[producer_id].release()
        self.logger.info("Finished publish(%s, %s): Published product!",
                         producer_id, product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Assigns a unique, sequential ID to the new cart. This operation is thread-safe.

        Returns:
            int: The unique integer ID for the new cart.
        """
        self.logger.info("Entered new_cart()!")
        
        self.cart_id_lock.acquire()
        cart_id = self.cart_id
        
        self.carts[cart_id] = []
        
        self.cart_id += 1
        
        self.cart_id_lock.release()
        self.logger.info("Finished new_cart(): New cart: %d!", cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds an available product to a specified shopping cart.

        This method atomically finds an available instance of the specified product,
        removes it from the global pool of available products, and adds it to the
        consumer's cart, tracking which producer it came from.

        Args:
            cart_id (int): The ID of the shopping cart.
            product: The product object to add.

        Returns:
            bool: True if the product was successfully added, False if the cart does
                  not exist or the product is unavailable.
        """
        self.logger.info("Entered add_to_cart(%d, %s)!", cart_id, product)
        
        if cart_id not in self.carts:
            self.logger.info("Finished add_to_cart(%d, %s): Cart doesn't exist!",
                             cart_id, product)
            return False
        
        if product not in self.products_producers:
            self.logger.info("Finished add_to_cart(%d, %s): Product is not available!",
                             cart_id, product)
            return False
        self.products_locks[product].acquire()
        if not self.products_producers[product]:
            self.products_locks[product].release()
            self.logger.info("Finished add_to_cart(%d, %s): Product is not available!",
                             cart_id, product)
            return False
        
        
        producer_id = self.products_producers[product].pop(0)
        self.products_locks[product].release()
        
        
        
        self.carts[cart_id].append({"product": product,
                                    "producer_id": producer_id})
        self.logger.info("Finished add_to_cart(%d, %s): Product added to cart!",
                         cart_id, product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the pool of
        available products.

        This method finds the specified product within the cart, removes it, and
        atomically adds it back to the list of available products associated with its
        original producer.

        Args:
            cart_id (int): The ID of the shopping cart.
            product: The product object to remove.

        Returns:
            bool: True if the product was found and removed, False otherwise.
        """
        self.logger.info("Entered remove_from_cart(%d, %s)!", cart_id, product)
        
        if cart_id not in self.carts:
            self.logger.info("Finished remove_from_cart(%d, %s): Cart doesn't exist!",
                             cart_id, product)
            return False
        
        cart_list = self.carts[cart_id]
        
        for cart_element in cart_list:
            if cart_element["product"] == product:
                
                producer_id = cart_element["producer_id"]
                self.products_producers[product].append(producer_id)
                
                self.carts[cart_id].remove(cart_element)
                self.logger.info("Finished remove_from_cart(%d, %s): Product removed from cart!",
                                 cart_id, product)
                return True
        self.logger.info("Finished remove_from_cart(%d, %s): Product not found in cart!",
                         cart_id, product)
        return False

    def place_order(self, cart_id):
        """
        Finalizes an order, consuming the products in the cart.

        For each product in the cart, this method decrements the corresponding
        producer's queue count, effectively removing the product from the marketplace
        permanently. It then clears the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of the product objects that were purchased, or None if the
                  cart does not exist.
        """
        self.logger.info("Entered place_order(%d)!", cart_id)
        result = []
        
        if cart_id not in self.carts:
            self.logger.info("Finished place_order(%d): Cart doesn't exist!", cart_id)
            return None
        
        cart_list = self.carts[cart_id]
        
        for cart_element in cart_list:
            product = cart_element["product"]
            result.append(product)
            producer_id = cart_element["producer_id"]
            
            self.producers_locks[producer_id].acquire()
            self.producers_queue[producer_id] -= 1
            self.producers_locks[producer_id].release()
        
        self.carts[cart_id] = []
        self.logger.info("Finished place_order(%d): Order placed: %s!", cart_id, result)
        return result


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class, verifying the correctness of its
    producer-consumer logic under various scenarios.
    """

    def setUp(self):
        
        self.marketplace = Marketplace(5)
        self.product0 = Coffee(name="Indonezia", acidity="5.05", roast_level="MEDIUM", price=1)
        self.product1 = Tea(name="Linden", type="Herbal", price=9)
        self.product2 = Coffee(name="Ethiopia", acidity="5.09", roast_level="MEDIUM", price=10)
        self.product3 = Coffee(name="Arabica", acidity="5.02", roast_level="MEDIUM", price=9)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 'prod0',
                         'Incorrect producer_id assigned for first producer!')
        self.assertEqual(self.marketplace.register_producer(), 'prod1',
                         'Incorrect producer_id assigned for second producer!')
        self.assertEqual(self.marketplace.register_producer(), 'prod2',
                         'Incorrect producer_id assigned for third producer!')

    def test_publish(self):
        
        self.test_register_producer()
        
        for _ in range(3):
            check = self.marketplace.publish('prod0', self.product0)
            self.assertTrue(check, 'Producer prod0 should be able to publish product!')
        
        for _ in range(2):
            check = self.marketplace.publish('prod0', self.product1)
            self.assertTrue(check, 'Producer prod0 should be able to publish product!')
        
        check = self.marketplace.publish('prod0', self.product0)
        self.assertFalse(check, 'Producer prod0 should not be able to publish product!')
        
        for _ in range(2):
            check = self.marketplace.publish('prod1', self.product2)
            self.assertTrue(check, 'Producer prod1 should be able to publish product!')
        
        check = self.marketplace.publish('prod1', self.product3)
        self.assertTrue(check, 'Producer prod1 should be able to publish product!')
        
        check = self.marketplace.publish('prod1', self.product1)
        self.assertTrue(check, 'Producer prod1 should be able to publish product!')
        
        self.assertEqual(self.marketplace.producers_queue['prod0'], 5,
                         'Producer prod0 queue should be full!')
        self.assertEqual(self.marketplace.producers_queue['prod1'], 4,
                         'Producer prod1 queue size should be 4!')
        
        self.assertEqual(len(self.marketplace.products_producers[self.product0]), 3,
                         'Product0 should be available in quantity = 3!')
        self.assertEqual(len(self.marketplace.products_producers[self.product1]), 3,
                         'Product1 should be available in quantity = 3!')
        self.assertEqual(len(self.marketplace.products_producers[self.product2]), 2,
                         'Product2 should be available in quantity = 2!')
        self.assertEqual(len(self.marketplace.products_producers[self.product3]), 1,
                         'Product3 should be available in quantity = 1!')

    def test_new_cart(self):
        
        self.test_publish()
        
        self.assertEqual(self.marketplace.new_cart(), 0,
                         'Incorrect cart_id assigned for first cart!')
        self.assertEqual(self.marketplace.new_cart(), 1,
                         'Incorrect cart_id assigned for second cart!')
        self.assertEqual(self.marketplace.new_cart(), 2,
                         'Incorrect cart_id assigned for third cart!')
        self.assertEqual(self.marketplace.new_cart(), 3,
                         'Incorrect cart_id assigned for fourth cart!')
        
        for i in range(4):
            self.assertEqual(self.marketplace.carts[i], [],
                             'Cart should be empty!')

    def test_add_to_cart(self):
        
        self.test_new_cart()
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.product0),
                        'Cannot add product0 to cart!')
        
        for _ in range(3):
            self.assertTrue(self.marketplace.add_to_cart(0, self.product1),
                            'Cannot add product1 to cart!')
        
        
        self.assertFalse(self.marketplace.add_to_cart(0, self.product1),
                         'Should not be able to add product1 to cart!')
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.product2),
                        'Cannot add product2 to cart!')
        
        self.assertEqual(len(self.marketplace.carts[0]), 5,
                         'Wrong number of products added to cart!')
        
        self.assertTrue(self.marketplace.add_to_cart(1, self.product2),
                        'Cannot add product2 to cart!')
        
        
        self.assertFalse(self.marketplace.add_to_cart(1, self.product2),
                         'Should not be able to add product2 to cart!')
        
        self.assertEqual(len(self.marketplace.carts[1]), 1,
                         'Wrong number of products added to cart!')
        
        self.assertEqual(len(self.marketplace.products_producers[self.product0]), 2,
                         'Product0 should be available in quantity = 0!')
        self.assertEqual(len(self.marketplace.products_producers[self.product1]), 0,
                         'Product1 should be available in quantity = 3!')
        self.assertEqual(len(self.marketplace.products_producers[self.product2]), 0,
                         'Product2 should be available in quantity = 2!')
        self.assertEqual(len(self.marketplace.products_producers[self.product3]), 1,
                         'Product3 should be available in quantity = 1!')

    def test_remove_from_cart(self):
        
        self.test_add_to_cart()
        
        self.assertTrue(self.marketplace.remove_from_cart(0, self.product1),
                        'Cannot remove product1 from cart!')
        
        self.assertEqual(len(self.marketplace.carts[0]), 4,
                         'Wrong number of products in cart0!')
        
        
        self.assertFalse(self.marketplace.remove_from_cart(0, self.product3),
                         'Should not be able to remove this product!')
        
        self.assertEqual(len(self.marketplace.products_producers[self.product1]), 1,
                         'Product1 should be available in quantity = 1!')

    def test_place_order(self):
        
        self.test_remove_from_cart()
        
        self.assertEqual(self.marketplace.place_order(0),
                         [self.product0, self.product1, self.product1, self.product2],
                         'Wrong cart list!')
        
        self.assertEqual(self.marketplace.producers_queue['prod0'], 3,
                         'Producer prod0 queue contain 3 products!')
        self.assertEqual(self.marketplace.producers_queue['prod1'], 2,
                         'Producer prod1 queue should contain 2 products!')
        
        self.assertEqual(self.marketplace.carts[0], [],
                         'Cart0 should be empty!')


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that generates and publishes products to the marketplace.

    This thread registers with the marketplace and then enters an infinite loop,
    producing a list of specified products at a given rate. If the marketplace
    is full, it will wait and retry publishing.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains the
                             product info, quantity to produce, and production time.
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before retrying
                                         to publish if the queue is full.
            **kwargs: Keyword arguments for the `Thread` constructor.
        """
        
        Thread.__init__(self, daemon=True)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution logic for the producer thread.

        After registering with the marketplace, it enters an infinite loop to produce
        and publish items. For each item in its product list, it sleeps for the
        specified production time and then attempts to publish the item, retrying
        on failure.
        """
        
        producer_id = self.marketplace.register_producer()
        
        while True:
            
            for element in self.products:
                
                product = element[0]
                quantity = element[1]
                production_time = element[2]
                
                sleep(production_time)
                
                for _ in range(quantity):
                    while not self.marketplace.publish(producer_id, product):
                        
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A data class for a generic product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
