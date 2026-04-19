
"""
@file consumer.py
@brief Concurrent Marketplace architecture with partitioned mutual exclusion.

This module implements a robust Producer-Consumer marketplace using Python's 
threading primitives. It features fine-grained locking (partitioned locks for 
producers, carts, and global products) to minimize contention and ensure 
transactional integrity during product publishing, cart management, and order 
fulfillment.

Algorithm: Multi-threaded state synchronization with fine-grained locking.
Domain: Production Systems, Concurrency Patterns.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Independent execution unit representing a consumer performing shopping operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        :param carts: List of shopping tasks defined as command sequences.
        :param marketplace: Shared synchronization hub.
        :param retry_wait_time: Throttling backoff for resource contention.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        Main execution lifecycle for the consumer thread.
        Logic: Processes multiple shopping carts, executing add/remove instructions 
        before finalize checkout.
        """
        for cart in self.carts:
            
            id_cart = self.marketplace.new_cart()

            # Block Logic: Iterative processing of cart commands.
            for command in cart:
                com_type = command['type']
                product = command['product']
                quantity = command['quantity']

                for _ in range(0, quantity):
                    okk = False

                    if com_type == "add":
                        okk = self.marketplace.add_to_cart(id_cart, product)

                        # Logic: Retries product acquisition until success.
                        while not okk:
                            time.sleep(self.retry_wait_time)
                            okk = self.marketplace.add_to_cart(id_cart, product)

                    if com_type == "remove":
                        self.marketplace.remove_from_cart(id_cart, product)

            
            # Checkout Phase: Converts current cart into a permanent order.
            checkout = self.marketplace.place_order(id_cart)

            
            for item in checkout:
                print("{0} bought {1}".format(self.name, item))


from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
import time

class TestMarketplace(unittest.TestCase):
    def setUp(self):
        
        self.marketplace = Marketplace(1)
        self.products = []

        pr1 = product.Product
        pr1.name = "Lipton"
        pr1.price = 5
        pr2 = product.Product
        pr2.name = "Jacobs"

        pr2.price = 12
        self.products.append(pr1)
        self.products.append(pr2)

    def test_register_producer(self):
        pr1 = self.marketplace.register_producer()
        pr2 = self.marketplace.register_producer()
        
        self.assertEqual(pr1, 0)
        self.assertEqual(pr2, 1)

    def test_publish(self):
        pr1 = self.marketplace.register_producer()
        pr2 = self.marketplace.register_producer()

        okk = self.marketplace.publish(pr1, self.products[0])
        self.assertEqual(okk, True)
        okk = self.marketplace.publish(pr2, self.products[1])
        self.assertEqual(okk, True)
        
        okk = self.marketplace.publish(pr1, self.products[0])
        self.assertEqual(okk, False)

        
        self.assertEqual(self.marketplace.products[pr1][0], self.products[0])
        self.assertEqual(self.marketplace.products[pr2][0], self.products[1])

    def test_new_cart(self):
        cr1 = self.marketplace.new_cart()
        cr2 = self.marketplace.new_cart()
        
        self.assertEqual(cr1, 0)
        self.assertEqual(cr2, 1)

    def test_add_to_cart(self):
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(1, self.products[1])
        cr1 = self.marketplace.new_cart()
        cr2 = self.marketplace.new_cart()

        okk = self.marketplace.add_to_cart(cr1, self.products[0])
        self.assertEqual(okk, True)
        
        okk = self.marketplace.add_to_cart(3, self.products[0])
        self.assertEqual(okk, False)
        okk = self.marketplace.add_to_cart(cr2, self.products[1])
        self.assertEqual(okk, True)

        
        prod = self.marketplace.carts[cr1][0][1]
        self.assertEqual(self.products[0], prod)
        prod = self.marketplace.carts[cr2][0][1]
        self.assertEqual(self.products[1], prod)

    def test_remove_from_cart(self):
        cr1 = self.marketplace.new_cart()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(1, self.products[1])

        self.marketplace.add_to_cart(cr1, self.products[0])
        okk = self.marketplace.remove_from_cart(cr1, self.products[0])
        self.assertEqual(okk, True)

        
        okk = self.marketplace.remove_from_cart(cr1, self.products[1])
        self.assertEqual(okk, False)

    def test_place_order(self):
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(1, self.products[1])
        cr1 = self.marketplace.new_cart()

        self.marketplace.add_to_cart(cr1, self.products[0])
        self.marketplace.add_to_cart(cr1, self.products[1])

        
        order = self.marketplace.place_order(0)
        self.assertEqual(self.products[0], order[0])
        self.assertEqual(self.products[1], order[1])

class Marketplace:
    """
    Central hub for marketplace logic, managing shared state through partitioned locks.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        :param queue_size_per_producer: Capacity limit for product stock per producer.
        """
        self.queue_size_per_producer = queue_size_per_producer
        
        self.products = {} # Global product registry.
        
        self.carts = {} # Consumer cart registry.
        
        self.id_prod = -1
        self.id_cart = -1
        
        # Partitioned Locking: Decouples identifier generation from inventory updates.
        self.lock_prod_id = Lock()
        self.lock_cart_id = Lock()
        self.lock_products = Lock()

        # Monitoring: Configures audit trails with automatic log rotation.
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.INFO)

        handler = RotatingFileHandler("marketplace.log", maxBytes=2000, backupCount=10)

        formatter = logging.Formatter("%(asctime)s; %(message)s", "%Y-%m-%d %H:%M:%S")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)


    def register_producer(self):
        """
        Creates a new unique producer identifier.
        """
        self.logger.info('Register_producer input: N/A')
        with self.lock_prod_id:
            self.id_prod += 1
            result = self.id_prod

        self.logger.info('Register_producer output: ' + str(self.id_prod))
        return result

    def publish(self, producer_id, product):
        """
        Allows a producer to add a product to the global inventory.
        Logic: Enforces production quotas using thread-safe state checks.
        """
        self.logger.info("Publish input:" + str(producer_id) + " AND " + str(product))

        
        if producer_id in self.products.keys():

            
            if len(self.products[producer_id]) < self.queue_size_per_producer:

                # Critical Section: Thread-safe inventory increment.
                with self.lock_products:
                    self.products[producer_id].append(product)

                self.logger.info("Publish output: True")
                return True

        else:
            
            # Initialization: Bootstraps the first product for a new producer.
            self.products[producer_id] = [product]
            self.logger.info("Publish output: True")
            return True
        return False

    def new_cart(self):
        """
        Generates a new shopping session and associated cart.
        """
        self.logger.info("New_cart input: N/A")
        with self.lock_cart_id:
            self.id_cart += 1
            result = self.id_cart
            self.carts[self.id_cart] = []

        self.logger.info("New_cart output: " + str(self.id_cart))
        return result

    def add_to_cart(self, cart_id, product):
        """
        Atomic transfer of a product from any producer to a consumer cart.
        Logic: Performs a global search across all active producers.
        """
        self.logger.info("Add_to_cart input: " + str(cart_id) + " AND " + str(product))

        
        if cart_id > self.id_cart:
            self.logger.info("Add_to_cart output: False")
            return False

        # Block Logic: Inventory scanning for atomic acquisition.
        for prod_id in self.products.keys():
            
            if product in self.products[prod_id]:

                with self.lock_products:
                    # Critical Section: Multi-structure state update.
                    elem = (prod_id, product)
                    self.carts[cart_id].append(elem)
                    self.products[prod_id].remove(product)

                self.logger.info("Add_to_cart output: True")
                return True

        self.logger.info("Add_to_cart output: False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Reverts an item from a cart back to the global inventory.
        """
        self.logger.info("Remove_from_cart input:" + str(cart_id) + " AND " + str(product))

        
        if cart_id > self.id_cart:
            self.logger.info("Remove_from_cart output: False")
            return False

        for elem in self.carts[cart_id]:
            prod = elem[1]
            prod_id = elem[0]

            
            if prod == product:
                # Critical Section: Restoration of inventory state.
                with self.lock_products:
                    self.carts[cart_id].remove(elem)
                    self.products[prod_id].append(prod)

                self.logger.info("Remove_from_cart output: True")
                return True
        self.logger.info("Remove_from_cart output: False")
        return False


    def place_order(self, cart_id):
        """
        Finalizes order and purges session state.
        """
        self.logger.info("Place_order input: card_id = " + str(cart_id))

        prod_list = []

        
        for elem in self.carts[cart_id]:
            prod = elem[1]
            prod_list.append(prod)

        
        del self.carts[cart_id]



        self.logger.info("Place_order output:" + str(prod_list))
        return prod_list


from pickle import TRUE
from threading import Thread
import time

class Producer(Thread):
    """
    Supply-side background thread that continuously produces and publishes items.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        :param products: Catalog of product metadata (product, quantity, prod_time).
        :param marketplace: Reference to the shared hub.
        :param republish_wait_time: Throttling delay for full inventory.
        """
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_prod = -1
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        Infinite production lifecycle.
        Logic: Onboards with the marketplace and iteratively produces the catalog.
        """
        self.id_prod = self.marketplace.register_producer()

        while TRUE:
            okk = False

            for (product, quantity, prod_time) in self.products:
                for _ in range(0, quantity):
                    
                    # Logic: Simulated production time.
                    time.sleep(prod_time)
                    okk = self.marketplace.publish(self.id_prod, product)

                    
                    # Logic: Retries publishing until space permits.
                    while not okk:
                        time.sleep(self.republish_wait_time)
                        okk = self.marketplace.publish(self.id_prod, product)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Immutable base representation of a marketplace item.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Variety metadata for tea items.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Intensity metadata for coffee items.
    """
    acidity: str
    roast_level: str
