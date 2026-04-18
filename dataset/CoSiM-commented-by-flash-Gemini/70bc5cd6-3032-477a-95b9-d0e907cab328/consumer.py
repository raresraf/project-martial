
"""
@70bc5cd6-3032-477a-95b9-d0e907cab328/consumer.py
@brief Synchronized marketplace simulation with automated transaction validation.
This file implements a concurrent producer-consumer system where multiple 
agents interact via a centralized broker. It ensures thread safety through 
granular mutex locking and features a comprehensive suite of unit tests 
to verify state consistency across product publication, acquisition, 
and order finalization.

Domain: Concurrent Programming, Synchronization Primitives, Unit Testing.
"""


from threading import Thread
import time

class Consumer(Thread):
    """
    Functional Utility: Represent a consumer agent that processes assigned shopping schedules.
    Logic: Iterates through shopping carts, executing batch 'add' or 'remove' 
    operations. It implements a retry mechanism with backoff for stock 
    depletion and finalizes transactions via the marketplace broker.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Binds the consumer thread to its carts and shared marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = marketplace.new_cart()

    def run(self):
        """
        Execution Logic: Orchestrates the multi-cart processing workflow.
        """
        for k in range(len(self.carts)):
            for elem in self.carts[k]:
                if elem['type'] == 'add':
                    /**
                     * Block Logic: Synchronized product acquisition.
                     * Logic: Repeatedly attempts to add products to the cart. 
                     * If stock is unavailable, it sleeps before re-evaluating state.
                     */
                    for i in range(elem['quantity']):
                        while not self.marketplace.add_to_cart(self.cart_id, elem['product']):
                            time.sleep(self.retry_wait_time)

                if elem['type'] == 'remove':
                    /**
                     * Block Logic: Product return.
                     */
                    for i in range(elem['quantity']):
                        self.marketplace.remove_from_cart(self.cart_id, elem['product'])

        self.marketplace.place_order(self.cart_id)

from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import unittest
from product import Coffee
from product import Tea

class Marketplace:
    """
    Functional Utility: Thread-safe coordinator for product exchange.
    Logic: Tracks registered producers, available product pools, and consumer 
    carts. It maintains thread safety via a global lock and generates 
    rotating audit logs for all transactional state transitions.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes storage and synchronization locks.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.cart_id = 0
        self.products_stock = []
        self.carts = []


        self.lock = Lock()
        
        # Block Logic: Audit logging configuration.
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler("marketplace.log", maxBytes=200000, backupCount=10)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Functional Utility: Atomically registers a new producer and initializes its slot.
        """
        self.logger.info("START: register_producer")
        with self.lock:
            self.id_producer += 1
            self.products_stock.append([])
        self.logger.info("END: register_producer")
        return self.id_producer

    def publish(self, producer_id, product):
        """
        Functional Utility: Adds a product instance to a producer's inventory.
        Logic: Enforces per-producer capacity limits. Returns True on success.
        """
        self.logger.info("START: publish")
        self.logger.info("Params-> producer_id: {}, product: {}".format(producer_id, product))
        with self.lock:
            if len(self.products_stock[producer_id - 1]) >= self.queue_size_per_producer:
                self.logger.info("END: publish")
                return False

            self.products_stock[producer_id - 1].append(product)

        self.logger.info("END: publish")
        return True

    def new_cart(self):
        """
        Functional Utility: Allocates a new unique shopping cart for a consumer.
        """
        self.logger.info("START: new_cart")
        with self.lock:
            self.cart_id += 1
            self.carts.append([])
        self.logger.info("END: new_cart")
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Transfers product from producer stock to consumer cart.
        Logic: Iterates through all producer pools. If an instance is found, 
        it performs an atomic ownership migration to the specified cart.
        """
        self.logger.info("START: add_to_cart")
        self.logger.info("Params-> cart_id: {}, product: {}".format(cart_id, product))

        for i in range(len(self.products_stock)):
            with self.lock:
                if self.products_stock[i].count(product) > 0:
                    self.carts[cart_id - 1].append((i, product))
                    self.products_stock[i].remove(product)
                    self.logger.info("START: add_to_cart")
                    return True
        self.logger.info("END: add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Restores a product from a cart back to the marketplace.
        """
        self.logger.info("START: remove_from_cart")
        self.logger.info("Params-> cart_id: {}, product: {}".format(cart_id, product))
        for i in range(len(self.carts[cart_id - 1])):
            with self.lock:
                if self.carts[cart_id - 1][i][1] == product:
                    producer_index = self.carts[cart_id - 1][i][0]
                    self.products_stock[producer_index].append(product)
                    self.carts[cart_id - 1].remove((producer_index, product))
                    self.logger.info("END: remove_from_cart")
                    return
        self.logger.info("END: remove_from_cart")

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes the transaction and returns purchased items.
        """
        self.logger.info("START: place_order")
        self.logger.info("Params-> cart_id: {}".format(cart_id))
        for product in self.carts[cart_id - 1]:
            with self.lock:
                print("cons{} bought {}".format(cart_id, product[1]))
        self.logger.info("END: place_order")
        return [elem[1] for elem in self.carts[cart_id - 1]]

class TestMarketplace(unittest.TestCase):
    """
    Functional Utility: Integrity validation suite for Marketplace transaction logic.
    """
    
    def setUp(self):
        """
        Pre-condition: Initialize a marketplace and mock agents for testing.
        """
        self.marketplace = Marketplace(4)
        self.coffee = Coffee('Arabic', 2, '0.25', 'MEDIUM')
        self.tea = Tea('CrazyLove', 2, 'Herbal')

    def test_register_producer(self):
        
        r_c = self.marketplace.register_producer()
        self.assertEqual(r_c, 1)
        r_c = self.marketplace.register_producer()
        self.assertEqual(r_c, 2)
        r_c = self.marketplace.register_producer()
        self.assertEqual(r_c, 3)

    def test_new_cart(self):
        
        r_c = self.marketplace.new_cart()
        self.assertEqual(r_c, 1)
        r_c = self.marketplace.new_cart()
        self.assertEqual(r_c, 2)
        r_c = self.marketplace.new_cart()
        self.assertEqual(r_c, 3)
        r_c = self.marketplace.new_cart()
        self.assertEqual(r_c, 4)

    def test_publish(self):
        
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
        r_c = self.marketplace.publish(id_ret, self.tea)
        self.assertEqual(r_c, False)

    def test_add_to_cart(self):
        
        id_ret = self.marketplace.register_producer()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.publish(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        id_ret = self.marketplace.new_cart()
        self.assertEqual(id_ret, 1)

        r_c = self.marketplace.add_to_cart(id_ret, self.coffee)
        self.assertEqual(r_c, True)

        r_c = self.marketplace.add_to_cart(id_ret, self.tea)
        self.assertEqual(r_c, False)

    def test_remove_from_cart(self):
        
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


from threading import Thread
import time


class Producer(Thread):
    """
    Functional Utility: Represent a production agent that continuously generates supply.
    Logic: Iterates through its inventory, attempting to publish products to the 
    marketplace. It incorporates simulated production delays and handles 
    congestion via a timed retry mechanism.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers the producer and initializes production parameters.
        """

        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.id_prod = marketplace.register_producer()
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        Execution Logic: Infinite production loop.
        """
        while True:
            for product in self.products:
                i = 0
                while i < product[1]:
                    if self.marketplace.publish(self.id_prod, product[0]):
                        i = i + 1
                    else:
                        # Block Logic: Congestion backoff.
                        time.sleep(self.republish_wait_time)
                    
                    # Inline: Simulated item manufacturing time.
                    time.sleep(product[2])
