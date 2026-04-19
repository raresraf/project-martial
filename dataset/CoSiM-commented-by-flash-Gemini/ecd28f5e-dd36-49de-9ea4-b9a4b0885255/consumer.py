
"""
@file consumer.py
@brief Thread-safe multi-cart Marketplace simulation with partitioned state locking.

This module implements a Producer-Consumer marketplace model using Python threads. 
It features a central Marketplace registry that partitions state into producer 
and consumer segments, each protected by dedicated locks. The implementation 
supports multiple carts per consumer and employs a two-tier locking strategy 
to ensure atomic state transitions across global and private inventory structures.

Algorithm: Concurrent state management with partitioned mutual exclusion.
Domain: Multi-threaded Production Systems.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Independent consumer thread executing a sequence of multi-cart shopping operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        :param carts: A list of carts, each containing a sequence of shopping instructions.
        :param marketplace: Reference to the shared synchronization hub.
        :param retry_wait_time: Throttling delay for failed item acquisitions.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.cart_ids = []

    def run(self):
        """
        Main execution lifecycle for the consumer.
        Logic: Iteratively creates new carts and processes their respective 
        operations (add/remove) before finalizing each order.
        """
        index = 0
        for cart in self.carts:
            # Initialization: Allocates a unique cart for the current task set.
            self.cart_ids.append(self.marketplace.new_cart())
            
            for op_cart in cart:
                product = op_cart['product']
                quantity = op_cart['quantity']
                op_type = op_cart['type']
                
                # Block Logic: Operation routing.
                if op_type == "add":
                    i = 0
                    while i < quantity:
                        # Logic: Retries addition until item is successfully reserved.
                        status = self.marketplace.add_to_cart(self.cart_ids[index], product)
                        if status:
                            i += 1
                        else:
                            sleep(self.wait_time)
                elif op_type == "remove":
                    # Logic: Batch restoration of items from cart to global inventory.
                    for i in range(0, quantity):
                        self.marketplace.remove_from_cart(self.cart_ids[index], product)
            
            # Finalization: Converts finalized cart into a purchase record.
            self.marketplace.place_order(self.cart_ids[index])
            index += 1

import threading
import time
from threading import Lock
import logging.handlers
import unittest

from tema.product import Tea


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(5)
        self.product1 = Tea('Test_tea1', 0, 'Test_tea2')
        self.product2 = Tea('Test_tea3', 0, 'Test_tea4')

    def test_register_producer(self):
        
        prod_id = self.marketplace.last_producer_id
        self.assertEqual(self.marketplace.register_producer(),
                         prod_id)

    def test_publish(self):
        
        id_prod = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(id_prod,
                                                 self.product1))
        self.assertEqual(len(self.marketplace.producers[0]), 1)

    def test_new_cart(self):
        
        cart_id = self.marketplace.last_cart_id
        self.assertEqual(self.marketplace.new_cart(), cart_id)

    def test_add_to_cart(self):
        
        cart_id = self.marketplace.new_cart()
        cart_len = len(self.marketplace.carts[cart_id])
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.product1))
        self.assertGreater(len(self.marketplace.carts[cart_id]), cart_len)

    def test_remove_from_cart(self):
        
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.product1)
        cart_len = len(self.marketplace.carts[cart_id])
        self.marketplace.remove_from_cart(cart_id, self.product1)
        self.assertLess(len(self.marketplace.carts[cart_id]), cart_len)

    def test_place_order(self):
        
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.product1)
        self.marketplace.add_to_cart(cart_id, self.product2)
        cart_len = len(self.marketplace.carts[cart_id])
        self.assertEqual(len(self.marketplace.place_order(cart_id)), cart_len)


class Marketplace:
    """
    Central coordination hub for product transactions, managing producer and consumer state tiers.
    """
    

    def __init__(self, queue_size_per_producer):
        """
        :param queue_size_per_producer: Capacity limit for producer-specific inventory buffers.
        """
        self.queue_max_size = queue_size_per_producer
        
        
        self.producers = [] # Registry of producer partitions.
        
        


        self.carts = [] # Registry of active consumer sessions.
        self.last_producer_id = 0
        self.last_cart_id = 0
        
        # Partitioned Locking: Separates producer-side updates from consumer-side cart operations.
        self.lock1 = Lock()
        self.lock2 = Lock()

        # Monitoring: Configures audit trails with rotating log files.
        logging.basicConfig(handlers=[logging.handlers.RotatingFileHandler("marketplace.log",
                                                                           mode='a',
                                                                           maxBytes=5000,
                                                                           backupCount=5)],
                            level=logging.INFO,
                            format=
                            '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')
        logging.Formatter.converter = time.gmtime

    def register_producer(self):
        """
        Onboards a new producer with an empty inventory slot and a unique ID.
        """
        logging.info("Entered register_producer")
        self.producers.append([])
        with self.lock1:
            # Logic: Atomic identifier generation.
            self.last_producer_id += 1
            id_prod = self.last_producer_id - 1


        logging.info("New producer id: " + str(id_prod))
        return id_prod

    def publish(self, producer_id, product):
        """
        Submits a product to a producer's partition.
        Logic: Enforces per-producer quotas.
        """
        logging.info("Entered publish with producer id " + str(producer_id)
                     + " and product " + str(product))
        if len(self.producers[producer_id]) == self.queue_max_size:
            logging.info("Return value: False")
            return False
        with self.lock1:
            # Critical Section: Update partitioned inventory.
            self.producers[producer_id].append((product, 1))
        logging.info("Return value: True")
        return True

    def new_cart(self):
        """
        Initializes a new shopping session for a consumer.
        """
        logging.info("Entered new_cart")
        with self.lock2:
            # Logic: Atomic cart identifier generation.
            self.last_cart_id += 1
            self.carts.append([])
            cart_id = self.last_cart_id - 1
        logging.info("Returned cart id: " + str(cart_id))
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Atomic acquisition of a product from global stock.
        Logic: Scans all producer partitions for availability and reserves the item.
        """
        logging.info("Entered add_to_cart with cart id " + str(cart_id)
                     + " and product " + str(product))
        for tmp_list in self.producers:
            for prod_tuple in tmp_list:
                tmp = list(prod_tuple)
                if tmp[0] == product:
                    if tmp[1] == 0:
                        logging.info("Return value: False")
                        return False
                    with self.lock1:
                        # Critical Section: Soft-reserve the item.
                        tmp[1] = 0
                        prod_tuple = tuple(tmp)

        with self.lock2:
            # Critical Section: Finalize transfer to consumer cart.
            self.carts[cart_id].append(product)
        logging.info("Return value: True")
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Reverts an item from a cart back to the producer stock pool.
        """
        logging.info("Entered remove_from_cart with cart id " +
                     str(cart_id) + " and product " + str(product))
        with self.lock2:
            # Critical Section: Purge from cart.
            self.carts[cart_id].remove(product)

        for tmp_list in self.producers:
            for prod_tuple in tmp_list:
                tmp = list(prod_tuple)
                if tmp[0] == product:
                    with self.lock1:
                        # Critical Section: Restore availability.
                        tmp[1] = 1
                        prod_tuple = tuple(tmp)

    def place_order(self, cart_id):
        """
        Executes the final transaction, purging purchased items from the registry.
        """
        logging.info("Entered place_order with cart id " + str(cart_id))
        prod_list = self.carts[cart_id]
        for prod_iter in prod_list:
            for list_prod in self.producers:
                if prod_iter in list_prod:
                    with self.lock1:
                        # Logic: Permanent removal from the marketplace.
                        list_prod.remove(prod_iter)
        self.carts[cart_id] = []
        for prod_iter in prod_list:
            print(threading.current_thread().name + " bought " + str(prod_iter))
            logging.info("Buyer " + threading.current_thread().name
                         + " bought " + str(prod_iter))
        return prod_list


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Supply-side background thread that continuously produces and publishes catalog items.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        :param products: Catalog of product metadata (product, quantity, wait_time).
        :param marketplace: Shared hub for publishing.
        :param republish_wait_time: Throttling delay for full inventory.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time
        self.id_prod = -1

    def run(self):
        """
        Main lifecycle loop.
        Logic: Onboards with the marketplace and iteratively produces the catalog 
        while respecting capacity constraints and manufacturing times.
        """
        self.id_prod = self.marketplace.register_producer()
        index = 0
        while True:
            # Logic: Infinite cycle over the assigned product list.
            if index == len(self.products):
                index = 0
            i = 0
            while i < self.products[index][1]:
                # Logic: Attempts to publish until marketplace accepts.
                status = self.marketplace.publish(self.id_prod, self.products[index][0])
                if not status:
                    sleep(self.wait_time)
                else:
                    # Logic: Simulated production time.
                    sleep(self.products[index][2])
                    i += 1

            index += 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Immutable representation of a generic marketplace item.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Variety specialization for tea products.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Quality specialization for coffee products.
    """
    acidity: str
    roast_level: str
