
"""
@51de32d0-a1b3-4ad7-a07f-dfab6fbafcfd/consumer.py
@brief Threaded marketplace simulation with automated state validation.
This file implements a concurrent producer-consumer system where agents interact 
via a central marketplace broker. It ensures thread safety through multiple mutex 
locks and includes a comprehensive test suite to verify the transactional 
integrity of product exchange and cart management.

Domain: Concurrent Programming, Synchronization Primitives, Unit Testing.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Functional Utility: Represent a consumer thread that iterates through multiple shopping carts.
    Logic: For each cart, it performs 'add' or 'remove' operations. It implements 
    a 'safe_add_to_cart' method that retries acquisitions if the marketplace is 
    temporarily out of stock.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Binds the consumer to its shopping carts and the shared marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def safe_add_to_cart(self, cart_id, product):
        """
        Block Logic: Synchronized product acquisition with backoff.
        Logic: Continuously attempts to add a product to the cart. If the 
        marketplace rejects the request (insufficient stock), it sleeps before 
        re-attempting.
        """
        while not self.marketplace.add_to_cart(cart_id, product):
            sleep(self.retry_wait_time)

    def run(self):
        """
        Execution Logic: Orchestrates the execution of all cart operations.
        Invariant: All orders within a cart are finalized before printing the 
        results to the console.
        """
        for cart in self.carts:
            my_cart = self.marketplace.new_cart()

            # Block Logic: Sequential execution of cart commands.
            for crt_op in cart:
                if crt_op["type"] == "add":
                    for _ in range(crt_op["quantity"]):
                        self.safe_add_to_cart(my_cart, crt_op["product"])
                elif crt_op["type"] == "remove":
                    for _ in range(crt_op["quantity"]):
                        self.marketplace.remove_from_cart(my_cart, crt_op["product"])
                else:
                    print("[Error] No such operation")

            # Block Logic: Order finalization and reporting.
            ordered_prods = self.marketplace.place_order(my_cart)
            for prod in ordered_prods:
                /**
                 * Inline: Exclusive access to stdout to prevent interleaved logs 
                 * from multiple consumer threads.
                 */
                self.marketplace.print_lock.acquire()
                print(self.name, "bought", prod)
                self.marketplace.print_lock.release()

import time
import unittest
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler

# Block Logic: Configuration of audit logging for the marketplace.
logging.basicConfig(
    handlers=[RotatingFileHandler('./marketplace.log', maxBytes=2000, backupCount=5)],


    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s"
)
logging.Formatter.converter = time.gmtime

class Marketplace:
    """
    Functional Utility: Centralized broker for thread-safe product transactions.
    Logic: Tracks registered producers, available stocks, and consumer carts. 
    It maintains thread safety through a set of granular locks (producer_reg, 
    new_cart, products, print) to minimize global contention.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes internal storage and synchronization primitives.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.next_cart_id = 0
        self.next_producer_id = 0
        self.producers = {}
        self.carts = {}
        self.producers_stock = {}

        self.producer_reg_lock = Lock()
        self.new_cart_lock = Lock()
        self.products_lock = Lock()
        self.print_lock = Lock()

    def register_producer(self):
        """
        Functional Utility: Issues a unique ID to a new producer.
        """
        logging.info('register_producer was called')
        self.producer_reg_lock.acquire()
        new_producer_id = str(self.next_producer_id)
        self.next_producer_id = self.next_producer_id + 1
        self.producer_reg_lock.release()

        self.producers[new_producer_id] = []
        self.producers_stock[new_producer_id] = 0

        logging.info("register_producer done: id = %s", new_producer_id)
        return new_producer_id

    def publish(self, producer_id, product):
        """
        Functional Utility: Adds a product instance to a producer's stock.
        Logic: Enforces per-producer capacity limits. Returns True on success.
        """
        logging.info("publish was called: producer_id = %s; \
                    prod = %s", producer_id, str(product))



        if self.producers_stock[producer_id] >= self.queue_size_per_producer:
            logging.info("publish done")
            return False

        self.producers[producer_id].append(product)
        self.producers_stock[producer_id] = self.producers_stock[producer_id] + 1

        logging.info("publish done")
        return True

    def new_cart(self):
        """
        Functional Utility: Creates a unique shopping cart for a consumer.
        """
        logging.info("new_cart was called")
        self.new_cart_lock.acquire()
        my_id = self.next_cart_id
        self.next_cart_id = self.next_cart_id + 1
        self.new_cart_lock.release()

        self.carts[my_id] = []
        logging.info("new_cart done: id = %s", str(my_id))
        return my_id

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Atomically migrates a product from a producer to a cart.
        Logic: Scans all producer stocks for the product. If found, it removes 
        it from the producer's inventory and appends it to the consumer's cart.
        """
        logging.info("add_to_cart was called: cart_id = %s; \
                    prod = %s", str(cart_id), str(product))
        if cart_id not in self.carts:
            return False

        self.products_lock.acquire()
        for producer in self.producers:
            for crt_prod in self.producers[producer]:
                if crt_prod == product:
                    self.carts[cart_id].append((product, producer))
                    self.producers[producer].remove(crt_prod)
                    self.products_lock.release()
                    logging.info("add_to_cart done: True")
                    return True


        self.products_lock.release()
        logging.info("add_to_cart done: False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Reverses a product acquisition.
        Logic: Returns the product to the respective producer's inventory 
        and removes it from the cart metadata.
        """
        logging.info("remove_from_cart was called: cart_id = %s; \
                        product = %s", str(cart_id), str(product))
        if cart_id not in self.carts:
            logging.info("remove_from_cart done")
            return

        for (prod, producer) in self.carts[cart_id]:
            if prod == product:
                self.products_lock.acquire()
                self.producers[producer].append(prod)
                self.products_lock.release()


                self.carts[cart_id].remove((prod, producer))
                break
        logging.info("remove_from_cart done")

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes order and releases capacity tokens.
        Logic: Updates the producers' stock occupancy counts and returns 
        the purchased product list.
        """
        logging.info("place_order called")
        result = []

        if cart_id not in self.carts:
            logging.info("place_order done: %s", result)
            return result

        for (prod, producer) in self.carts[cart_id]:
            result.append(prod)
            self.products_lock.acquire()
            self.producers_stock[producer] = self.producers_stock[producer] - 1
            self.products_lock.release()

        logging.info("place_order done: %s", result)
        return result

class TestMarketplace(unittest.TestCase):
    """
    Functional Utility: Validation suite for Marketplace functional requirements.
    """
    
    def setUp(self):
        """
        Pre-condition: Initialize a marketplace with a fixed capacity per producer.
        """
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        
        for i in range(1000):
            self.assertEqual(self.marketplace.register_producer(), str(i))

    def test_publish(self):
        
        self.marketplace.register_producer()
        for i in range(2):
            self.assertEqual(self.marketplace.publish("0", str(i)), True)
        for i in range(2):
            self.assertEqual(self.marketplace.publish("0", str(i)), False)

        self.marketplace.register_producer()
        for i in range(2):
            self.assertEqual(self.marketplace.publish("1", str(i)), True)
        for i in range(2):
            self.assertEqual(self.marketplace.publish("1", str(i)), False)

    def test_new_cart(self):
        
        for i in range(1000):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_add_to_cart(self):
        
        for i in range(10):
            self.assertEqual(self.marketplace.new_cart(), i)
            self.assertEqual(self.marketplace.register_producer(), str(i))
        for i in range(10):
            self.assertEqual(self.marketplace.publish(str(i), str(i + 1000)), True)
        for i in range(10):
            self.assertEqual(self.marketplace.add_to_cart(i, str(i + 1000)), True)
        for i in range(10):
            self.assertEqual(self.marketplace.add_to_cart(i, str(i + 1000)), False)

    def test_remove_from_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.register_producer(), "0")
        self.assertEqual(self.marketplace.publish("0", "00"), True)
        self.assertEqual(self.marketplace.publish("0", "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), False)
        self.marketplace.remove_from_cart(0, "00")
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)

    def test_place_order(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.register_producer(), "0")
        self.assertEqual(self.marketplace.publish("0", "00"), True)
        self.assertEqual(self.marketplace.publish("0", "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.assertEqual(self.marketplace.remove_from_cart(0, "01") , None)
        self.assertEqual(self.marketplace.remove_from_cart(0, "02"), None)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), False)
        self.marketplace.remove_from_cart(0, "00")
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.place_order(0), ["01", "00"])


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Functional Utility: Represent a production agent that continuously supplies the marketplace.
    Logic: Iterates through its product list, publishing items and observing 
    simulated production times. It uses 'safe_publish' to handle congestion.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers the producer and initializes production parameters.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def safe_publish(self, product, producer_id):
        """
        Block Logic: Congestion backoff.
        Logic: Retries publishing a product until the marketplace has 
        available capacity.
        """
        while not self.marketplace.publish(producer_id, product):
            sleep(self.republish_wait_time)

    def run(self):
        """
        Execution Logic: Infinite production loop.
        """
        my_id = self.marketplace.register_producer()

        while True:
            for (id_prod, quantity_prod, wait_time_prod) in self.products:
                sleep(wait_time_prod)

                for _ in range(quantity_prod):
                    self.safe_publish(id_prod, my_id)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Functional Utility: Immutable base data carrier for products.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Functional Utility: Specialized carrier for tea products.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Functional Utility: Specialized carrier for coffee products.
    """
    acidity: str
    roast_level: str
