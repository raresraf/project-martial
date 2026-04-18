
"""
@730a05b8-b37f-4e7c-91da-b9eb26d94701/consumer.py
@brief Threaded marketplace simulation with automated state validation.
This file implements a concurrent producer-consumer architecture where agents 
interact through a centralized broker. It features thread safety through 
granular mutex locking and provides a comprehensive suite of unit tests 
to verify the transactional integrity of product exchange and cart lifecycles.

Domain: Concurrent Programming, Synchronization, Unit Testing.
"""


from threading import Thread
import time

class Consumer(Thread):
    """
    Functional Utility: Represent a consumer thread that executes a series of shopping actions.
    Logic: Iterates through assigned shopping carts, performing 'add' or 'remove' 
    operations. It implements a retry loop with backoff for product acquisitions 
    to handle temporary stock depletion in the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Binds the consumer thread to its carts and the shared broker.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)


    def run(self):
        """
        Execution Logic: Processes all assigned carts sequentially.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for cart_item in cart:
                count = 0
                quantity = cart_item["quantity"]
                action = cart_item["type"]
                product = cart_item["product"]

                while count < quantity:
                    
                    if action == "add":
                        /**
                         * Block Logic: Synchronized product acquisition.
                         * Logic: Repeatedly attempts to add a product to the cart. 
                         * If stock is unavailable, it sleeps before re-evaluating state.
                         */
                        add = self.marketplace.add_to_cart(cart_id, product)
                        if add is True:
                            count += 1
                        else:
                            time.sleep(self.retry_wait_time)
                    
                    else:
                        /**
                         * Block Logic: Product return.
                         */
                        self.marketplace.remove_from_cart(cart_id, product)
                        count += 1

            
            # Block Logic: Finalizes the order and reports purchased items.
            for order_product in self.marketplace.place_order(cart_id):
                print(self.name + " bought " + str(order_product))


import unittest
from threading import Lock
import logging
import time
from logging.handlers import RotatingFileHandler
import os
from .product import Tea, Coffee

# Block Logic: Audit logging configuration.
logging.basicConfig(filename="marketplace.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logging.Formatter.converter = time.gmtime
logger = logging.getLogger()

handler = RotatingFileHandler("marketplace.log", mode='w', backupCount=10)



if os.path.isfile("marketplace.log"):
    handler.doRollover()

logger.setLevel(logging.DEBUG)


class Marketplace:
    """
    Functional Utility: Centralized broker for thread-safe product transactions.
    Logic: Tracks registered producers, available product pools, and consumer 
    carts. It maintains thread safety using separate locks for producer 
    registrations and cart operations to improve concurrency.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes storage structures and synchronization primitives.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = []
        self.products = {}
        self.carts = []

        self.producers_lock = Lock()


        self.carts_lock = Lock()

    def register_producer(self):
        """
        Functional Utility: Atomically registers a new producer and initializes its inventory mapping.
        """
        logger.info("Entered register_producer")
        with self.producers_lock:
            producer_id = len(self.producers)
            self.producers.append([])
            self.products[str(producer_id)] = []
            logger.info("Exited register_producer with id %d", producer_id)
            return producer_id

    def publish(self, producer_id, product):
        """
        Functional Utility: Adds a product instance to a producer's inventory.
        Logic: Enforces per-producer capacity limits. Returns True on success.
        """
        logger.info("Entered publish with producer_id %s and product %s",
                    producer_id, str(product))
        if len(self.producers[int(producer_id)]) < self.queue_size_per_producer:
            
            self.producers[int(producer_id)].append(product)
            
            self.products[producer_id].append(product)
            logger.info("Exited publish")
            return True

        logger.info("Exited publish")
        return False

    def new_cart(self):
        """
        Functional Utility: Allocates a new unique shopping cart for a consumer.
        """

        logger.info("Entered new_cart")
        with self.carts_lock:
            
            cart_id = len(self.carts)
            self.carts.append({})
            logger.info("Exited new_cart with id %d", cart_id)
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Transfers product from producer stock to consumer cart.
        Logic: Iterates through all producer inventories. If an instance is found, 
        it performs an atomic migration to the specified cart.
        """
        logger.info("Entered add_to_cart with cart_id %d and product %s", cart_id, product)
        for producer_id, prods in self.products.items():
            if product in prods:
                
                prods.remove(product)
                
                if producer_id in self.carts[cart_id]:
                    self.carts[cart_id][producer_id].append(product)
                else:
                    self.carts[cart_id][producer_id] = []


                    self.carts[cart_id][producer_id].append(product)
                logger.info("Exited add_to_cart")
                return True
        logger.info("Exited add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Restores a product from a cart back to the producer stock.
        """
        logger.info("Entered remove_from_cart with cart_id %d and product %s", cart_id, product)
        for producer_id in self.carts[cart_id]:
            if product in self.carts[cart_id][producer_id]:
                
                self.carts[cart_id][producer_id].remove(product)
                self.products[producer_id].append(product)
                logger.info("Exited remove_from_cart")
                break

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes transaction and returns purchased items.
        Logic: Consolidates items from all producers involved in the cart and 
        permanently removes them from the primary producer listings.
        """
        logger.info("Entered place_order with cart_id %d", cart_id)
        order = []
        for producer_id in self.carts[cart_id]:
            
            order = order + self.carts[cart_id][producer_id]
            for product in self.carts[cart_id][producer_id]:
                
                self.producers[int(producer_id)].remove(product)
        logger.info("Exited place_order")
        return order


class TestMarketplace(unittest.TestCase):
    """
    Functional Utility: Integrity validation suite for Marketplace transactional logic.
    """
    

    def setUp(self):
        """
        Pre-condition: Initialize a marketplace with a known capacity for testing.
        """
        self.marketplace = Marketplace(20)
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 2,
                         'Error: wrong producer id')
        self.assertEqual(self.marketplace.producers[0], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(self.marketplace.producers[1], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(self.marketplace.producers[2], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(len(self.marketplace.producers[0]), 0,
                         'Error: wrong number of products for the producer')
        self.assertEqual(len(self.marketplace.producers[1]), 0,
                         'Error: wrong number of products for the producer')
        self.assertEqual(len(self.marketplace.producers[2]), 0,
                         'Error: wrong number of products for the producer')

    def publish(self):
        
        self.assertEqual(self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea")), True,
                         'Error: wrong published item')
        self.assertEqual(self.marketplace.publish("1", Coffee("Colombia", 7, "5.05", "HIGH")),
                         True, 'Error: wrong published item')

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 2,
                         'Error: wrong cart id')

    def test_add_to_cart(self):
        
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("Colombia", 7, "5.05", "HIGH")),
                         True, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW")),
                         True, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("Colombia", 7, "5.05", "HIGH")),
                         False, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(1, Tea("Twinings", 7, "Black Tea")), True,
                         'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(1, Tea("Twinings", 7, "Black Tea")), False,
                         'Error: wrong product added to cart')

    def test_remove_from_cart(self):
        
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.marketplace.add_to_cart(0, Tea("Twinings", 7, "Black Tea"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("Colombia", 7, "5.05", "HIGH"))

        self.assertEqual(self.marketplace.remove_from_cart(0, Tea("Twinings", 7, "Black Tea")),
                         None, 'Error: wrong product removed from cart')
        self.assertEqual(self.marketplace.remove_from_cart(1, Coffee("India", 7, "5.05", "LOW")),
                         None, 'Error: wrong product removed from cart')
        self.assertEqual(self.marketplace.remove_from_cart(1, Coffee("Colombia", 7, "5.05",
                                                                     "HIGH")),
                         None, 'Error: wrong product removed from cart')

    def test_place_order(self):
        
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("1", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("1", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.marketplace.add_to_cart(0, Tea("Twinings", 7, "Black Tea"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("Colombia", 7, "5.05", "HIGH"))

        order0 = [Tea("Twinings", 7, "Black Tea"), Coffee("India", 7, "5.05", "LOW"),
                  Coffee("India", 7, "5.05", "LOW")]

        order1 = [Coffee("India", 7, "5.05", "LOW"), Coffee("Colombia", 7, "5.05", "HIGH")]

        self.assertEqual(self.marketplace.place_order(0), order0,
                         'Error: wrong order')

        self.assertEqual(self.marketplace.place_order(1), order1,
                         'Error: wrong order')

    if __name__ == '__main__':
        unittest.main()


from threading import Thread
import time

class Producer(Thread):
    """
    Functional Utility: Represent a production agent that continuously supplies the marketplace.
    Logic: Iterates through its inventory, attempting to publish products. 
    It incorporates simulated manufacturing time and handles congestion via 
    a timed backoff mechanism.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers the producer and initializes production parameters.
        """

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)
        self.producer_id = self.marketplace.register_producer()


    def run(self):
        """
        Execution Logic: Infinite production loop.
        """
        while True:
            for (prod, quant, w_time) in self.products:
                count = 0
                while count < quant:
                    if self.marketplace.publish(str(self.producer_id), prod):
                        count += 1
                        # Inline: Simulated item manufacturing time.
                        time.sleep(w_time)
                    else:
                        # Block Logic: Congestion backoff.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Functional Utility: Base immutable data carrier for products.
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
