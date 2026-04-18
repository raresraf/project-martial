
"""
@4f5d3295-d371-4d7f-8280-aebc3b1bffef/consumer.py
@brief Thread-safe multi-agent marketplace architecture for product exchange.
This file implements a concurrent system where producers generate products 
and consumers acquire them through shared shopping carts. The system ensures 
state consistency via mutex locks and provides transactional integrity for 
publish/acquire operations.

Domain: Concurrent Programming, Synchronization, System Simulation.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Functional Utility: Represent a consumer thread that executes a shopping schedule.
    Logic: For each assigned cart, it performs 'add' or 'remove' operations. 
    Acquisitions include a retry mechanism with backoff to handle stock 
    fluctuations in the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Binds the consumer thread to its cart list and the shared broker.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_to_cart(self, quantity, cart_id, product):
        """
        Block Logic: Product acquisition loop with retry backoff.
        Logic: Attempts to add 'quantity' units of a product. If the marketplace 
        rejects the request (stock empty), it sleeps before re-evaluating state.
        """
        i = 0
        while i < quantity:
            added_ok = self.marketplace.add_to_cart(cart_id, product)
            if added_ok:
                i = i + 1
            else:
                time.sleep(self.retry_wait_time)

    def remove_from_cart(self, quantity, cart_id, product):
        """
        Block Logic: Product return.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """
        Execution Logic: Orchestrates the sequential processing of shopping carts.
        Invariant: All operations within a single cart are completed before the 
        final order is placed and result printed.
        """
        cart_id = self.marketplace.new_cart()
        for cart_list in self.carts:
            for cart_event in cart_list:
                if cart_event["type"] == "add":
                    self.add_to_cart(cart_event["quantity"], cart_id, cart_event["product"])
                else:
                    self.remove_from_cart(cart_event["quantity"], cart_id, cart_event["product"])
        
        # Finalizes order and outputs the sequence of purchased products.
        for product in self.marketplace.place_order(cart_id):
            print(self.name, "bought", product)


from threading import Lock
import unittest
import time
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

class Marketplace:
    """
    Functional Utility: Synchronized coordinator for producer-consumer interactions.
    Logic: Tracks registered producers, available product pools, and consumer 
    carts. It maintains thread safety using granular locks for carts and producers, 
    and uses a mapping (product_in_cart) to track item ownership status.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes internal storage and audit logging.
        """
        self.queue_size_per_peroducer = queue_size_per_producer
        self.products = []
        self.carts = []
        self.product_in_cart = {}
        self.lock_cart = Lock()
        self.lock_producer = Lock()
        
        # Audit logging configuration.
        self.logger = logging.getLogger('marketplace')
        handler = RotatingFileHandler('marketplace.log', maxBytes=4096, backupCount=10)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        logging.Formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel("INFO")

    def register_producer(self):
        """
        Functional Utility: Atomically registers a new producer and returns its index.
        """
        self.logger.info("Method register_producer started")
        self.lock_producer.acquire()
        self.products.append([])
        ret = len(self.products) - 1
        self.lock_producer.release()
        self.logger.info("Method register_producer returned " + str(ret))
        return ret

    def publish(self, producer_id, product):
        """
        Functional Utility: Adds a product to a producer's inventory.
        Logic: Enforces per-producer capacity limits. Returns True on success.
        """
        self.logger.info("Method publish started")
        self.logger.info("producer_id = " + str(producer_id))
        self.logger.info("product = " + str(product))
        self.lock_producer.acquire()
        if len(self.products[producer_id]) < self.queue_size_per_peroducer:
            self.products[producer_id].append(product)
            self.product_in_cart[product] = False
            self.lock_producer.release()
            self.logger.info("New product published to marketplace")
            return True

        self.lock_producer.release()
        self.logger.info("Method publish returned False")
        return False

    def new_cart(self):
        """
        Functional Utility: Allocates a new shopping cart for a consumer.
        """
        self.logger.info("Method new_cart started")
        self.lock_cart.acquire()
        self.carts.append([])
        ret = len(self.carts) - 1
        self.lock_cart.release()
        self.logger.info("Method new_cart returned " + str(ret))
        return ret

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Transfers product from marketplace listing to a cart.
        Logic: Checks availability in global mapping and marks the product 
        as 'in cart' to prevent duplicate acquisitions.
        """
        self.logger.info("Method add_to_cart started")
        self.logger.info("cart_id = " + str(cart_id))
        self.logger.info("product = " + str(product))
        if product in self.product_in_cart.keys() and not self.product_in_cart[product]:
            self.carts[cart_id].append(product)
            self.product_in_cart[product] = True
            self.logger.info("New product added to cart " + str(cart_id))
            return True

        self.logger.info("Method add_to_cart returned False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Returns an item from a cart back to the marketplace pool.
        """
        self.logger.info("Method remove_from_cart started")
        self.logger.info("cart_id = " + str(cart_id))
        self.logger.info("product = " + str(product))
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            self.product_in_cart[product] = False
            self.logger.info("Product removed from cart")

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes order and clears producer inventories.
        Logic: Identifies the producers for all items in the cart and removes 
        the specific instances from their stock before returning the order list.
        """
        self.logger.info("Method place_order started")
        self.logger.info("cart_id = " + str(cart_id))
        for cart_product in self.carts[cart_id]:
            for prod_products in self.products:
                if cart_product in prod_products:
                    prod_products.remove(cart_product)
        self.logger.info("Method place_order returned " + str(self.carts[cart_id]))
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    """
    Functional Utility: Integrity validation suite for Marketplace transaction logic.
    """
    
    def setUp(self):
        self.marketplace = Marketplace(15)
        self.products = [Coffee("Espresso", 7, 4.00, "MEDIUM"), \
                        Coffee("Irish", 10, 5.00, "MEDIUM"), \
                        Tea("Black", 10, "Green")]

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(0, self.products[0]))
        self.assertTrue(self.marketplace.publish(0, self.products[1]))

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[0]))
        self.assertEqual(len(self.marketplace.carts[0]), 1)
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[1]))
        self.assertEqual(len(self.marketplace.carts[0]), 2)

    def test_remove_from_cart(self):
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[1])
        self.marketplace.remove_from_cart(0, self.products[2])
        self.assertEqual(len(self.marketplace.carts[0]), 2)
        self.marketplace.remove_from_cart(0, self.products[0])
        self.assertEqual(len(self.marketplace.carts[0]), 1)

    def test_place_order(self):
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[1])
        self.assertEqual(self.marketplace.place_order(0), [self.products[0], self.products[1]])


from threading import Thread
import time

class Producer(Thread):
    """
    Functional Utility: Represent a production agent that continuously supplies the marketplace.
    Logic: Iteratively publishes items from its inventory. It incorporates 
    simulated manufacturing time and congestion handling.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers with the marketplace and initializes production parameters.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Execution Logic: Infinite production loop.
        """
        while True:
            producer_id = self.marketplace.register_producer()
            for product in self.products:
                i = 0
                num_of_products = product[1]
                curr_product = product[0]
                curr_product_wait_time = product[2]
                while i < num_of_products:
                    published_ok = self.marketplace.publish(producer_id, curr_product)
                    if published_ok:
                        i += 1
                        # Inline: Simulates item manufacturing time.
                        time.sleep(curr_product_wait_time)
                    else:
                        # Block Logic: Congestion backoff.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Functional Utility: Immutable base data carrier for marketplace products.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Functional Utility: Specialized product type for tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Functional Utility: Specialized product type for coffee.
    """
    acidity: str
    roast_level: str
