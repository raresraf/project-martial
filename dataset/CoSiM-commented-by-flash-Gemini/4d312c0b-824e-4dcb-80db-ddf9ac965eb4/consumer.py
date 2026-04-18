
"""
@4d312c0b-824e-4dcb-80db-ddf9ac965eb4/consumer.py
@brief Thread-safe Marketplace Simulation with automated testing.
This file implements a concurrent marketplace architecture where multiple 
producers and consumers interact through shared shopping carts and product 
queues. It features rigorous synchronization using locks and provides a 
comprehensive suite of unit tests for state validation.

Domain: Concurrent Systems, Synchronization, Unit Testing.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Functional Utility: Represent a consumer agent that processes assigned shopping carts.
    Logic: Iterates through its carts, executing 'add' or 'remove' operations. 
    If a product is unavailable, it performs a timed retry. Orders are placed 
    atomically per cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Initializes the consumer with cart data and marketplace connection.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Execution Logic: Orchestrates the multi-cart processing workflow.
        Invariant: All products for all carts are collected before final order 
        printing occurs.
        """
        products_list = []
        cart_ids = []

        # Block Logic: Cart initialization and population loop.
        for list_cart in self.carts:
            cart_id = self.marketplace.new_cart()
            cart_ids.append(cart_id)

            for dict_command in list_cart:
                command = dict_command['type']
                prod = dict_command['product']
                quantity = dict_command['quantity']

                counter_add = 0
                counter_remove = 0

                if command == 'add':
                    /**
                     * Block Logic: Product acquisition with backoff.
                     * Logic: Repeatedly attempts to add the product to the cart. 
                     * If unsuccessful (stock empty), it sleeps before retrying.
                     */
                    while counter_add < quantity:
                        if not self.marketplace.add_to_cart(cart_id, prod):
                            time.sleep(self.retry_wait_time)
                        else:
                            counter_add += 1
                else:
                    /**
                     * Block Logic: Product removal.
                     */
                    while counter_remove < quantity:
                        self.marketplace.remove_from_cart(cart_id, prod)
                        counter_remove += 1

        # Block Logic: Finalizes all orders and consolidates results.
        for cart in cart_ids:
            products_list.extend(self.marketplace.place_order(cart))

        /**
         * Block Logic: Synchronized result reporting.
         * Logic: Uses a specialized print lock to prevent interleaved console 
         * output from multiple consumer threads.
         */
        self.marketplace.print_lock.acquire()
        for product in products_list:
            print(f'{self.name} bought', product)
        self.marketplace.print_lock.release()

from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea


# Functional Utility: Configures a rotating log for transaction monitoring.
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('marketplace.log', maxBytes=3000, backupCount=5)
logger.addHandler(handler)

class Marketplace:
    """
    Functional Utility: Central coordinator for product exchange.
    Logic: Manages producer registrations, stock tracking, and consumer cart lifecycles. 
    It maintains thread safety through granular locking (lock_prod, lock_cons) 
    and tracks product ownership migration between producers and consumers.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes internal storage structures and synchronization locks.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_list = []
        self.consumer_carts = []
        self.lock_prod = Lock()
        self.lock_cons = Lock()
        self.producer_id = 0
        self.cart_id = 0
        self.taken_products = []
        self.print_lock = Lock()

    def register_producer(self):
        """
        Functional Utility: Registers a new producer and initializes its inventory slot.
        """
        self.lock_prod.acquire()
        logger.info("Producer wants to register")
        self.producer_id += 1
        self.producer_list.insert(self.producer_id - 1, [])
        logger.info("Producer %d got registered", self.producer_id)
        self.lock_prod.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Functional Utility: Adds a product to a producer's inventory.
        Logic: Enforces per-producer capacity limits. Returns True on success.
        """
        logger.info("Producer %d wants to publish %s", producer_id, product)
        if len(self.producer_list[producer_id - 1]) == self.queue_size_per_producer:
            return False
        self.producer_list[producer_id - 1].append(product)
        logger.info("Producer %d published %s", producer_id, product)
        return True

    def new_cart(self):
        """
        Functional Utility: Creates a unique shopping cart for a consumer.
        """
        self.lock_cons.acquire()
        logger.info("Consumer requested a cart")
        self.cart_id += 1
        self.consumer_carts.insert(self.cart_id - 1, [])
        logger.info("Consumer got cart %d", self.cart_id)
        self.lock_cons.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Atomically migrates a product from any producer to a specific cart.
        Logic: Iterates through all producer inventories. If a match is found, 
        it removes the item from the source and appends it to the destination cart, 
        tracking the migration for potential returns.
        """
        self.lock_cons.acquire()
        logger.info("Cart %d wants to add %s", cart_id, product)
        copy_lst = list(self.producer_list)

        # Block Logic: Search and migration across producers.
        for lst in copy_lst:
            if product in lst:
                indx = copy_lst.index(lst)
                self.consumer_carts[cart_id - 1].append(product)
                self.taken_products.append((indx + 1, product))
                self.producer_list[indx].remove(product)
                self.lock_cons.release()
                logger.info("Cart %d added %s", cart_id, product)
                return True

        self.lock_cons.release()
        logger.info("Cart %d added %s", cart_id, product)
        return False


    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Restores a product from a cart back to its original producer inventory.
        """
        logger.info("Cart %d wants to remove %s", cart_id, product)
        self.consumer_carts[cart_id - 1].remove(product)
        for prod in self.taken_products:
            if prod[1] == product:
                self.producer_list[prod[0] - 1].append(product)
                break
        logger.info("Cart %d removed %s", cart_id, product)

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes the order and returns the purchased products.
        """
        logger.info("Cart %d placed order", cart_id)
        return self.consumer_carts[cart_id - 1]

class TestMarketplace(unittest.TestCase):
    """
    Functional Utility: Unit test suite for verifying Marketplace contract and state transitions.
    """
    
    def setUp(self):
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        self.assertNotEqual(self.marketplace.register_producer(), 0, "Wrong id for producer!")

    def test_publish(self):
        actual_product = self.marketplace.publish(self.marketplace.register_producer(), \
                                                    Tea("Linden", 13, "Floral"))
        self.assertTrue(actual_product, "The product should be published!")

    def test_new_cart(self):
        self.assertNotEqual(self.marketplace.new_cart(), 0, "Wrong id for cart!")

    def test_add_to_cart(self):
        actual_cart_id = self.marketplace.new_cart()
        wanted_product = self.marketplace.add_to_cart(actual_cart_id,\
                                                    Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.assertFalse(wanted_product, "This product should not be added to cart now!")

    def test_remove_from_cart(self):
        actual_cart_id = self.marketplace.new_cart()
        self.marketplace.publish(self.marketplace.register_producer(),\
                                    Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.add_to_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.remove_from_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.assertListEqual(self.marketplace.consumer_carts[actual_cart_id - 1], [],\
                                        "Product was not removed!")

    def test_place_order(self):
        prod_id = self.marketplace.register_producer()
        self.marketplace.publish(prod_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.publish(prod_id, Tea("Brewstar", 17, "Green"))
        actual_cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.add_to_cart(actual_cart_id, Tea("Brewstar", 17, "Green"))
        self.marketplace.remove_from_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        remainder_list = []
        remainder_list.append(Tea("Brewstar", 17, "Green"))
        self.assertCountEqual(self.marketplace.place_order(actual_cart_id),\
                                    remainder_list, "Wrong order!")


from threading import Thread
import time


class Producer(Thread):
    """
    Functional Utility: Represent a production agent that continuously supplies the marketplace.
    Logic: Iterates through its production schedule, attempting to publish products. 
    If the marketplace is full, it implements a spin-wait with delay.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers the producer and initializes its schedule.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Execution Logic: Infinite production loop.
        """
        producer_id = self.marketplace.register_producer()

        while True:
            for prod in self.products:
                i = 0
                while i < prod[1]:
                    if not self.marketplace.publish(producer_id, prod[0]):
                        # Block Logic: Congestion backoff.
                        time.sleep(self.time)
                    else:
                        i += 1
                        # Inline: Simulates product manufacturing time.
                        time.sleep(prod[2])
