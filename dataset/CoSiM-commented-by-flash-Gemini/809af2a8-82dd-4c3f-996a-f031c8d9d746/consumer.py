
"""
@809af2a8-82dd-4c3f-996a-f031c8d9d746/consumer.py
@brief Threaded marketplace simulation with prioritized synchronization and transaction logging.
This module implements a concurrent producer-consumer architecture where agents 
exchange products via a central marketplace coordinator. It features thread safety 
through a set of granular mutex locks (producer_lock, cart_lock, cart_add_lock, 
place_order_lock) and a rotating logging mechanism to track the lifecycle 
of every transaction and state mutation.

Domain: Concurrent Systems, Synchronization, System Auditing.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Functional Utility: Represent a consumer thread that executes assigned shopping schedules.
    Logic: For each shopping cart, it performs 'add' or 'remove' operations in specified 
    quantities. It implements a timed retry loop for acquisitions to handle 
    momentary stock depletion in the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Binds the consumer thread to its carts and the shared marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Execution Logic: Orchestrates the processing of multiple shopping carts.
        """
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            # Block Logic: Sequential execution of cart actions.
            for action in cart:
                curr_quantity = 0

                
                while curr_quantity < action["quantity"]:
                    if action["type"] == "add":
                        /**
                         * Block Logic: Synchronized acquisition with backoff.
                         * Logic: Attempts to add a product to the cart. If the marketplace 
                         * is empty, it sleeps before re-trying.
                         */
                        if self.marketplace.add_to_cart(cart_id, action["product"]):
                            curr_quantity += 1
                        else:
                            
                            sleep(self.retry_wait_time)
                    elif action["type"] == "remove":
                        /**
                         * Block Logic: Product return.
                         */
                        self.marketplace.remove_from_cart(cart_id, action["product"])
                        curr_quantity += 1

            
            self.marketplace.place_order(cart_id)

import time
import unittest
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock, currentThread


from tema.consumer import Consumer
from tema.producer import Producer
from tema.product import Coffee, Tea


class Marketplace:
    """
    Functional Utility: Thread-safe coordinator for product publication and acquisition.
    Logic: Manages producer registration, global stock pool, and consumer cart lifecycles. 
    It ensures transactional integrity using a variety of locks to prevent race 
    conditions during identifier assignment and resource migration.
    """

    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes storage structures, locks, and rotating logging.
        """
        self.queue_size_per_producer = queue_size_per_producer


        self.nr_producers = -1
        self.nr_carts = -1
        self.queues = []
        self.products = []
        self.carts = []

        self.producer_lock = Lock()
        self.cart_lock = Lock()
        self.cart_add_lock = Lock()
        self.place_order_lock = Lock()

        
        # Block Logic: Audit log initialization with rotation.
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.Formatter.converter = time.gmtime
        self.logger = logging.getLogger("marketplace_logger")
        handler = RotatingFileHandler("file.log", maxBytes=5000, backupCount=15)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def register_producer(self):
        """
        Functional Utility: Issues a unique ID to a new producer.
        """
        with self.producer_lock:
            self.nr_producers += 1
            self.queues.append([])
            self.logger.info(f'register_producer output: producer_id={self.nr_producers}')
        return self.nr_producers

    def publish(self, producer_id, product):
        """
        Functional Utility: Adds a product to a producer's inventory and global pool.
        Logic: Enforces per-producer capacity limits. Returns True on success.
        """
        self.logger.info(f'publish input: producer_id={producer_id}, product={product}')

        p_id = int(producer_id)

        
        if len(self.queues[p_id]) == self.queue_size_per_producer:
            self.logger.info("publish output: FALSE")
            return False

        
        self.products.append(product)
        self.queues[p_id].append(product)

        self.logger.info("publish output: TRUE")
        return True

    def new_cart(self):
        """
        Functional Utility: Allocates a new shopping cart for a consumer.
        """
        with self.cart_lock:
            self.nr_carts += 1
            self.carts.append([])
            self.logger.info(f'new_cart output: cart_id={self.nr_carts}')
        return self.nr_carts

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Transfers a product from the global pool to a specific cart.
        Logic: Checks availability in the global 'products' list and performs 
        an atomic transfer to the destination cart.
        """
        self.logger.info(f'add_to_cart input: cart_id={cart_id}, product={product}')

        with self.cart_add_lock:
            
            if product not in self.products:
                self.logger.info("add_to_cart output: FALSE")
                return False

            
            self.products.remove(product)
            self.carts[cart_id].append(product)
            self.logger.info("add_to_cart output: TRUE")

        return True

    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Restores a product from a cart back to the global pool.
        """
        self.logger.info(f'remove_from_cart input: cart_id={cart_id}, product={product}')
        self.products.append(product)
        self.carts[cart_id].remove(product)

    def remove_from_queue(self, product):
        """
        Functional Utility: Permanently removes a product from its producer's inventory.
        """
        self.logger.info(f'remove_from_queue input: product={product}')
        for producer_queue in self.queues:
            if product in producer_queue:
                producer_queue.remove(product)
                break

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes transaction and returns purchased items.
        """
        self.logger.info(f'place_order input: cart_id={cart_id}')

        with self.place_order_lock:
            for product in self.carts[cart_id]:
                self.remove_from_queue(product)
                print(currentThread().name, "bought", product)

        self.logger.info(f'place_order output: cart_list={self.carts[cart_id]}')
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    """
    Functional Utility: Integrity validation suite for Marketplace state transitions.
    """
    
    def setUp(self):
        self.marketplace = Marketplace(15)
        self.product1 = Coffee("Indonezia", 1, "5.05", "MEDIUM")
        self.product2 = Tea("Linden", 9, "Herbal")


        self.producer = Producer([[self.product1, 2, 0.18],
                                  [self.product2, 1, 0.23]],
                                 self.marketplace,
                                 0.15)
        self.consumer = Consumer([[{"type": "add", "product": self.product2, "quantity": 2},
                                   {"type": "add", "product": self.product1, "quantity": 2},
                                   {"type": "remove", "product": self.product1, "quantity": 1}
                                   ]],
                                 self.marketplace,
                                 0.31)

        self.cart_id = self.marketplace.new_cart()

    def test_register_producer(self):
        self.assertEqual(self.producer.producer_id, "0")

    def test_publish(self):
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.assertEqual(self.marketplace.products,
                         self.marketplace.queues[int(self.producer.producer_id)])

    def test_new_cart(self):
        self.assertEqual(self.cart_id, 0)

    def test_add_to_cart(self):
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product2)

        self.assertEqual(self.marketplace.carts[self.cart_id], [self.product1, self.product2])

    def test_remove_from_cart(self):
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product2)

        self.marketplace.remove_from_cart(self.cart_id, self.product1)
        self.assertEqual(self.marketplace.carts[self.cart_id], [self.product1, self.product2])

    def test_place_order(self):
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.consumer.run()

        self.assertEqual(self.marketplace.queues[int(self.producer.producer_id)],
                         [self.product1])


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Functional Utility: Represent a production agent that continuously generates supply.
    Logic: Iteratively publishes items from its schedule. It incorporates 
    simulated manufacturing delay and handles congestion via a timed backoff mechanism.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers the producer and initializes production parameters.
        """
        Thread.__init__(self, **kwargs)
        self.products = products


        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = str(self.marketplace.register_producer())

    def run(self):
        """
        Execution Logic: Infinite production loop.
        """
        while True:
            for (product, max_products, success_wait_time) in self.products:
                curr_products = 0

                
                while curr_products < max_products:
                    if self.marketplace.publish(self.producer_id, product):
                        curr_products += 1
                        # Inline: Simulated item manufacturing time.
                        sleep(success_wait_time)
                    else:
                        # Block Logic: Congestion backoff.
                        sleep(self.republish_wait_time)
