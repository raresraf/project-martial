"""
@c4bd3ded-3e65-4dac-8236-3f21291458a5/consumer.py
@brief multi-threaded simulation of a retail ecosystem using concurrent Producer and Consumer threads.
Architecture: Centralized Marketplace mediator manages shared state, coordinating autonomous agents for asynchronous inventory processing.
Functional Utility: Handles inventory flow, session-persistent shopping carts, and concurrent transactional updates.
Synchronization: Employs threading.Lock for critical sections and cooperative yield patterns (sleep) for demand-supply flow control.
"""

from threading import Thread
import time
import sys

class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping sessions.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested products from the Marketplace.
    Error Handling: Implements a polling retry loop with yields (sleep) for handling temporary stock depletions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts List of product acquisition lists to be fulfilled.
        @param marketplace Shared resource mediator.
        @param retry_wait_time Duration to wait when the marketplace is depleted.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name")

    def run(self):
        """
        @brief lifecycle manager for the consumer thread.
        Logic: Allocates a new transactional session (id_cart) and fulfills all commands before finalizing the order.
        """
        for cart in self.carts:
            # Initialization: Establishes a unique inventory buffer in the marketplace.
            id_cart = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]
                my_type = operation["type"]
                product = operation["product"]
                contor = 0
                
                # Block Logic: Fulfillment loop.
                while contor < quantity:
                    
                    if my_type == "add":
                        # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                        if self.marketplace.add_to_cart(id_cart, product):
                            contor = contor + 1
                        else:
                            # Logic: Yields execution to handle temporary stock depletion.
                            time.sleep(self.retry_wait_time)
                    
                    else:
                        # Logic: Returns reserved commodities to the marketplace inventory.
                        self.marketplace.remove_from_cart(id_cart, product)
                        contor = contor + 1
            
            # Finalization: executes the transaction and flushes results to standard output.
            placed_order = self.marketplace.place_order(id_cart)
            for each_p in placed_order:
                
                sys.stdout.flush() # Synchronization: ensures immediate visibility of output.
                print(f"{self.name} bought {each_p}")


import unittest
from threading import Lock
from time import gmtime



import logging
from logging.handlers import RotatingFileHandler


# Block Logic: Global audit logging configuration.
# Functional Utility: Persistent log with rotation to manage disk consumption during concurrent execution.
logging.basicConfig(handlers=[
    RotatingFileHandler('marketplace.log', maxBytes=100000, backupCount=10)
],
                    level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = gmtime
logger = logging.getLogger()


class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer stock queues and session-based consumer carts.
    Synchronization: Employs fine-grained locks (lock_register, lock_maximum_elements, etc.) to minimize global contention.
    """

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum inventory allowed per supplier for backpressure control.
        """

        logger.info(
            'The marketplace %s with maximum queue of %s is initializing...',
            self, queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer
        self.product_to_producer = {} 
        self.producer_to_products = {} 
        self.carts = {} 
        self.cart_counter = 0
        self.producer_counter = 0

        self.lock_register = Lock()
        self.lock_maximum_elements = Lock()
        self.lock_cart_size = Lock()
        self.lock_remove_from = Lock()

        logger.info('Initialization ended successfully!')

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its thread-safe inventory list.
        @return Unique producer identifier.
        """

        logger.info('Starting producer registration by %s...', self)
        with self.lock_register:
            
            # Initialization: Scaffolds the inventory pool for the new producer.
            self.producer_to_products[self.producer_counter] = []
            self.producer_counter += 1
            logger.info('Producer with id: %d was created!',
                        self.producer_counter - 1)
            return self.producer_counter - 1

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the marketplace.
        Constraint: Operation rejected if the supplier's individual queue is saturated (backpressure).
        """

        logger.info("Providing product %s by producer %s to marketplace %s...",
                    product, producer_id, self)
        with self.lock_maximum_elements:
            
            # Block Logic: Threshold check for supply-side flow control.
            if len(self.producer_to_products[producer_id]) \
                >= self.queue_size_per_producer:
                logger.info('Providing product failed!')
                return False
            
            # Invariant: Updates both local pool and global tracking metadata.
            self.producer_to_products[producer_id].append(product)
            self.product_to_producer[product] = producer_id
        logger.info('Providing product ended successfully!')
        return True

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """

        with self.lock_cart_size:
            
            logger.info('Creating new cart by marketplace %s...', self)
            self.carts[self.cart_counter] = []
            self.cart_counter += 1
            logger.info('A new cart with id %d was created!',
                        self.cart_counter - 1)
        return self.cart_counter - 1

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        Strategy: Exhaustive search across all supplier pools. First-available fulfillment strategy.
        @return Boolean indicating acquisition success.
        """

        logger.info('Adding product %s in the cart %s using marketplace %s...',
                    product, cart_id, self)
        
        all_producers = self.producer_to_products.keys()
        for producer in all_producers:
            # Block Logic: Acquisition and reservation.
            number_of_products = \
                self.producer_to_products[producer].count(product)
            
            if number_of_products > 0:
                
                # Invariant: Item must be removed from global pool before being assigned to a session.
                self.carts[cart_id].append(product)
                self.producer_to_products[producer].remove(product)
                logger.info('Adding a new product ended successfully!')
                return True
        logger.info('Adding a new product failed!')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to its originating producer's pool.
        """

        logger.info(
            'Removing product %s from the cart %s using marketplace %s...',
            product, cart_id, self)
        
        # Logic: Identifies original producer from cached metadata.
        producer = self.product_to_producer[product]
        with self.lock_remove_from:
            self.carts[cart_id].remove(product)
            # State Sync: Restores the physical unit to the supplier pool.
            self.producer_to_products[producer].append(product)
        logger.info('Removing a new product ended successfully!')

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and flushes result data.
        """

        logger.info('Placing a new order from cart %s using marketplace %s',
                    cart_id, self)
        
        # Finalization: Resets the session state.
        final_order = self.carts.pop(cart_id, None)
        logger.info('The order %s was provided!', final_order)
        return final_order


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and concurrency safety.
    """

    def setUp(self):
        
        self.size_marketplace = 2

        self.marketplace = Marketplace(self.size_marketplace)
        self.consumer0 = Consumer(carts=[],
                                  marketplace=self.marketplace,
                                  retry_wait_time=100,
                                  kwargs=dict({"name": "consumer0"}))
        self.consumer1 = Consumer(carts=[],
                                  marketplace=self.marketplace,
                                  retry_wait_time=250,
                                  kwargs=dict({"name": "consumer1"}))
        self.product0 = Coffee('Arabica', 12, 6, 'MEDIUM')
        self.product1 = Coffee('Cappucino', 10, 12, 'LOW')
        self.product2 = Tea('Complex', 9, 'White')
        self.product3 = Tea('Honey tea', 11, 'Sweet')
        self.producer0 = Producer(products=[],
                                  marketplace=self.marketplace,
                                  republish_wait_time=120,
                                  kwargs={})
        self.producer1 = Producer(products=[],
                                  marketplace=self.marketplace,
                                  republish_wait_time=176,
                                  kwargs={})

    def test___init__(self):
        
        self.assertEqual(self.marketplace.queue_size_per_producer,
                         self.size_marketplace)
        self.assertEqual(self.marketplace.cart_counter, 0)
        self.assertEqual(self.marketplace.producer_counter, 2)
        self.assertEqual(self.producer0.id_producer, 0)
        self.assertEqual(self.producer1.id_producer, 1)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.producer_counter, 3)

    def test_publish(self):
        


        self.assertEqual(self.marketplace.publish(0, self.product1), True)
        self.assertEqual(self.marketplace.publish(0, self.product2), True)
        self.assertEqual(self.marketplace.publish(0, self.product3), False)
        self.assertEqual(self.marketplace.publish(1, self.product0), True)
        self.assertEqual(self.marketplace.publish(1, self.product3), True)

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.publish(0, self.product1)
        self.marketplace.publish(0, self.product2)
        self.marketplace.publish(1, self.product0)
        self.marketplace.publish(1, self.product3)

        self.assertTrue(self.marketplace.add_to_cart(0, self.product0))
        self.assertFalse(self.marketplace.add_to_cart(1, self.product0))
        self.assertTrue(self.marketplace.add_to_cart(0, self.product1))
        self.assertTrue(self.marketplace.add_to_cart(0, self.product3))
        self.assertTrue(self.marketplace.add_to_cart(1, self.product2))

    def test_remove_from_cart(self):
        
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.publish(0, self.product1)
        self.marketplace.publish(0, self.product2)
        self.marketplace.publish(1, self.product0)
        self.marketplace.publish(1, self.product3)
        self.marketplace.add_to_cart(0, self.product0)
        self.marketplace.add_to_cart(0, self.product1)
        self.marketplace.add_to_cart(0, self.product3)
        self.marketplace.add_to_cart(1, self.product2)

        self.marketplace.remove_from_cart(1, self.product2)
        self.assertFalse(self.product2 in self.marketplace.carts[1])

        self.marketplace.remove_from_cart(0, self.product3)
        self.assertFalse(self.product3 in self.marketplace.carts[0])

    def test_place_order(self):
        
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.publish(0, self.product1)
        self.marketplace.publish(0, self.product2)
        self.marketplace.publish(1, self.product0)
        self.marketplace.publish(1, self.product3)
        self.marketplace.add_to_cart(0, self.product0)
        self.marketplace.add_to_cart(0, self.product1)
        self.marketplace.add_to_cart(0, self.product3)
        self.marketplace.add_to_cart(1, self.product2)
        self.marketplace.remove_from_cart(1, self.product2)
        self.marketplace.remove_from_cart(0, self.product3)

        self.assertEqual(self.marketplace.place_order(0),
                         [self.product0, self.product1])
        self.assertEqual(len(self.marketplace.place_order(1)), 0)
        self.assertEqual(len(self.marketplace.place_order(2)), 0)


from threading import Thread
import time


class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continually fulfills production quotas and publishes results to the Marketplace mediator.
    Functional Utility: Models manufacturing latencies and handles supply-side flow control.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) production metrics.
        @param marketplace Shared resource management interface.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Initialization: Registers as a supplier to obtain a persistent ID.
        self.id_producer = marketplace.register_producer()

    def run(self):
        """
        @brief Main manufacturing loop for the producer execution context.
        """
        products = self.products
        while True:
            for product in products:
                contor = 0
                
                # Block Logic: Quota fulfillment with backpressure handling.
                while contor < product[1]:
                    
                    if self.marketplace.publish(self.id_producer, product[0]):
                        # Logic: Simulated industrial processing duration.
                        time.sleep(product[2])
                        contor = contor + 1
                    
                    else:
                        # Synchronization: Exponential wait during marketplace inventory overflow.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base representation of a marketable commodity unit.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized beverage commodity.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized beverage commodity with profile attributes.
    """
    acidity: str
    roast_level: str
