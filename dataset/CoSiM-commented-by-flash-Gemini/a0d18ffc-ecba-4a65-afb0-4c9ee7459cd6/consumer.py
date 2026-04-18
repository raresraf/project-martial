"""
@a0d18ffc-ecba-4a65-afb0-4c9ee7459cd6/consumer.py
@brief Distributed simulation of a retail marketplace using concurrent Producer and Consumer agents.
Architecture: Multi-threaded actor model where Producers generate stock and Consumers execute shopping workflows via a centralized Marketplace mediator.
Functional Utility: Manages inventory tracking, virtual shopping cart lifecycle, and thread-safe console reporting.
Synchronization: Employs fine-grained concurrency control using BoundedSemaphores and cooperative yield patterns (sleep) for demand-supply flow control.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping lists.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested products from the Marketplace.
    Error Handling: Implements a polling retry loop when inventory is unavailable.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Hierarchical list of shopping requests (type, product, quantity).
        @param marketplace Shared resource management interface.
        @param retry_wait_time Temporal duration to yield when inventory is depleted.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Lifecycle manager for the consumer execution context.
        Logic: Allocates a new transactional session (id_cart) and fulfills all commands before finalizing the order.
        """
        for cart in self.carts:
            # Initialization: Establishes a session-scoped inventory buffer.
            id_cart = self.marketplace.new_cart()
            for action in cart:
                # Block Logic: Dispatcher for marketplace operations (acquisition or return).
                if action['type'] == 'add':
                    quantity = 0
                    # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                    while quantity < action['quantity']:
                        if self.marketplace.add_to_cart(id_cart, action['product']):
                            quantity += 1
                        else:
                            # Logic: Yields execution to handle temporary stock depletion.
                            sleep(self.retry_wait_time)
                elif action['type'] == 'remove':
                    # Logic: Returns reserved commodities to the marketplace inventory.
                    for _ in range(action['quantity']):
                        self.marketplace.remove_from_cart(id_cart, action['product'])
            
            # Finalization: Commits the transaction and outputs results.
            self.marketplace.place_order(id_cart)


import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import BoundedSemaphore
from random import randint
import sys
import threading
import time
from uuid import UUID, uuid1
from unittest import TestCase

from tema.product import Coffee, Product, Tea


class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer stock queues and session-based consumer carts.
    Synchronization: Uses per-producer semaphores and global mutexes (carts_mutex, print_mutex) to ensure atomicity.
    Observability: Integrates RotatingFileHandler for persistent transactional auditing.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on inventory per supplier for backpressure management.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producer_queues = {} # Mapping: ProducerID -> [Semaphore, Occupancy, List of (Product, IsAvailable)].
        self.carts = {} # Mapping: CartID -> List of (SourceProducerID, Product).
        
        self.carts_mutex = BoundedSemaphore(1) # Protects global cart registry updates.
        self.print_mutex = BoundedSemaphore(1) # Serializes console output across threads.

        # Block Logic: Audit logging configuration.
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=10**6, backupCount=10)
        
        formatter = logging.Formatter('%(asctime)s UTC %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def register_producer(self) -> UUID:
        """
        @brief Onboards a new supplier and initializes its thread-safe inventory structures.
        @return Unique producer identifier (UUID v1).
        """
        logging.info('register_producer() was called')
        id_prod = uuid1()
        # Initialization: Scaffolds the state container for the new producer.
        self.producer_queues[id_prod] = [BoundedSemaphore(1), 0, []]
        logging.info('register_producer() returned (%s)', id_prod)


        return id_prod

    def publish(self, producer_id: UUID, product: Product) -> bool:
        """
        @brief Allows a producer to add commodities to their marketplace pool.
        Constraint: Operation rejected if the supplier's individual queue is saturated.
        """
        logging.info('publish(%s, %s) was called', producer_id, product)
        
        # Block Logic: Threshold check for supply-side flow control.
        if self.producer_queues[producer_id][1] < self.queue_size_per_producer:
            self.producer_queues[producer_id][2].append([product, True])
            # Synchronization: Atomically updates occupancy count.
            with self.producer_queues[producer_id][0]:
                self.producer_queues[producer_id][1] += 1
            logging.info('publish(%s, %s) returned True', producer_id, product)
            return True
        logging.info('publish(%s, %s) returned False', producer_id, product)


        return False

    def new_cart(self) -> int:
        """
        @brief Allocates a new transactional session for a consumer.
        Logic: Generates a unique random identifier within a synchronized context.
        """
        logging.info('new_cart() was called')
        with self.carts_mutex:
            id_cart = randint(0, sys.maxsize)
            # Invariant: Ensures uniqueness of the session identifier.
            while id_cart in list(self.carts.keys()):
                id_cart = randint(0, sys.maxsize)
            self.carts[id_cart] = []
            logging.info('new_cart() returned %d', id_cart)


            return id_cart

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """
        @brief Atomically transfers a commodity from any available producer to a specific cart.
        Strategy: Exhaustive search across all supplier pools. First-available fulfillment strategy.
        @return Boolean indicating acquisition success.
        """
        logging.info('add_to_cart(%d, %s) was called', cart_id, product)
        for id_prod in self.producer_queues:
            # Block Logic: Search pass under producer-specific lock.
            with self.producer_queues[id_prod][0]:
                for prod in self.producer_queues[id_prod][2]:
                    if prod[0] == product and prod[1]:
                        # Invariant: Product must be marked unavailable before being assigned to a session.
                        prod[1] = False
                        self.producer_queues[id_prod][1] -= 1
                        self.carts[cart_id].append((id_prod, product))
                        logging.info('publish(%s, %s) returned True', cart_id, product)
                        return True
        logging.info('add_to_cart(%s, %s) returned False', cart_id, product)


        return False

    def remove_from_cart(self, cart_id: int, product: Product):
        """
        @brief Reverts an acquisition, restoring the unit to its source producer's quota.
        """
        logging.info('remove_from_cart(%d, %s) was called', cart_id, product)
        for item in self.carts[cart_id]:
            if item[1] == product:
                # Logic: Uses the source producer ID cached in the cart entry to route the return.
                with self.producer_queues[item[0]][0]:
                    
                    for prod in self.producer_queues[item[0]][2]:
                        if prod[0] == product:
                            prod[1] = True
                            self.producer_queues[item[0]][1] += 1
                            self.carts[cart_id].remove((item[0], product))
                            logging.info('remove_from_cart(%d, %s) returned', cart_id, product)
                            return

    def place_order(self, cart_id: int):
        """
        @brief Finalizes the transaction and serializes results to stdout.
        Side Effect: Purges the units from the respective supplier queues.
        """
        logging.info('place_order(%d) was called', cart_id)
        result = []
        for item in self.carts[cart_id]:
            result.append(item[1])
            # Logic: Atomic removal from supplier's internal list.
            with self.producer_queues[item[0]][0]:
                for prod in self.producer_queues[item[0]][2]:
                    if prod[0] == item[1]:
                        self.producer_queues[item[0]][2].remove(prod)
                        # Invariant: Decrements occupancy to reflect final removal from marketplace.
                        self.producer_queues[item[0]][1] -= 1
                        break
        
        # Finalization: Resets the session state.
        self.carts.pop(cart_id)
        with self.print_mutex:
            for item in result:
                print(threading.current_thread().name, "bought", item)
        logging.info('place_order(%d) returned %s', cart_id, result)
        return result

class TestMarketplace(TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and concurrency safety.
    """
    def setUp(self):
        "Initializare variabile locale"
        self.marketplace = Marketplace(1)
        self.coffee = Coffee('Indonezia', 4, '5.05', 'MEDIUM')
        self.tea = Tea('Linden', 9, 'Herbal')

    def test_register_producer(self):
        "Test test_register_producer"
        id_prod = self.marketplace.register_producer()
        self.assertIsInstance(id_prod, UUID, "Return type not UUID")
        self.assertEqual(self.marketplace.producer_queues[id_prod][1], 0, "Initial size not 0")
        self.assertEqual(len(self.marketplace.producer_queues[id_prod][2]), 0, "Queue not empty")
        self.assertNotEqual(id_prod, self.marketplace.register_producer(), "IDs equal")

    def test_publish(self):
        "Test test_publish"
        id_prod = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(id_prod, self.coffee), "First publish should pass")
        self.assertFalse(self.marketplace.publish(id_prod, self.tea), "Second publish should fail")

    def test_new_cart(self):
        "Test test_new_cart"
        cart_id = self.marketplace.new_cart()
        self.assertGreaterEqual(cart_id, 0, "Cart ID should not be negative")
        self.assertNotEqual(cart_id, self.marketplace.new_cart(), "IDs equal")

    def test_add_to_cart(self):
        "Test test_add_to_cart"
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.coffee), "Coffee is in store")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.tea), "Tea not in store")

    def test_remove_from_cart(self):
        "Test test_remove_from_cart"
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.coffee)
        self.marketplace.remove_from_cart(cart_id, self.tea)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1, "Item not in cart")
        self.marketplace.remove_from_cart(cart_id, self.coffee)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 0, "Item in cart, size 0")

    def test_place_order(self):
        "Test test_place_order"
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.tea)
        self.marketplace.add_to_cart(cart_id, self.coffee)
        self.marketplace.remove_from_cart(cart_id, self.tea)
        self.assertEqual(self.marketplace.place_order(cart_id), [self.coffee], "Results differ")


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continually fulfills production quotas and publishes to the Marketplace.
    Functional Utility: Models the supply chain with simulated processing delays and backpressure handling.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) production metrics.
        @param marketplace Shared resource management interface.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        Thread.__init__(self, **kwargs)
        self.product_actions = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        
        # Initialization: Registers as a supplier to obtain a unique ID.
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        """
        @brief Main industrial loop for the producer thread.
        """
        while True:
            for (product, quantity, delay) in self.product_actions:
                total = 0
                
                # Block Logic: Quota fulfillment with backpressure handling.
                while total < quantity:
                    if self.marketplace.publish(self.id_producer, product):
                        total += 1
                        # Logic: Simulated production duration.
                        sleep(delay)
                    else:
                        # Synchronization: Exponential wait during marketplace inventory overflow.
                        sleep(self.republish_wait_time)
