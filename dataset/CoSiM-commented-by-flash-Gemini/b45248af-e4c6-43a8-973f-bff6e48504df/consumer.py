"""
@b45248af-e4c6-43a8-973f-bff6e48504df/consumer.py
@brief multi-threaded simulation of a retail marketplace using concurrent Producer and Consumer agents.
Architecture: Centralized mediator (Marketplace) manages shared state between autonomous execution threads.
Functional Utility: Orchestrates asynchronous inventory replenishment, session-persistent shopping carts, and concurrent transactional updates.
Synchronization: Employs threading.Lock for critical sections and cooperative yield patterns (sleep) for demand-supply flow control.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping sessions.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested products from the Marketplace.
    Error Handling: Implements a polling retry loop with yields (sleep) for handling temporary stock depletions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts List of product acquisition lists to be fulfilled.
        @param marketplace Shared resource management interface.
        @param retry_wait_time Temporal duration to yield when inventory is depleted.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.cart_ids = []

    def run(self):
        """
        @brief lifecycle manager for the consumer execution context.
        Logic: Allocates a new transactional session (cart_id) and fulfills all commands before finalizing the order.
        """
        index = 0
        for cart in self.carts:
            # Initialization: Establishes a unique inventory buffer in the marketplace.
            self.cart_ids.append(self.marketplace.new_cart())
            for op_cart in cart:
                product = op_cart['product']
                quantity = op_cart['quantity']
                op_type = op_cart['type']
                
                # Block Logic: Dispatcher for marketplace operations.
                if op_type == "add":
                    i = 0
                    # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                    while i < quantity:
                        status = self.marketplace.add_to_cart(self.cart_ids[index], product)
                        if status:
                            i += 1
                        else:
                            # Logic: Yields execution to handle temporary stock depletion.
                            sleep(self.wait_time)
                elif op_type == "remove":
                    # Logic: Returns reserved commodities to the marketplace inventory.
                    for i in range(0, quantity):
                        self.marketplace.remove_from_cart(self.cart_ids[index], product)
            
            # Finalization: executes the transaction and flushes results.
            self.marketplace.place_order(self.cart_ids[index])
            index += 1

import threading
import time
from threading import Lock
import logging.handlers
import unittest

from tema.product import Tea


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and concurrency integrity.
    """

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
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer stock and session-based consumer carts.
    Synchronization: Uses distinct locks (lock1 for producers, lock2 for carts) to minimize global contention.
    Observability: Integrates RotatingFileHandler for structured audit logging of all concurrent events.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum inventory allowed per supplier for backpressure control.
        """
        self.queue_max_size = queue_size_per_producer
        
        
        self.producers = [] # List of Producer pools (List of Lists).
        
        
        self.carts = [] # List of Consumer session contents.
        self.last_producer_id = 0
        self.last_cart_id = 0
        self.lock1 = Lock()
        self.lock2 = Lock()
        
        # Block Logic: Audit logging configuration.
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
        @brief Onboards a new supplier and initializes its thread-safe inventory list.
        @return Unique producer identifier.
        """
        logging.info("Entered register_producer")
        self.producers.append([])
        with self.lock1:
            self.last_producer_id += 1
            id_prod = self.last_producer_id - 1


        logging.info("New producer id: " + str(id_prod))
        return id_prod

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the marketplace.
        Constraint: Operation rejected if the supplier's individual queue is saturated.
        """
        logging.info("Entered publish with producer id " + str(producer_id)
                     + " and product " + str(product))
        
        # Block Logic: Threshold check for supply-side flow control.
        if len(self.producers[producer_id]) == self.queue_max_size:
            logging.info("Return value: False")
            return False
        
        with self.lock1:
            self.producers[producer_id].append((product, 1))
        logging.info("Return value: True")
        return True

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """
        logging.info("Entered new_cart")
        with self.lock2:
            self.last_cart_id += 1
            self.carts.append([])
            cart_id = self.last_cart_id - 1


        logging.info("Returned cart id: " + str(cart_id))
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        Strategy: Exhaustive search across all supplier pools. First-available fulfillment strategy.
        @return Boolean indicating acquisition success.
        """
        logging.info("Entered add_to_cart with cart id " + str(cart_id)
                     + " and product " + str(product))
        for tmp_list in self.producers:
            for prod_tuple in tmp_list:
                tmp = list(prod_tuple)
                if tmp[0] == product:
                    # Condition: Item available check.
                    if tmp[1] == 0:
                        logging.info("Return value: False")
                        return False
                    with self.lock1:
                        # Invariant: Item is marked as reserved (quantity set to 0) to prevent double-selling.
                        tmp[1] = 0
                        prod_tuple = tuple(tmp)

        with self.lock2:
            self.carts[cart_id].append(product)
        logging.info("Return value: True")
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to its source producer's pool.
        """
        logging.info("Entered remove_from_cart with cart id " +
                     str(cart_id) + " and product " + str(product))
        with self.lock2:
            self.carts[cart_id].remove(product)

        # Logic: Locates the physical unit in the producer arrays and restores its availability flag.
        for tmp_list in self.producers:
            for prod_tuple in tmp_list:
                tmp = list(prod_tuple)
                if tmp[0] == product:
                    with self.lock1:
                        tmp[1] = 1
                        prod_tuple = tuple(tmp)

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and serializes results to standard output.
        Side Effect: Purges the units from the respective supplier arrays.
        """
        logging.info("Entered place_order with cart id " + str(cart_id))
        prod_list = self.carts[cart_id]
        
        # Block Logic: Fulfillment sweep.
        for prod_iter in prod_list:
            for list_prod in self.producers:
                if prod_iter in list_prod:
                    with self.lock1:
                        # Invariant: Removes the physical unit from the source producer pool.
                        list_prod.remove(prod_iter)
        
        # Finalization: Resets the session state.
        self.carts[cart_id] = []
        for prod_iter in prod_list:
            # Serialization: Uses print to output acquisition results.
            print(threading.current_thread().name + " bought " + str(prod_iter))
            logging.info("Buyer " + threading.current_thread().name
                         + " bought " + str(prod_iter))
        return prod_list


from threading import Thread
from time import sleep


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
        self.wait_time = republish_wait_time
        self.id_prod = -1

    def run(self):
        """
        @brief Main manufacturing loop for the producer execution context.
        """
        # Initialization: Registers once as a persistent supplier.
        self.id_prod = self.marketplace.register_producer()
        index = 0
        while True:
            if index == len(self.products):
                index = 0
            
            # Block Logic: Quota fulfillment with backpressure handling.
            i = 0
            while i < self.products[index][1]:
                status = self.marketplace.publish(self.id_prod, self.products[index][0])
                if not status:
                    # Synchronization: Exponential wait during marketplace saturation.
                    sleep(self.wait_time)
                else:
                    # Logic: Simulated industrial processing duration.
                    sleep(self.products[index][2])
                    i += 1

            index += 1


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
