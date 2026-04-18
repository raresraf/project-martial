"""
@aaa44c49-1677-4e8e-8951-e6676095b38b/consumer.py
@brief Event-driven simulation of a retail marketplace using multi-threaded Producer and Consumer agents.
Architecture: Centralized Marketplace mediator manages shared state, coordinating autonomous threads for asynchronous commerce.
Functional Utility: Handles inventory management, virtual shopping cart persistence, and concurrent transactional updates.
Synchronization: Employs threading.Semaphore for critical sections and cooperative yield patterns (sleep) for demand-supply flow control.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing high-level shopping requests.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested products from the Marketplace.
    Error Handling: Implements a polling retry mechanism with configurable delays for out-of-stock scenarios.
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
        # Initialization: Establishes a session-scoped inventory buffer in the marketplace.
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """
        @brief lifecycle manager for the consumer thread.
        Logic: Processes all assigned shopping lists, fulfilling each requested item before finalizing the order.
        """
        for i in range(len(self.carts)):
            for j in range(len(self.carts[i])):
                # Block Logic: Fulfillment loop for a specific product request.
                for k in range(self.carts[i][j]["quantity"]):
                    if self.carts[i][j]['type'] == 'add':
                        product = self.carts[i][j]['product']
                        # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                        while not self.marketplace.add_to_cart(self.cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif self.carts[i][j]['type'] == 'remove':
                        # Logic: Returns reserved commodities to the marketplace inventory.
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
        
        # Finalization: executes the transaction and flushes results.
        self.marketplace.place_order(self.cart_id)

from threading import Semaphore
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Tea

class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer stock and session-based consumer carts.
    Synchronization: Uses distinct semaphores for cart (sem_carts) and producer (sem_producers) registries to minimize contention.
    Observability: Integrates RotatingFileHandler for structured audit logging of all concurrent events.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum inventory allowed per supplier for backpressure control.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0

        self.carts = {} # Mapping: CartID -> List of [Product, SourceProducerID] tuples.
        self.producers = {} # Mapping: ProducerID -> List of available Products.

        self.sem_carts = Semaphore(value=1) # Mutex for global cart registry.
        self.sem_producers = Semaphore(value=1) # Mutex for global producer registry.

        # Block Logic: Audit logging configuration.
        handler = RotatingFileHandler("marketplace.log", maxBytes=500000, backupCount=5)
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its inventory tracking.
        @return Unique producer identifier.
        """
        self.logger.info("register_producer started")

        with self.sem_producers:
            self.producer_id += 1
            id_to_return = self.producer_id

        # Initialization: Scaffolds the inventory pool for the new producer.
        self.producers[id_to_return] = []

        self.logger.info("register_producer ended")
        return id_to_return

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to their marketplace pool.
        Constraint: Operation rejected if the supplier's individual queue is saturated.
        """
        self.logger.info("publish started")
        self.logger.info("publish parameters: {}, {}".format(producer_id, product))

        # Block Logic: Threshold check for supply-side flow control.
        if len(self.producers[producer_id]) < self.queue_size_per_producer:
            self.producers[producer_id].append(product)

            self.logger.info("publish ended")
            return True

        self.logger.info("publish ended")
        return False

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """
        self.logger.info("new_cart started")

        with self.sem_carts:
            self.cart_id += 1
            id_to_return = self.cart_id

        self.carts[id_to_return] = []

        self.logger.info("register_producer ended")
        return id_to_return

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        Strategy: Exhaustive search across all supplier pools. First-available fulfillment strategy.
        @return Boolean indicating acquisition success.
        """
        self.logger.info("add_to_cart started")
        self.logger.info("add_to_cart parameters: {}, {}".format(cart_id, product))

        for key, value in self.producers.items():
            # Block Logic: Search pass under producer-specific lock.
            with self.sem_producers:
                if product in value:
                    # Invariant: Item must be removed from global pool before being assigned to a session.
                    value.remove(product)
                    self.logger.info("add_to_cart ended")
                    self.carts[cart_id].append((product, key))
                    return True

        self.logger.info("add_to_cart ended")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to its originating producer's pool.
        """

        self.logger.info("remove_from_cart started")
        self.logger.info("remove_from_cart parameters: {}, {}".format(cart_id, product))

        for i in range(len(self.carts[cart_id])):
            with self.sem_carts:
                if product == self.carts[cart_id][i][0]:
                    # Logic: Identifies original producer from cached metadata and restores unit.
                    self.producers[self.carts[cart_id][i][1]].append(product)
                    self.carts[cart_id].remove((product, self.carts[cart_id][i][1]))

                    self.logger.info("remove_from_cart ended")
                    return

        self.logger.info("remove_from_cart ended")

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and serializes results to standard output.
        Side Effect: Outputs acquisition logs and returns the aggregate product list.
        """
        self.logger.info("place_order started")
        self.logger.info("place_order parameters: {}".format(cart_id))

        # Block Logic: Finalization loop.
        for i in range(len(self.carts[cart_id])):
            with self.sem_carts:
                # Serialization: Serializes output to prevent interleaved lines from multiple threads.
                print("cons{} bought {}".format(cart_id, self.carts[cart_id][i][0]))

        self.logger.info("place_order ended")
        return [elem[0] for elem in self.carts[cart_id]]

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and concurrency integrity.
    """
    
    def setUp(self):
        
        self.marketplace = Marketplace(3)
        self.product = Tea('Green', 2, 'Tea')

    def test_register_producer(self):
        
        iterations = 10

        while iterations > 0:
            producer_id = self.marketplace.register_producer()
            iterations -= 1

        self.assertEqual(producer_id, 10)

    def test_new_cart(self):
        
        iterations = 5

        while iterations > 0:
            cart_id = self.marketplace.new_cart()
            iterations -= 1

        self.assertEqual(cart_id, 5)

    def test_publish(self):
        
        producer = self.marketplace.register_producer()

        iterations = 3
        while iterations > 0:
            ret = self.marketplace.publish(producer, self.product)
            self.assertEqual(ret, True)
            iterations -= 1

        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, False)

    def test_add_to_cart(self):
        
        producer = self.marketplace.register_producer()
        cons = self.marketplace.new_cart()

        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, False)

        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, True)

        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, True)

    def test_place_order(self):
        
        producer = self.marketplace.register_producer()
        cons = self.marketplace.new_cart()

        iterations = 3
        while iterations > 0:
            ret = self.marketplace.publish(producer, self.product)
            self.assertEqual(ret, True)
            iterations -= 1

        iterations = 3
        while iterations > 0:
            ret = self.marketplace.add_to_cart(cons, self.product)
            self.assertEqual(ret, True)
            iterations -= 1

        ret = self.marketplace.place_order(cons)
        expected_ret = [Tea(name='Green', price=2, type='Tea')] * 3
        self.assertEqual(ret, expected_ret)

    def test_remove_from_cart(self):
        
        producer = self.marketplace.register_producer()
        cons = self.marketplace.new_cart()

        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, True)

        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, True)

        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, True)

        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, True)

        self.marketplace.remove_from_cart(cons, self.product)

        ret = self.marketplace.place_order(cons)
        expected_ret = [Tea(name='Green', price=2, type='Tea')]
        self.assertEqual(ret, expected_ret)


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continuously fulfills production quotas and publishes results to the Marketplace mediator.
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
        # Initialization: Registers as a supplier to obtain a unique ID.
        self.cart_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Main industrial loop for the producer thread.
        """
        while True:
            for i in range(len(self.products)):
                # Block Logic: Quota fulfillment.
                for j in range(self.products[i][1]):
                    # Synchronization: Publish-retry loop for backpressure management.
                    while not self.marketplace.publish(self.cart_id, self.products[i][0]):
                        time.sleep(self.republish_wait_time)
                    # Logic: Simulated industrial processing duration.
                    time.sleep(self.products[i][2])
