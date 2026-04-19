"""
@cbcc0b14-6d5e-4d5a-af91-1e77ff371d24/consumer.py
@brief multi-threaded simulation of a retail marketplace using autonomous Producer and Consumer agents.
Architecture: Decoupled design where a centralized Marketplace mediator manages global state (inventory, carts).
Functional Utility: Handles asynchronous inventory replenishment, session-persistent shopping carts, and concurrent transactional updates.
Synchronization: Employs threading.Lock for critical sections and cooperative yield patterns (sleep) for demand-supply flow control.
"""

from threading import Thread, Lock
import time


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing high-level shopping requests.
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


        self.retry_wait_time = retry_wait_time
        self.cart_id = 0

    def run(self):
        """
        @brief lifecycle manager for the consumer thread.
        Logic: Allocates a unique session (cart_id) and fulfills all commands before finalizing the order.
        Synchronization: uses a local lock to protect session ID acquisition.
        """
        for cart in self.carts:
            lock = Lock()
            lock.acquire()
            self.cart_id = self.marketplace.new_cart()
            lock.release()

            for ops in cart:
                type_operation = ops['type']
                product = ops['product']
                quantity = ops['quantity']
                i = 0

                # Block Logic: Fulfillment loop.
                if type_operation == "add":
                    # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                    while i < quantity:
                        status = self.marketplace.add_to_cart(self.cart_id, product)
                        if not status:
                            time.sleep(self.retry_wait_time)
                        else:
                            i += 1
                else:
                    # Logic: Returns reserved commodities to the marketplace inventory.
                    while i < quantity:
                        self.marketplace.remove_from_cart(self.cart_id, product)
                        i += 1

            # Finalization: executes the transaction and flushes results to standard output.
            placed_order_cart = self.marketplace.place_order(self.cart_id)

            lock = Lock()
            for product_bought in placed_order_cart:
                # Synchronization: Serializes output to prevent interleaved log lines from concurrent consumers.
                lock.acquire()
                print("{} bought {}".format(self.name, product_bought))
                lock.release()

import logging
from logging.handlers import RotatingFileHandler
import time
import unittest
from dataclasses import dataclass

class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for per-producer occupancy, global product availability, and active carts.
    Observability: Integrates RotatingFileHandler for structured audit logging of all concurrent events.
    """

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on inventory per supplier for backpressure management.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.count_producers = 0  
        self.carts = []  
        self.producer_products = []  
        self.reserved_products = [] # Global Registry: Tracks reserved items per producer.
        
        # Block Logic: Audit logging infrastructure.
        # Functional Utility: Persistent log with rotation to prevent disk exhaustion.
        logger = logging.getLogger('my_logger') 
        logger.setLevel(logging.INFO) 
        handler = RotatingFileHandler('marketplace.log', maxBytes=2000, backupCount=10)
        formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
        handler.setFormatter(formatter)
        logging.Formatter.converter = time.gmtime
        logger.addHandler(handler)

        logger.info("Marketplace created")

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its inventory tracking.
        @return Unique producer identifier.
        """
        logger = logging.getLogger('my_logger')
        logger.info("Producer registration started")

        # Initialization: Scaffolds the tracking structures for the new producer.
        self.producer_products.append([])
        self.reserved_products.append([])
        self.count_producers = self.count_producers + 1

        logger.info("Producer registration finished")
        return self.count_producers - 1

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to their marketplace pool.
        Constraint: Operation rejected if the supplier's individual queue is saturated.
        """

        logger = logging.getLogger('my_logger')
        logger.info("Product publishing started")

        # Block Logic: Threshold check for supply-side flow control.
        if len(self.producer_products[producer_id]) < self.queue_size_per_producer:
            self.producer_products[producer_id].append(product)

            logger.info("Product publishing finished successfully")
            return True

        logger.info("Product publishing: Caller should wait")
        return False

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """

        logger = logging.getLogger('my_logger')
        logger.info("Cart creation started")

        self.carts.append([])
        logger.info("Cart creation finished")
        return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        Logic: Scans all producer pools. First-available fulfillment strategy.
        @return Boolean indicating acquisition success.
        """
        
        logger = logging.getLogger('my_logger')
        logger.info("Product adding in cart started")

        for i in range(self.count_producers):

            # Block Logic: Acquisition check across all suppliers.
            if product in self.producer_products[i]:
                # Invariant: Item must be removed from global pool and marked as reserved for potential returns.
                self.carts[cart_id].append(product)
                self.reserved_products[i].append(product)
                self.producer_products[i].remove(product)
                return True

        logger.info("Product added in cart successfully")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to its originating producer's pool.
        """

        logger = logging.getLogger('my_logger')
        logger.info("Product removing started")


        # Invariant: removes the product from the specific consumer cart.
        self.carts[cart_id].remove(product)

        # Logic: Uses the reserved_products tracker to route the unit back to its original supplier.
        for i in range(self.count_producers):
            if product in self.reserved_products[i]:
                self.reserved_products[i].remove(product)
                self.producer_products[i].append(product)
                return True

        logger.info("Product removing finished")
        return False

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and flushes results.
        """

        logger = logging.getLogger('my_logger')
        logger.info("Order placing finished successfully")
        return self.carts[cart_id]


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable representation of a marketable commodity unit.
    """
    name: str
    price: int

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and state transitions.
    """
    
    def setUp(self):
        
        logging.disable(logging.CRITICAL)
        self.marketplace = Marketplace(10)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 0,
                         'wrong producer id')
        self.assertEqual(len(self.marketplace.producer_products), 1,
                         'wrong producer products size')

    def test_publish(self):
        
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.assertTrue(self.marketplace.publish(0, product),
                        'product not published')
        self.assertEqual(len(self.marketplace.producer_products[0]), 1,
                         'wrong producer products size')

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0,
                         'wrong cart id')
        self.assertEqual(len(self.marketplace.carts), 1,
                         'wrong carts size')

    def test_add_to_cart(self):
        
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product),
                        'product not added to cart')
        self.assertEqual(len(self.marketplace.carts[0]), 1,
                         'wrong cart size')
        self.assertEqual(len(self.marketplace.producer_products[0]), 0,
                         'wrong producer products size')

    def test_remove_from_cart(self):
        
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product),
                        'product not removed from cart')
        self.assertEqual(len(self.marketplace.carts[0]), 0,
                         'wrong cart size')
        self.assertEqual(len(self.marketplace.producer_products[0]), 1,
                         'wrong producer products size')

    def test_place_order(self):
        
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.place_order(cart_id), [product],
                         'wrong order')


from threading import Thread, Lock
import time


class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continually fulfills production quotas and publishes results to the Marketplace.
    Functional Utility: Models manufacturing latencies and handles supply-side flow control.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) production metrics.
        @param marketplace Shared resource mediator.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        Thread.__init__(self, **kwargs)


        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = 0

    def run(self):
        """
        @brief Main manufacturing loop for the producer execution context.
        Synchronization: uses a local lock to protect registration.
        """
        lock = Lock()
        lock.acquire()
        self.producer_id = self.marketplace.register_producer()
        lock.release()

        while True:
            for product in self.products:
                product_id = product[0]
                quantity = product[1]
                waiting_time = product[2]
                i = 0

                # Block Logic: Quota fulfillment with backpressure handling.
                while i < quantity:
                    status = self.marketplace.publish(self.producer_id, product_id)
                    if not status:
                        # Synchronization: Exponential wait during marketplace saturation.
                        time.sleep(self.republish_wait_time)
                    else:
                        i += 1
                        # Logic: Simulated industrial processing duration.
                        time.sleep(waiting_time)
