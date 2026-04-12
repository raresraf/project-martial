"""
Module providing a multi-threaded simulation of a marketplace where consumers
purchase products from producers via a centralized marketplace coordinator.
The system utilizes thread-safe primitives (Semaphores) to manage shared state
and simulates real-world constraints like retry intervals and queue sizes.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents an active agent that interacts with the marketplace to acquire products.
    Acts as a worker thread that executes a series of shopping actions (add/remove)
    defined in its carts and finalizes transactions by placing orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread with its shopping agenda.
        
        Args:
            carts (list): A nested list of shopping actions per cart.
            marketplace (Marketplace): Reference to the central marketplace coordinator.
            retry_wait_time (float): Interval to wait when a product is temporarily unavailable.
            **kwargs: Additional thread configuration parameters.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """
        Executes the consumer's shopping sequence.
        Orchestrates a series of 'add' and 'remove' operations across multiple carts.
        Implements a blocking-with-sleep strategy for adding products that are not currently in stock.
        """
        # Block Logic: Iterates through each cart and its corresponding set of actions.
        for i in range(len(self.carts)):
            for j in range(len(self.carts[i])):
                # Block Logic: Processes individual product actions based on requested quantity.
                for k in range(self.carts[i][j]["quantity"]):
                    if self.carts[i][j]['type'] == 'add':
                        product = self.carts[i][j]['product']
                        # Block Logic: Busy-wait loop with exponential backoff (fixed wait) until product is available.
                        # Invariant: Continues trying to add the product until the marketplace signals success.
                        while not self.marketplace.add_to_cart(self.cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif self.carts[i][j]['type'] == 'remove':
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
        
        # Finalizes the session by converting the cart contents into a finalized order.
        self.marketplace.place_order(self.cart_id)

from threading import Semaphore
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Tea

class Marketplace:
    """
    Acts as a thread-safe mediator between Producers and Consumers.
    Manages global product availability, producer registration, and consumer cart state.
    Utilizes Semaphores to ensure atomic access to shared producer and cart registries.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace with capacity constraints and synchronization primitives.
        
        Args:
            queue_size_per_producer (int): Maximum products allowed in a producer's buffer.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0

        # Data Structures: Map of cart IDs to product lists and Producer IDs to inventory.
        self.carts = {}
        self.producers = {}

        # Synchronization: Semaphores for protecting registry modifications.
        self.sem_carts = Semaphore(value=1)
        self.sem_producers = Semaphore(value=1)

        # Telemetry: Configures rotating logs for audit trails.
        handler = RotatingFileHandler("marketplace.log", maxBytes=500000, backupCount=5)
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Registers a new producer in the system and assigns a unique identifier.
        Ensures thread-safe increment of the global producer counter.
        
        Returns:
            int: The unique ID assigned to the new producer.
        """
        self.logger.info("register_producer started")

        self.sem_producers.acquire()
        self.producer_id += 1
        self.sem_producers.release()

        self.producers[self.producer_id] = []

        self.logger.info("register_producer ended")
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Adds a product to a producer's public inventory if capacity permits.
        
        Args:
            producer_id (int): The ID of the publishing producer.
            product: The product object to be listed.
            
        Returns:
            bool: True if publication succeeded, False if queue is full.
        """
        self.logger.info("publish started")
        self.logger.info("publish parameters: {}, {}".format(producer_id, product))

        # Block Logic: Enforces buffer limits to prevent producer overflow.
        if len(self.producers[producer_id]) < self.queue_size_per_producer:
            self.producers[producer_id].append(product)

            self.logger.info("publish ended")
            return True

        self.logger.info("publish ended")
        return False

    def new_cart(self):
        """
        Allocates a new shopping cart for a consumer.
        Ensures thread-safe allocation of cart identifiers.
        
        Returns:
            int: The unique identifier for the allocated cart.
        """
        self.logger.info("new_cart started")

        self.sem_carts.acquire()
        self.cart_id += 1
        self.sem_carts.release()

        self.carts[self.cart_id] = []

        self.logger.info("register_producer ended")
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers a product from producer inventory to a consumer's cart.
        Performs a search across all active producers for the requested item.
        
        Args:
            cart_id (int): Target cart for the product.
            product: The product being requested.
            
        Returns:
            bool: True if product was found and transferred, False otherwise.
        """
        self.logger.info("add_to_cart started")
        self.logger.info("add_to_cart parameters: {}, {}".format(cart_id, product))

        # Block Logic: Linearly scans producers for product availability.
        for key, value in self.producers.items():
            self.sem_producers.acquire()
            if product in value:
                # Atomic Transfer: Removes from producer and releases lock before adding to cart.
                value.remove(product)
                self.sem_producers.release()
                self.logger.info("add_to_cart ended")
                self.carts[cart_id].append((product, key))
                return True

            self.sem_producers.release()

        self.logger.info("add_to_cart ended")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Returns a product from a consumer's cart back to its original producer's inventory.
        
        Args:
            cart_id (int): Source cart.
            product: Product to be returned.
        """
        self.logger.info("remove_from_cart started")
        self.logger.info("remove_from_cart parameters: {}, {}".format(cart_id, product))

        # Block Logic: Locates the product in the cart while maintaining reference to the source producer.
        for i in range(len(self.carts[cart_id])):
            self.sem_carts.acquire()
            if product == self.carts[cart_id][i][0]:
                # Restoration: Returns item to producer's specific list based on the stored producer ID.
                self.producers[self.carts[cart_id][i][1]].append(product)
                self.carts[cart_id].remove((product, self.carts[cart_id][i][1]))
                self.sem_carts.release()

                self.logger.info("remove_from_cart ended")
                return

            self.sem_carts.release()
        self.logger.info("remove_from_cart ended")

    def place_order(self, cart_id):
        """
        Finalizes the purchase process and logs the completed transaction.
        
        Args:
            cart_id (int): The cart being finalized.
            
        Returns:
            list: The list of products contained in the finalized order.
        """
        self.logger.info("place_order started")
        self.logger.info("place_order parameters: {}".format(cart_id))

        # Block Logic: Prints and records the successful acquisition of each item.
        for i in range(len(self.carts[cart_id])):
            self.sem_carts.acquire()
            print("cons{} bought {}".format(cart_id, self.carts[cart_id][i][0]))
            self.sem_carts.release()

        self.logger.info("place_order ended")
        return [elem[0] for elem in self.carts[cart_id]]

class TestMarketplace(unittest.TestCase):
    """
    Unit test suite for validating the core logic of the Marketplace class.
    Tests registration, publication limits, cart management, and order placement.
    """
    
    def setUp(self):
        """
        Initializes the test environment with a standard marketplace and product sample.
        """
        self.marketplace = Marketplace(3)
        self.product = Tea('Green', 2, 'Tea')

    def test_register_producer(self):
        """
        Validates sequential producer registration and ID assignment.
        """
        iterations = 10

        while iterations > 0:
            producer_id = self.marketplace.register_producer()
            iterations -= 1

        self.assertEqual(producer_id, 10)

    def test_new_cart(self):
        """
        Validates sequential cart allocation and ID assignment.
        """
        iterations = 5

        while iterations > 0:
            cart_id = self.marketplace.register_producer()
            iterations -= 1

        self.assertEqual(cart_id, 5)

    def test_publish(self):
        """
        Verifies that publication respects the queue size limits per producer.
        """
        producer = self.marketplace.register_producer()

        iterations = 3
        while iterations > 0:
            ret = self.marketplace.publish(producer, self.product)
            self.assertEqual(ret, True)
            iterations -= 1

        # Invariant: Subsequent publication should fail when queue_size_per_producer (3) is reached.
        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, False)

    def test_add_to_cart(self):
        """
        Ensures products can only be added to carts if they are published by producers.
        """
        producer = self.marketplace.register_producer()
        cons = self.marketplace.new_cart()

        # Pre-condition: Item is not yet published.
        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, False)

        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, True)

        # Post-condition: Item should now be available for transfer.
        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, True)

    def test_place_order(self):
        """
        Validates that finalized orders contain the exact items added to the cart.
        """
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
        """
        Validates that removing an item from a cart returns it to global availability.
        """
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

        # Action: Remove one item.
        self.marketplace.remove_from_cart(cons, self.product)

        # Verification: Order should now contain only one item.
        ret = self.marketplace.place_order(cons)
        expected_ret = [Tea(name='Green', price=2, type='Tea')]
        self.assertEqual(ret, expected_ret)


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a manufacturing entity that continuously supplies products to the marketplace.
    Operates as a worker thread that generates items based on a production schedule.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer with its production capabilities.
        
        Args:
            products (list): List of (product, quantity, production_time) tuples.
            marketplace (Marketplace): Coordinator for product listings.
            republish_wait_time (float): Interval to wait when marketplace buffer is full.
            **kwargs: Thread configuration.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.cart_id = self.marketplace.register_producer()

    def run(self):
        """
        Infinite loop representing the continuous production cycle.
        Attempts to publish products in the specified order and quantity.
        Implements a back-pressure response (busy-wait) if the marketplace is congested.
        """
        while True:
            # Block Logic: Iterates through the catalog of products to be produced.
            for i in range(len(self.products)):
                for j in range(self.products[i][1]):
                    # Block Logic: Blocking publish call with retry logic.
                    # Invariant: Retries until the marketplace accepts the product publication.
                    while not self.marketplace.publish(self.cart_id, self.products[i][0]):
                        time.sleep(self.republish_wait_time)
                    
                    # Simulates the temporal cost of production for the current item.
                    time.sleep(self.products[i][2])
