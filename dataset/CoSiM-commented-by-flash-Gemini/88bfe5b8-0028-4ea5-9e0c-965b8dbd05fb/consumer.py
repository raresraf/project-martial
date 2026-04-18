"""
@88bfe5b8-0028-4ea5-9e0c-965b8dbd05fb/consumer.py
@brief Event-driven simulation of a retail marketplace using multi-threaded Producer and Consumer agents.
Architecture: Centralized Marketplace mediator manages shared state, coordinating autonomous threads for asynchronous commerce.
Functional Utility: Models inventory flow, virtual shopping cart persistence, and atomic transaction fulfillment.
Synchronization: Employs threading.Lock for serializing state transitions and cooperative sleep intervals for demand-supply flow control.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping strategies.
    Logic: Processes a collection of carts, each containing a sequence of inventory mutations (add/remove).
    Functional Utility: Implements a persistent polling loop to fulfill product acquisition despite transient stock depletion.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts List of product acquisition lists to be fulfilled.
        @param marketplace Shared resource management interface.
        @param retry_wait_time Temporal duration to yield when the marketplace is depleted.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        @brief Main execution lifecycle for the consumer thread.
        Logic: Allocates a unique session (cart_id) and executes all internal requests before finalizing the order.
        """
        for cart in self.carts:
            # Initialization: Establishes a session-scoped inventory buffer.
            cart_id = self.marketplace.new_cart()


            for operation in cart:
                
                # Block Logic: Dispatcher for marketplace operations (acquisition or return).
                if operation["type"] == "add":
                    count = 0
                    
                    # Synchronization: Fulfillment barrier. Ensures the thread blocks until all requested units are acquired.
                    while count < operation["quantity"]:
                        
                        if self.marketplace.add_to_cart(cart_id, operation["product"]):


                            count += 1
                        else:
                            # Logic: Yields to allow producers time to replenish the global pool.
                            sleep(self.retry_wait_time)
                
                elif operation["type"] == "remove":
                    count = 0
                    
                    while count < operation["quantity"]:
                        # Logic: Returns reserved units back to the supplier inventory.
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        
                        count += 1
             
            # Finalization: Commits the transaction and outputs acquisition results.
            products_bought = self.marketplace.place_order(cart_id)
            for product in products_bought:
                print(self.kwargs["name"], "bought", product, flush=True)

# Note: The following section was originally in marketplace.py
# >>>> file: marketplace.py


from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
from distutils.log import INFO
import time
import unittest
import sys
sys.path.insert(1, './tema')
from product import Tea, Coffee

class Marketplace:
    """
    @brief Centralized resource manager for inventory tracking and transactional synchronization.
    State Management: Maintains isolated inventory pools for each producer and session-based carts for consumers.
    Synchronization: Uses a central mutex (Lock) to protect critical updates to shared dictionaries and sequence counters.
    Observability: Integrates RotatingFileHandler for structured audit logging of all concurrent operations.
    """
    
    # Block Logic: Static configuration for system-wide auditing.
    myLogger = logging.getLogger('marketplace.log')
    myLogger.setLevel(INFO)
    file_handler = RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=5)
    file_handler.setLevel(INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    formatter.converter = time.gmtime
    file_handler.setFormatter(formatter)
    myLogger.addHandler(file_handler)

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Capacity limit for each supplier's inventory pool to prevent buffer overflow.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        
        self.available_products = [] # Mapping: ProducerID -> List of Products.
        self.cart_id = -1
        
        self.carts = [] # Mapping: CartID -> List of [Product, SourceProducerID] tuples.
        
        self.queue_size = [] # Tracks current utilization per producer pool.

        self.lock = Lock() # Core synchronization primitive.

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its telemetry metrics.
        @return Unique producer identifier.
        """
        self.lock.acquire()
        self.myLogger.info("Entered method register_producer")
        
        self.producer_id += 1
        
        # Initialization: Allocates resources for the new producer's inventory lifecycle.
        self.available_products.append([])
        self.available_products[self.producer_id] = []
        self.queue_size.append(0)
        
        self.myLogger.info("Exited method register_producer with producer_id=%s", self.producer_id)
        self.lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add inventory to its respective marketplace pool.
        Constraint: Enforces the queue_size_per_producer limit for backpressure.
        """
        self.myLogger.info("Entered method publish with producer_id=%s, product=%s",
                           producer_id, product)
        id_producer = int(producer_id)
        
        # Block Logic: Occupancy check.
        if self.queue_size[id_producer] < self.queue_size_per_producer:
            
            # Logic: Adds the item to the isolated pool.
            self.available_products[id_producer].append(product[0])
            self.queue_size[id_producer] += 1
            self.myLogger.info("Exited method publish with True")
            return True
        self.myLogger.info("Exited method publish with False")
        return False

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """
        self.lock.acquire()
        self.myLogger.info("Entered method new_cart")
        self.cart_id += 1
        
        self.carts.append([])
        self.carts[self.cart_id] = []
        self.myLogger.info("Exited method new_cart with cart_id=%s", self.cart_id)
        self.lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a product from any available producer pool to a specific consumer cart.
        Logic: Iteratively scans all producer pools until the requested product is located.
        @return Boolean indicating acquisition success.
        """
        self.lock.acquire()
        self.myLogger.info("Entered method add_to_cart with cart_id=%s, product=%s",
                           cart_id, product)
        ids = 0
        
        # Block Logic: Global search and reserve.
        while ids <= self.producer_id:
            
            if product in self.available_products[ids]:
                # Invariant: Item must be removed from the producer's pool to prevent double-selling.
                self.carts[cart_id].append([product, ids])
                
                self.available_products[ids].remove(product)
                self.myLogger.info("Exited method add_to_cart with True")
                self.lock.release()
                return True
            
            ids += 1
        self.myLogger.info("Exited method add_to_cart with False")
        self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the item to its original supplier.
        """
        self.myLogger.info("Entered method remove_from_cart with cart_id=%s, product=%s",
                           cart_id, product)
        
        for produs in self.carts[cart_id]:
            if produs[0] == product:
                
                # Logic: Uses the source producer ID cached in the cart entry to route the return.
                self.carts[cart_id].remove([product, produs[1]])
                self.available_products[produs[1]].append(product)
                self.myLogger.info("Exited method remove_from_cart")
                return

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and updates global utilization metrics.
        """
        self.myLogger.info("Entered place_order with cart_id=%s", cart_id)
        cart_products = []
        
        # Finalization: Flushes the cart and decrements active queue sizes for each source producer.
        for products in self.carts[cart_id]:
            cart_products.append(products[0])
            
            self.lock.acquire()
            self.queue_size[products[1]] -= 1
            self.lock.release()
        self.myLogger.info("Exited place_order.")
        return cart_products

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for validating Marketplace transactional logic and concurrency safety.
    """
    def setUp(self):
        
        
        self.marketplace = Marketplace(2)
        
        self.tea = Tea(name='Linden', price=9, type='Herbal')
        self.coffee = Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.register_producer(), 3)

    def test_publish(self):
        
        
        self.assertTrue(self.marketplace.publish(0, self.tea))
        self.assertListEqual(self.marketplace.available_products[0], [self.tea])
        self.assertEqual(self.marketplace.queue_size[0], 1)
        
        self.assertTrue(self.marketplace.publish(0, self.coffee))
        self.assertListEqual(self.marketplace.available_products[0], [self.tea, self.coffee])
        self.assertEqual(self.marketplace.queue_size[0], 2)
        
        self.assertFalse(self.marketplace.publish(0, self.tea))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.new_cart(), 3)

    def test_add_to_cart(self):
        
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        self.assertFalse(self.marketplace.add_to_cart(0, self.coffee))

    def test_remove_from_cart(self):
        
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        self.marketplace.remove_from_cart(0, self.tea)
        
        self.assertListEqual(self.marketplace.carts[0], [])
        
        self.assertListEqual(self.marketplace.available_products[0], [self.tea])

    def test_place_order(self):
        
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        self.assertEqual(self.marketplace.queue_size[0], 1)
        
        cart_list = self.marketplace.place_order(0)
        self.assertListEqual(cart_list, [self.tea])
        
        
        self.assertEqual(self.marketplace.queue_size[0], 0)


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continuously fulfills production quotas and publishes to the Marketplace mediator.
    Functional Utility: Models the supply chain with simulated manufacturing delays.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) metrics.
        @param marketplace Shared resource management interface.
        @param republish_wait_time Duration to wait when the marketplace is full.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Main industrial loop for the producer thread.
        """


        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                count = 0
                
                # Block Logic: Quota fulfillment with backpressure handling.
                while count < product[1]:
                    
                    if self.marketplace.publish(producer_id, product):
                        # Logic: Simulated production latency.
                        sleep(product[2])
                        
                        count += 1
                    else:
                        # Synchronization: Exponential wait during marketplace saturation.
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base entity for marketable goods.
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
    @brief Specialized beverage commodity with acidity and roasting metrics.
    """
    acidity: str
    roast_level: str
