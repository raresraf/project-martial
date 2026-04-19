"""
@bcf08810-3fba-4b78-b768-f749184f5985/consumer.py
@brief multi-threaded simulation of a retail marketplace using concurrent Producer and Consumer agents.
Architecture: Centralized Marketplace mediator manages shared state, coordinating autonomous threads for asynchronous commerce.
Functional Utility: Handles inventory flow control, virtual shopping cart management, and transactional order fulfillment.
Synchronization: Employs threading.Lock for serializing state transitions and cooperative sleep intervals for demand-supply flow control.
"""

from time import sleep
from threading import Thread


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing high-level shopping requests.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested products from the Marketplace.
    Error Handling: Implements a polling retry loop with yields (sleep) for handling temporary stock depletions.
    """
    
    carts = []
    marketplace = None
    retry_wait_time = -1
    name = None

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts List of product acquisition lists to be fulfilled.
        @param marketplace Shared resource management interface.
        @param retry_wait_time Duration to wait when the marketplace is depleted.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts


        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']
        

    def run(self):
        """
        @brief lifecycle manager for the consumer execution context.
        Logic: Allocates a new transactional session (cart_id) and fulfills all commands before finalizing the order.
        """
        for cart in self.carts:
            # Initialization: Establishes a unique inventory buffer in the marketplace.
            cart_id = self.marketplace.new_cart()
            for cmd in cart:
                cmd_type = cmd['type']
                product = cmd['product']
                quantity = cmd['quantity']

                # Block Logic: Dispatcher for marketplace operations.
                if cmd_type == 'add':
                    i = 0
                    # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                    while i < quantity:
                        product_added = self.marketplace.add_to_cart(cart_id, product)
                        
                        if product_added:
                            i += 1
                        
                        else:
                            # Logic: Yields execution to handle temporary stock depletion.
                            sleep(self.retry_wait_time)
                elif cmd_type == 'remove':
                    # Logic: Returns reserved commodities to the marketplace inventory.
                    for i in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

            # Finalization: executes the transaction and flushes results.
            products = self.marketplace.place_order(cart_id)
            for i in products:
                print(self.name + ' bought ' + str(i))

import time
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import unittest
from random import randint
from tema.product import Product, Tea, Coffee


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and concurrency safety.
    """

    def setUp(self):
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        
        old_id = -1
        new_id = -1
        for _ in range(randint(3, 100)):
            old_id = self.marketplace.producers_ids
            new_id = self.marketplace.register_producer()
        self.assertEqual(old_id + 1, new_id)

    def test_new_cart(self):
        
        old_cart_id = -1
        new_cart_id = -1
        for _ in range(randint(3, 100)):
            old_cart_id = self.marketplace.carts_ids
            new_cart_id = self.marketplace.new_cart()
        self.assertEqual(old_cart_id + 1, new_cart_id)

    def test_publish_true(self):
        
        max_len = self.marketplace.queue_size_per_producer
        id_prod = self.marketplace.register_producer()
        for _ in range(randint(0, max_len - 2)):
            published = self.marketplace.publish(id_prod, Tea('test_tea', 10, 'test_type'))
            self.assertTrue(published)

    def test_publish_false(self):
        
        published = False
        max_len = self.marketplace.queue_size_per_producer
        id_prod = self.marketplace.register_producer()
        for _ in range(randint(max_len + 1, 2 * max_len)):
            published = self.marketplace.publish(id_prod, Tea('test_tea', 10, 'test_type'))
        self.assertFalse(published)

    def test_add_to_cart_true(self):
        
        cart = self.marketplace.new_cart()

        id1 = self.marketplace.register_producer()
        id2 = self.marketplace.register_producer()

        product = Tea('test_tea', 10, 'test_type')

        published = self.marketplace.publish(id1, product)
        self.assertTrue(published)

        published = self.marketplace.publish(id2, product)
        self.assertTrue(published)

        found = self.marketplace.add_to_cart(cart, product)
        self.assertTrue(found)

    def test_add_to_cart_false(self):
        
        cart = self.marketplace.new_cart()

        id1 = self.marketplace.register_producer()
        id2 = self.marketplace.register_producer()

        product1 = Tea('test_tea', 10, 'test_type')
        product2 = Coffee('test_coffee', 20, 'test', 'test')

        published = self.marketplace.publish(id1, product1)
        self.assertTrue(published)

        published = self.marketplace.publish(id2, product1)
        self.assertTrue(published)

        found = self.marketplace.add_to_cart(cart, product2)
        self.assertFalse(found)

    def test_remove_from_cart(self):
        
        cart = self.marketplace.new_cart()
        id1 = self.marketplace.register_producer()
        product = Tea('test_tea', 10, 'test_type')

        published = self.marketplace.publish(id1, product)
        self.assertTrue(published)

        found = self.marketplace.add_to_cart(cart, product)
        self.assertTrue(found)

        dim_before = len(self.marketplace.carts[cart])
        self.marketplace.remove_from_cart(cart, product)
        dim_after = len(self.marketplace.carts[cart])

        self.assertTrue(dim_before > dim_after)

    def test_place_order(self):
        
        c_1 = self.marketplace.new_cart()
        c_2 = self.marketplace.new_cart()
        id_1 = self.marketplace.register_producer()
        id_2 = self.marketplace.register_producer()
        p_1 = Tea('test_tea', 10, 'test_type')
        p_2 = Coffee('test_coffee', 20, 'test', 'test')

        published = self.marketplace.publish(id_1, p_2)
        self.assertTrue(published)
        published = self.marketplace.publish(id_2, p_1)
        self.assertTrue(published)
        published = self.marketplace.publish(id_2, p_2)
        self.assertTrue(published)
        published = self.marketplace.publish(id_1, p_1)
        self.assertTrue(published)

        found = self.marketplace.add_to_cart(c_2, p_2)
        self.assertTrue(found)
        found = self.marketplace.add_to_cart(c_1, p_1)
        self.assertTrue(found)
        found = self.marketplace.add_to_cart(c_1, p_2)
        self.assertTrue(found)
        found = self.marketplace.add_to_cart(c_1, p_1)
        self.assertTrue(found)

        prod = self.marketplace.place_order(c_1)
        self.assertEqual(len(prod), 3)
        self.assertEqual(prod[0], p_1)
        self.assertEqual(prod[1], p_2)
        self.assertEqual(prod[2], p_1)


class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer stock queues and session-based consumer carts.
    Synchronization: Uses a central mutex (Lock) to protect critical registry updates and occupancy checks.
    Observability: Integrates RotatingFileHandler for structured audit logging of all concurrent events.
    """
    
    queue_size_per_producer = -1
    producers_ids = -1
    carts_ids = -1
    
    producers_queues = {} # Mapping: ProducerID -> List of [Product, ReservationStatus].
    carts = {} # Mapping: CartID -> List of (Product, SourceProducerID).
    
    lock = Lock() # Core synchronization primitive.

    # Block Logic: Audit logging configuration.
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        handlers=[RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=10)],
        level=logging.INFO,
        format="[%(asctime)s] - [%(levelname)s] : %(funcName)s:%(lineno)d -> %(message)s",
        datefmt='%Y-%m-%d  %H:%M:%S'
    )

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum inventory allowed per supplier for backpressure control.
        """


        self.queue_size_per_producer = queue_size_per_producer
        

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its thread-safe inventory list.
        @return Unique producer identifier.
        """
        
        logging.info('ENTER')

        with self.lock:
            self.producers_ids += 1
            new_id = self.producers_ids
        # Initialization: Scaffolds the inventory pool for the new producer.
        self.producers_queues[new_id] = []

        logging.info('EXIT')
        return new_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the marketplace.
        Constraint: Rejects publication if the supplier's individual queue is saturated (backpressure).
        """
        
        logging.info('ENTER\n %s %s', str(producer_id), str(product))

        if len(self.producers_queues[producer_id]) < self.queue_size_per_producer:
            
            # Logic: Item is stored with a reservation sentinel (-1).
            # Invariant: Item remains in the producer's pool until final order placement.
            item = [product, -1]
            self.producers_queues[producer_id].append(item)
            logging.info('EXIT')
            return True

        logging.info('EXIT')
        return False

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """
        
        logging.info('ENTER')

        with self.lock:
            self.carts_ids += 1
            new_cart = self.carts_ids
        self.carts[new_cart] = []

        logging.info('EXIT')
        return new_cart

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from any available producer pool to a specific cart.
        Strategy: Exhaustive search across all supplier pools. First-available fulfillment strategy.
        @return Boolean indicating acquisition success.
        """
        
        logging.info('ENTER\n %s %s', str(cart_id), str(product))

        
        for key, value in self.producers_queues.items():
            for product_tuple in value:
                with self.lock:
                    
                    # Logic: Identifies an unreserved unit (-1) matching the requested product.
                    if product_tuple[0] == product and product_tuple[1] == -1:
                        
                        # Invariant: Marks the unit as reserved for the specific session.
                        product_tuple[1] = cart_id
                        
                        # Logic: Caches unit metadata in the session cart.
                        self.carts[cart_id].append((product, key))
                        logging.info('EXIT')
                        return True

        logging.info('EXIT')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to an available state.
        """
        
        logging.info('ENTER\n %s %s', str(cart_id), str(product))

        for product_tuple in self.carts.get(cart_id):
            if product_tuple[0] == product:
                producer = product_tuple[1]
                
                # Logic: Finds the specific reserved unit and resets its status sentinel.
                for item in self.producers_queues[producer]:
                    with self.lock:
                        
                        if item[0] == product and item[1] == cart_id:
                            
                            item[1] = -1
                            break
                self.carts[cart_id].remove(product_tuple)
                break

        logging.info('EXIT')

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and flushes results.
        Side Effect: Synchronizes global inventory by removing reserved units from supplier pools.
        """
        
        logging.info('ENTER\n %s', str(cart_id))

        products = []
        for product_tuple in self.carts[cart_id]:
            product = product_tuple[0]
            producer_id = product_tuple[1]
            products.append(product)
            
            # Logic: Atomic removal from supplier's internal list.
            for item in self.producers_queues[producer_id]:
                with self.lock:
                    if item[0] == product and item[1] == cart_id:
                        self.producers_queues[producer_id].remove(item)
                        break

        logging.info('EXIT')
        return products

from time import sleep
from threading import Thread


class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continually fulfills production quotas and publishes results to the Marketplace mediator.
    Functional Utility: Models manufacturing latencies and handles supply-side flow control.
    """
    
    p_id = -1
    products = []
    marketplace = None
    republish_wait_time = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) production metrics.
        @param marketplace Shared resource management interface.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        Thread.__init__(self, **kwargs)
        self.p_id = marketplace.register_producer()
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        

    def run(self):
        """
        @brief Main industrial loop for the producer execution context.
        """
        while True:
            for product in self.products:
                product_type = product[0]
                quantity = product[1]
                time = product[2]

                i = 0
                # Block Logic: Quota fulfillment with backpressure handling.
                while i < quantity:
                    
                    published = self.marketplace.publish(self.p_id, product_type)
                    if published:
                        # Logic: Simulated industrial processing duration.
                        sleep(time)
                        i += 1
                    
                    else:
                        # Synchronization: Exponential wait during marketplace saturation.
                        sleep(self.republish_wait_time)
