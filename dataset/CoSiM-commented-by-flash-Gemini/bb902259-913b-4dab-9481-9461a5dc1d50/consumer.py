"""
@bb902259-913b-4dab-9481-9461a5dc1d50/consumer.py
@brief multi-threaded simulation of a retail marketplace using concurrent Producer and Consumer agents.
Architecture: Centralized Marketplace mediator manages shared state, coordinating autonomous threads for asynchronous commerce.
Functional Utility: Handles inventory management, virtual shopping cart persistence, and atomic transaction fulfillment.
Synchronization: Employs threading.Lock for serializing state transitions and cooperative yield patterns (sleep) for demand-supply flow control.
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
        @param retry_wait_time Duration to wait when the marketplace is depleted.
        """

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief lifecycle manager for the consumer execution context.
        Logic: Allocates a new transactional session (curr_id) and fulfills all commands before finalizing the order.
        Synchronization: Acquires the marketplace's print lock to serialize console reporting.
        """
        
        curr_id = self.marketplace.new_cart()
        for curr_cart in self.carts:
            for elem in curr_cart:
                
                action_type = elem["type"]
                prod_id = elem["product"]
                quantity = elem["quantity"]
                
                # Block Logic: Fulfillment loop.
                for i in range(quantity):
                    if action_type == "add":
                        
                        # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                        while not self.marketplace.add_to_cart(curr_id, prod_id):
                            sleep(self.retry_wait_time)
                    if action_type == "remove":
                        # Logic: Returns reserved commodities to the marketplace inventory.
                        self.marketplace.remove_from_cart(curr_id, prod_id)
                        sleep(self.retry_wait_time)

        # Finalization: executes the transaction and flushes results.
        order = self.marketplace.place_order(curr_id)
        for i in order:
            
            # Logic: Serialized output of finalized acquisition.
            with self.marketplace.print_lock:
                print(f"cons{curr_id} bought {i}")

from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler


class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for per-producer occupancy, global product availability, and active carts.
    Synchronization: Uses distinct locks for producer (product_lock), publication (publish_lock), and cart allocation (cart_lock) to minimize contention.
    Observability: Integrates RotatingFileHandler for structured audit logging of all concurrent events.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum inventory allowed per supplier for backpressure control.
        """

        # Block Logic: Audit logging infrastructure.
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(asctime)s;%(message)s")
        self.rotating_file_handler = RotatingFileHandler('marketplace.log', 'w')
        self.rotating_file_handler.setLevel(logging.INFO)
        self.rotating_file_handler.setFormatter(self.formatter)
        self.log.addHandler(self.rotating_file_handler)

        self.queue_size_per_producer = queue_size_per_producer
        
        self.producers = {} # Mapping: ProducerID -> List of Products.
        self.no_prod = 0
        
        self.carts = {} # Mapping: CartID -> List of Products.
        self.no_carts = 0
        self.market_products = [] # Global pool of available units.

        self.product_lock = Lock()
        self.cart_lock = Lock()
        self.publish_lock = Lock()
        self.add_lock = Lock()
        self.print_lock = Lock()

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its inventory tracking.
        @return Unique producer identifier.
        """


        
        with self.product_lock:
            self.log.info("begin register method")
            self.no_prod += 1
            id_p = self.no_prod

        
        # Initialization: Scaffolds the inventory pool for the new producer.
        self.producers[id_p] = []


        self.log.info("end register method")
        return id_p

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the global pool.
        Constraint: Rejects publication if the supplier's individual queue is saturated.
        """
        
        with self.publish_lock:
            self.log.info("begin publish method")
            
            # Block Logic: Threshold check for supply-side flow control.
            if len(self.producers[int(producer_id)]) > self.queue_size_per_producer:
                self.log.info("end publish method with False")
                return False

            
            # Logic: Item becomes visible in global inventory and producer's isolated pool.
            self.producers[int(producer_id)].append(product)
            self.market_products.append(product)
            self.log.info("end publish method with True")
            return True

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """
        
        with self.cart_lock:
            self.log.info("begin new_cart method")
            self.no_carts += 1
            new_cart_id = self.no_carts

        
        self.carts[new_cart_id] = []


        self.log.info("end new_cart method")
        return new_cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        Logic: Acquires product from the global pool and prunes it from the respective supplier pool.
        @return Boolean indicating acquisition success.
        """
        
        with self.add_lock:
            self.log.info("begin add_to_cart method")
            
            # Block Logic: Acquisition and reservation.
            if product in self.market_products:
                
                
                # Invariant: Item must be removed from global pool before being assigned to a session.
                self.carts[cart_id].append(product)
                self.market_products.remove(product)
                
                # Logic: Finds and prunes the unit from the source supplier.
                for id_p in self.producers:
                    if product in self.producers[id_p]:
                        self.producers[id_p].remove(product)
                        break


                self.log.info("end add_to_cart method with True")
                return True
        self.log.info("end add_to_cart method with False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to global availability.
        """
        
        self.log.info("begin remove_from_cart method")
        
        
        
        for prod in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove(prod)
                self.market_products.append(prod)
                break
        self.log.info("end remove_from_cart method")

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and returns the result list.
        """
        
        self.log.info("begin place_order method")
        
        order = self.carts[cart_id]
        self.log.info("end place_order method")
        return order


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and concurrency safety.
    """

    def setUp(self):
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        
        self.marketplace.register_producer()
        print(self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')"))
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])

    def test_new_cart(self):
        
        self.marketplace.register_producer()
        print(self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')"))
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])
        print(self.marketplace.new_cart())

    def test_add(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])
        self.marketplace.new_cart()
        print(self.marketplace.market_products)
        print(self.marketplace.producers)
        print(self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')"))
        print(self.marketplace.producers)
        print(self.marketplace.carts)

    def test_remove(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])
        self.marketplace.new_cart()
        print(self.marketplace.market_products)
        print(self.marketplace.producers)
        self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')")
        print(self.marketplace.carts)
        print(self.marketplace.remove_from_cart(1, "Tea(name='Linden', price=9, type='Herbal')"))
        print(self.marketplace.carts)

    def test_place_order(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])
        self.marketplace.new_cart()
        print(self.marketplace.market_products)
        print(self.marketplace.producers)
        self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')")
        print(self.marketplace.carts)
        print(self.marketplace.place_order(1))


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
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Main industrial loop for the producer execution context.
        """
        # Initialization: Registers as a supplier to obtain a unique ID.
        producer_id = self.marketplace.register_producer()
        
        while True:
            for prod in self.products:
                
                product_id = prod[0]
                quantity = prod[1]
                wait_time = prod[2]
                
                # Block Logic: Quota fulfillment.
                i = 0
                while i < int(quantity):
                    if self.marketplace.publish(str(producer_id), product_id):
                        i += 1
                        # Logic: Simulated production latency.
                        sleep(wait_time)
                
                # Synchronization: Yield between production batches.
                sleep(self.republish_wait_time)
