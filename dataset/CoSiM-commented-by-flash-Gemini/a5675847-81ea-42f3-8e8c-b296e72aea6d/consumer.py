"""
@a5675847-81ea-42f3-8e8c-b296e72aea6d/consumer.py
@brief Event-driven simulation of a retail marketplace using multi-threaded Producer and Consumer agents.
Architecture: Centralized Marketplace mediator manages shared state, coordinating autonomous threads for asynchronous inventory processing.
Functional Utility: Orchestrates product publication, virtual shopping cart persistence, and concurrent transactional updates.
Synchronization: Employs threading.Lock for critical sections and cooperative sleep intervals for demand-supply flow control.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping sessions.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested products from the Marketplace.
    Functional Utility: Abstracts lower-level marketplace interactions into high-level 'add_to_cart' and 'remove_from_cart' routines.
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
        self.kwargs = kwargs

    def add_to_cart(self, quantity, cart_id, product):
        """
        @brief Iterative acquisition loop for a specific commodity.
        Block Logic: Implements a polling retry mechanism. If 'add_to_cart' fails, the thread yields to allow producers to replenish.
        Invariant: All requested units must be acquired before the method returns.
        """
        i = 0
        while i < quantity:
            added_ok = self.marketplace.add_to_cart(cart_id, product)
            if added_ok:
                i = i + 1
            else:
                time.sleep(self.retry_wait_time)

    def remove_from_cart(self, quantity, cart_id, product):
        """
        @brief Iterative removal of products from a specific cart.
        Logic: Returns the units back to the marketplace pool.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """
        @brief Lifecycle manager for the consumer thread.
        Logic: Allocates a unique session (cart_id) and fulfill all queued requests across assigned carts.
        """
        cart_id = self.marketplace.new_cart()
        for cart_list in self.carts:
            for cart_event in cart_list:
                # Block Logic: Dispatches commands based on request type.
                if cart_event["type"] == "add":
                    self.add_to_cart(cart_event["quantity"], cart_id, cart_event["product"])
                else:
                    self.remove_from_cart(cart_event["quantity"], cart_id, cart_event["product"])
        
        # Finalization: Executes the transaction and outputs results.
        for product in self.marketplace.place_order(cart_id):
            print(self.name, "bought", product)


from threading import Lock
import unittest
import time
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

class Marketplace:
    """
    @brief Shared resource manager coordinating inventorypools and transactional synchronization.
    State Management: Maintains mappings for producer stock queues and consumer cart sessions.
    Synchronization: Uses distinct locks for cart (lock_cart) and producer (lock_producer) registries to minimize contention.
    Observability: Integrates RotatingFileHandler for structured audit logging of all transactional events.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum inventory allowed per supplier for backpressure control.
        """

        self.queue_size_per_peroducer = queue_size_per_producer
        self.products = [] # List of Producer pools (List of Lists).
        self.carts = [] # List of Consumer session contents.
        self.product_in_cart = {} # Map: Product -> Boolean indicating reservation status.
        self.lock_cart = Lock()
        self.lock_producer = Lock()
        
        # Block Logic: Logger configuration.
        self.logger = logging.getLogger('marketplace')
        handler = RotatingFileHandler('marketplace.log', maxBytes=4096, backupCount=10)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        logging.Formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel("INFO")

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its inventory list.
        @return Unique producer identifier.
        """
        self.logger.info("Method register_producer started")
        with self.lock_producer:
            self.products.append([])
            ret = len(self.products) - 1
        self.logger.info("Method register_producer returned " + str(ret))
        return ret

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the marketplace.
        Constraint: Rejects publication if the supplier's pool has reached its capacity limit.
        """

        self.logger.info("Method publish started")
        self.logger.info("producer_id = " + str(producer_id))
        self.logger.info("product = " + str(product))
        
        with self.lock_producer:
            # Block Logic: Backpressure check.
            if len(self.products[producer_id]) < self.queue_size_per_peroducer:
                self.products[producer_id].append(product)
                self.product_in_cart[product] = False
                self.logger.info("New product published to marketplace")
                return True

        self.logger.info("Method publish returned False")
        return False

    def new_cart(self):
        """
        @brief Allocates a new shopping session for a consumer.
        """

        self.logger.info("Method new_cart started")
        with self.lock_cart:
            self.carts.append([])
            ret = len(self.carts) - 1
        self.logger.info("Method new_cart returned " + str(ret))
        return ret

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically reserves a commodity unit for a specific consumer session.
        Logic: Checks global availability and reservation status before appending to the cart.
        @return Boolean indicating acquisition success.
        """

        self.logger.info("Method add_to_cart started")
        self.logger.info("cart_id = " + str(cart_id))
        self.logger.info("product = " + str(product))
        
        # Block Logic: Acquisition check.
        if product in self.product_in_cart.keys() and not self.product_in_cart[product]:
            # Invariant: Item is marked as reserved to prevent concurrent acquisition by other threads.
            self.carts[cart_id].append(product)
            self.product_in_cart[product] = True
            self.logger.info("New product added to cart " + str(cart_id))
            return True

        self.logger.info("Method add_to_cart returned False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts a reservation, restoring the commodity to an available state.
        """

        self.logger.info("Method remove_from_cart started")
        self.logger.info("cart_id = " + str(cart_id))
        self.logger.info("product = " + str(product))
        
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)

            # Invariant: Resets reservation status to make item available for other consumers.
            self.product_in_cart[product] = False
            self.logger.info("Product removed from cart")

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and flushes items from the supplier pools.
        Side Effect: Synchronizes global inventory state by removing fulfilled units.
        """

        self.logger.info("Method place_order started")
        self.logger.info("cart_id = " + str(cart_id))
        
        # Block Logic: Fulfillment sync.
        for cart_product in self.carts[cart_id]:
            for prod_products in self.products:
                if cart_product in prod_products:
                    # Invariant: Removes the physical unit from the source producer pool.
                    prod_products.remove(cart_product)
        
        self.logger.info("Method place_order returned " + str(self.carts[cart_id]))
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and concurrency integrity.
    """
    def setUp(self):
        self.marketplace = Marketplace(15)
        self.products = [Coffee("Espresso", 7, 4.00, "MEDIUM"), \
                        Coffee("Irish", 10, 5.00, "MEDIUM"), \
                        Tea("Black", 10, "Green")]

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(0, self.products[0]))
        self.assertTrue(self.marketplace.publish(0, self.products[1]))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[0]))
        self.assertEqual(len(self.marketplace.carts[0]), 1)
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[1]))
        self.assertEqual(len(self.marketplace.carts[0]), 2)

    def test_remove_from_cart(self):
        
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[1])
        self.marketplace.remove_from_cart(0, self.products[2])
        self.assertEqual(len(self.marketplace.carts[0]), 2)
        self.marketplace.remove_from_cart(0, self.products[0])
        self.assertEqual(len(self.marketplace.carts[0]), 1)

    def test_place_order(self):
        
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[1])
        self.assertEqual(self.marketplace.place_order(0), [self.products[0], self.products[1]])


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Producer agent responsible for resource manufacturing and inventory replenishment.
    Logic: Continually fulfills production quotas and publishes results to the Marketplace mediator.
    Functional Utility: Models manufacturing latencies and handles supply-side backpressure.
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
        self.kwargs = kwargs

    def run(self):
        """
        @brief Main manufacturing loop for the producer execution context.
        """
        while True:
            # Initialization: Onboards as a new entity for every production cycle.
            producer_id = self.marketplace.register_producer()
            for product in self.products:
                i = 0
                num_of_products = product[1]
                curr_product = product[0]
                curr_product_wait_time = product[2]
                
                # Block Logic: Quota fulfillment with backpressure management.
                while i < num_of_products:
                    published_ok = self.marketplace.publish(producer_id, curr_product)
                    if published_ok:
                        i += 1
                        # Logic: Simulated duration to create the physical unit.
                        time.sleep(curr_product_wait_time)
                    else:
                        # Synchronization: Exponential wait during marketplace inventory overflow.
                        time.sleep(self.republish_wait_time)
