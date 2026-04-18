"""
@8797c424-93cc-4517-b4f2-013767b1daad/consumer.py
@brief Distributed commerce simulation using concurrent Producer and Consumer agents via a centralized Marketplace.
Architecture: Multi-threaded actor-based model where Marketplace acts as a thread-safe state manager.
Functional Utility: Manages asynchronous inventory replenishment, transactional shopping carts, and serialized console reporting.
Synchronization: Employs threading.Lock for critical sections and event-driven waiting patterns (retry_wait_time) for inventory flow control.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing high-level shopping workflows.
    Logic: Sequentially processes assigned carts, interacting with the Marketplace to acquire or return commodities.
    Error Handling: Implements a polling retry mechanism when requested inventory is unavailable.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Hierarchical list of shopping requests (product, quantity, operation).
        @param marketplace Shared resource management interface.
        @param retry_wait_time Temporal duration to yield when inventory is depleted.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs["name"]

    def run(self):
        """
        @brief Main execution lifecycle for the consumer thread.
        Logic: Authenticates a new session (cart_id) and executes all internal requests before finalization.
        """
        for cart in self.carts:

            # Logic: Initializes a session-scoped inventory buffer in the marketplace.
            cart_id = self.marketplace.new_cart()

            for data in cart:
                i = 0
                item = data["product"]
                operation = data["type"]

                # Block Logic: Fulfillment loop for a specific product request.
                while i < data["quantity"]:

                    if operation == "add":
                        # Logic: Attempts to atomically reserve the product.
                        available = self.marketplace.add_to_cart(cart_id, item)
                        if available:
                            i += 1
                        else:
                            # Synchronization: Yields execution to allow producers to replenish.
                            time.sleep(self.retry_wait_time)

                    if operation == "remove":
                        # Logic: Returns a reserved product to the global inventory pool.
                        self.marketplace.remove_from_cart(cart_id, item)
                        i += 1

            # Finalization: Commits the transaction and retrieves the acquisition list.
            order = self.marketplace.place_order(cart_id)

            # Logic: Outputs the results using the marketplace's thread-safe printing primitive.
            self.marketplace.print_list(order, self.consumer_name)

from logging.handlers import RotatingFileHandler
import logging
import time
import os

# Block Logic: Persistent logging infrastructure setup.
if not os.path.exists("Logs"):
    os.makedirs("Logs")

def get_log(name):
    """
    @brief Configures a rotating file handler for audit logging.
    Functional Utility: Ensures diagnostic data is preserved while limiting disk consumption.
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    
    logging.Formatter.converter = time.gmtime
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    
    handler = RotatingFileHandler('Logs/marketplace.log', maxBytes=2000, backupCount=20)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)

    return logger

from threading import Lock
import unittest
import io
import sys
sys.path.append("tema")
from logger import get_log

LOGGER = get_log('marketplace.log')

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and boundary conditions.
    """
    def setUp(self):
        
        self.marketplace = Marketplace(15)
        self.product_1 = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        self.product_2 = {
            "product_type": "Tea",
            "name": "Bubble Tea",
            "price": 10
        }
        self.producer_id = self.marketplace.register_producer()
        self.cart_id = self.marketplace.new_cart()
        self.cart_id_2 = self.marketplace.new_cart()

    def test_register_producer(self):
        
        print("\nTesting register_producer")
        self.assertIsNotNone(self.producer_id)
        self.assertEqual(self.producer_id, 0)
        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 0)
        self.assertEqual(self.marketplace.items[self.producer_id], [])

    def test_publish(self):
        
        print("\nTesting publish")
        self.assertEqual(self.marketplace.publish(self.producer_id, self.product_1), True)
        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 1)
        self.assertEqual(self.marketplace.items[self.producer_id], [self.product_1])
        self.marketplace.prod_num_items[self.producer_id] = 1000
        self.assertEqual(self.marketplace.publish(self.producer_id, self.product_1), False)

    def test_new_cart(self):
        
        print("\nTesting new_cart")
        self.assertIsNotNone(self.cart_id)
        self.assertEqual(self.cart_id, 0)
        self.assertEqual(self.marketplace.carts[self.cart_id], [])
        self.assertIsNotNone(self.cart_id_2)
        self.assertEqual(self.cart_id_2, 1)
        self.assertEqual(self.marketplace.carts[self.cart_id_2], [])

    def test_add_to_cart(self):
        
        print("\nTesting add_to_cart")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.assertEqual(self.marketplace.add_to_cart(self.cart_id, self.product_1), True)

        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 0)
        self.assertEqual(self.marketplace.items[self.producer_id], [])

        self.assertEqual(self.marketplace.carts[self.cart_id], [(self.product_1, self.producer_id)])
        self.assertEqual(self.marketplace.add_to_cart(self.cart_id, self.product_2), False)

    def test_remove_from_cart(self):
        
        print("\nTesting remove_from_cart")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        self.marketplace.remove_from_cart(self.cart_id, self.product_1)

        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 1)
        self.assertEqual(self.marketplace.items[self.producer_id], [self.product_1])
        self.assertEqual(self.marketplace.carts[self.cart_id], [])

    def test_place_order(self):
        
        print("\nTesting place_order")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.publish(self.producer_id, self.product_2)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id_2, self.product_2)
        order_1 = []
        order_2 = []

        order_1 = self.marketplace.place_order(self.cart_id)
        self.assertEqual(order_1, [(self.product_1, self.producer_id)])
        self.assertEqual(self.marketplace.carts,
                         {self.cart_id_2: [(self.product_2, self.producer_id)]})

        order_2 = self.marketplace.place_order(self.cart_id_2)
        self.assertEqual(order_2, [(self.product_2, self.producer_id)])
        self.assertEqual(self.marketplace.carts, {})

        self.assertIsNotNone(order_1)
        self.assertIsNotNone(order_2)
        self.assertNotEqual(order_1, {})
        self.assertNotEqual(order_2, {})

    def test_print_list(self):
        
        cons_name = "Consumer 1"
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        order = self.marketplace.place_order(self.cart_id)

        output = io.StringIO()
        sys.stdout = output
        self.marketplace.print_list(order, cons_name)
        sys.stdout = sys.__stdout__
        self.assertEqual(output.getvalue(),
                         'Consumer 1 bought {\'product_type\': \'Coffee\','
                         '\'name\': \'Indonezia\', \'acidity\': 5.05,'
                         '\'roast_level\': \'MEDIUM\','
                         '\'price\': 1}\n')

class Marketplace:
    """
    @brief Shared resource mediator managing inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer inventory, active consumer carts, and occupancy metrics.
    Synchronization: Employs fine-grained locks (register_lock, new_cart_lock, cart_lock, print_lock) to minimize thread contention.
    """

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on inventory per producer for backpressure control.
        """
        LOGGER.info('creating an instance of Marketplace')
        LOGGER.info('Max size of queue in Marketplace: %d', queue_size_per_producer)



        self.queue_size_per_producer = queue_size_per_producer

        self.num_prod = 0
        self.num_carts = 0
        
        self.prod_num_items = [] # Tracks unit counts per producer.
        
        self.items = {} # Mapping: ProducerID -> List of available Products.
        
        self.carts = {} # Mapping: CartID -> List of (Product, SourceProducerID) tuples.

        self.register_lock = Lock()
        self.new_cart_lock = Lock()
        self.cart_lock = Lock()
        
        self.print_lock = Lock()



    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes their inventory pool.
        """
        LOGGER.info("In method 'register_producer' from class Marketplace")


        with self.register_lock:
            prod_id = self.num_prod
            self.num_prod += 1

        self.prod_num_items.append(0)
        self.items[prod_id] = []
        LOGGER.info("Output of 'register_producer' - producer id: %d", prod_id)
        return prod_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the marketplace.
        Constraint: Rejects the operation if the producer's individual queue is saturated.
        """
        LOGGER.info("In method 'publish' from class Marketplace\
                    \nInputs: producer_id =%s; product=%s",
                    producer_id, product)
        
        # Block Logic: Threshold check for supply-side flow control.
        if self.prod_num_items[producer_id] >= self.queue_size_per_producer:
            LOGGER.info("Output of 'publish' - %s", "False")
            return False
        
        self.items[producer_id].append(product)
        self.prod_num_items[producer_id] += 1
        LOGGER.info("Output of 'publish' - %s", "True")
        return True

    def new_cart(self):
        """
        @brief Allocates a new shopping session for a consumer.
        @return Unique cart identifier.
        """
        LOGGER.info("In method 'new_cart' from class Marketplace")
        with self.new_cart_lock:
            cart_id = self.num_carts
            self.num_carts += 1

        self.carts[cart_id] = []

        LOGGER.info("Output of 'new_cart' - cart_id = %s", cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically moves a product from a producer pool to a consumer cart.
        Logic: Performs an exhaustive search across all registered producers. First-available strategy.
        @return Boolean indicating acquisition success.
        """
        LOGGER.info("In method 'add_to_cart' from class Marketplace\nInputs:\
        cart_id =%s; product=%s", cart_id, product)
        found = False
        with self.cart_lock:
            # Block Logic: Inventory traversal.
            for i, (_, val) in enumerate(self.items.items()):
                if product in val:

                    # Invariant: Item must be removed from global pool and its source ID cached for potential returns.
                    val.remove(product)
                    self.prod_num_items[i] -= 1
                    prod_id = i
                    found = True
                    break

        if found:
            self.carts[cart_id].append((product, prod_id))

        LOGGER.info("Output of 'add_to_cart' - %s", found)
        return found

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the item to its original source producer.
        """
        LOGGER.info("In method 'remove_from_cart' from class Marketplace\nInputs:\
        cart_id =%s; product=%s", cart_id, product)


        for item, producer in self.carts[cart_id]:
            if item is product:
                prod_id = producer
                self.carts[cart_id].remove((item, producer))
                break

        self.items[prod_id].append(product)

        with self.cart_lock:
            self.prod_num_items[prod_id] += 1
        LOGGER.info("Finished 'remove_from_cart', no return value")

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and flushes the cart contents.
        """
        LOGGER.info("In method 'place_order' from class Marketplace\
        \nInputs:cart_id =%s", cart_id)
        res = self.carts.pop(cart_id)
        LOGGER.info("Output of 'place_order' - res = %s", res)
        return res

    def print_list(self, order, consumer_name):
        """
        @brief Serializes acquisition results to standard output.
        Synchronization: Uses print_lock to prevent interleaved text from multiple threads.
        """
        LOGGER.info("In method 'print_list' from class Marketplace\
        \nInputs:order =%s; consumer_name: %s", order, consumer_name)
        for item in order:
            with self.print_lock:
                print(consumer_name + " bought "+ str(item[0]))
        LOGGER.info("Finished 'print_list', no return value")


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continuously fulfills production quotas and publishes to the Marketplace.
    Functional Utility: Models the supply chain with configurable production latencies.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) metrics.
        @param marketplace Shared resource mediator.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        Thread.__init__(self, **kwargs)
        self.products = products


        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Main manufacturing loop.
        """
        prod_id = self.marketplace.register_producer()
        while True:
            for (item, quantity, wait_time) in self.products:
                i = 0
                while i < quantity:
                    # Block Logic: Resource publication with backpressure handling.
                    available = self.marketplace.publish(prod_id, item)

                    if available:
                        # Logic: Simulated industrial processing duration.
                        time.sleep(wait_time)
                        i += 1
                    else:
                        # Synchronization: Polling wait during marketplace saturation.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base representation of a commodity.
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
