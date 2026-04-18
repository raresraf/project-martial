"""
@a82dcc15-9d90-4fd4-8216-139e53cecad0/consumer.py
@brief multi-threaded simulation of a retail marketplace using concurrent Producer and Consumer agents via a centralized Marketplace.
Architecture: Decoupled producer-consumer design where a thread-safe Marketplace mediator manages global state (inventory, carts).
Functional Utility: Facilitates asynchronous inventory replenishment, virtual shopping cart persistence, and serialized transaction reporting.
Synchronization: Employs threading.Lock for critical sections and cooperative yield patterns (sleep) for demand-supply flow control.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping sessions.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested products from the Marketplace.
    Error Handling: Implements a polling retry mechanism with configurable delays when requested inventory is out of stock.
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
        self.kwargs = kwargs

    def run(self):
        """
        @brief lifecycle manager for the consumer thread.
        Logic: Allocates a unique session (cart_id) and fulfills all commands before finalizing the order.
        Synchronization: Acquires the marketplace's consumer lock to serialize console reporting.
        """
        for cart in self.carts:
            # Initialization: Establishes a session-scoped inventory buffer in the marketplace.
            cart_id = self.marketplace.new_cart()
            for ops in cart:
                # Block Logic: Dispatcher for marketplace operations.
                if ops['type'] == "add":
                    for _ in range(0, ops['quantity']):
                        # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                        while self.marketplace.add_to_cart(cart_id, ops['product']) is not True:
                            sleep(self.retry_wait_time)
                else:
                    # Logic: Returns reserved commodities to the marketplace inventory.
                    for _ in range(0, ops['quantity']):
                        self.marketplace.remove_from_cart(cart_id, ops['product'])
            
            # Finalization: executes the transaction and flushes results.
            products = self.marketplace.place_order(cart_id)

            lock = self.marketplace.get_consumer_lock()

            # Logic: Serialized output of finalized acquisition.
            lock.acquire()
            for product in products:
                print(self.kwargs['name'] + " bought " + str(product))
            lock.release()


import logging
from logging.handlers import RotatingFileHandler
import time
from threading import Lock
import unittest

# Block Logic: Audit logging infrastructure.
# Functional Utility: Persistent log with rotation to prevent disk exhaustion.
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO,
                    handlers=[RotatingFileHandler('marketplace.log',
                                                  maxBytes=20000, backupCount=10)])
logging.Formatter.converter = time.gmtime

class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for per-producer occupancy, global product availability, and active carts.
    Synchronization: Uses distinct locks for producer (producer_lock) and consumer (consumer_lock) registries to minimize contention.
    """

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on inventory per supplier for backpressure management.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_lock = Lock() # Protects global producer registry and occupancy counts.
        self.consumer_lock = Lock() # Protects global cart registry and product allocation.
        self.producer_id = -1
        self.cart_id = -1
        self.size_per_producer = {} # Mapping: ProducerID -> Current stock count.
        self.carts = {} # Mapping: CartID -> {Product -> [SourceProducerID, ...]}
        self.products_dict = {} # Global Registry: Product -> [SourceProducerID, ...] available for purchase.

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its inventory tracking.
        @return Unique producer identifier.
        """
        
        self.producer_lock.acquire()
        logging.info("New producer entered register_producer method")
        self.producer_id += 1
        # Initialization: Scaffolds the occupancy metrics for the new producer.
        self.size_per_producer[self.producer_id] = 0

        self.producer_lock.release()
        logging.info("New producer registered with id %d", self.producer_id)
        return self.producer_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the global pool.
        Constraint: Rejects publication if the supplier's individual queue is saturated.
        """
        logging.info("Producer with id %d entered publish method", producer_id)

        self.producer_lock.acquire()
        
        # Block Logic: Threshold check for supply-side flow control.
        if self.size_per_producer[producer_id] == self.queue_size_per_producer:
            logging.info(f"Producer with id {producer_id} failed to publish product {product}")
            self.producer_lock.release()
            return False
        
        # Invariant: Updates both global product registry and per-producer occupancy.
        if product not in self.products_dict:
            self.products_dict[product] = [producer_id]
        else:
            self.products_dict[product].append(producer_id)

        self.size_per_producer[producer_id] += 1
        logging.info(f"Producer with id {producer_id} published product {product}")
        self.producer_lock.release()
        return True

    def new_cart(self):
        """
        @brief Allocates a new shopping session for a consumer.
        """

        self.consumer_lock.acquire()
        logging.info("Consumer entered new_cart method")
        self.cart_id += 1
        self.carts[self.cart_id] = {}
        logging.info("Consumer registered new cart with id %d", self.cart_id)
        self.consumer_lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        Logic: Acquires product from the global registry and caches the source producer ID for potential returns.
        @return Boolean indicating acquisition success.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered add_to_cart method", cart_id)
        
        # Block Logic: Acquisition and reservation.
        if product in self.products_dict:
            
            # Logic: Pulls the first available unit from the registry.
            producer_id = self.products_dict[product].pop(0)
            if product in self.carts[cart_id]:
                self.carts[cart_id][product].append(producer_id)
            else:
                self.carts[cart_id][product] = [producer_id]
            
            # Finalization: Prunes product entries when stock is fully depleted.
            if len(self.products_dict[product]) == 0:
                del self.products_dict[product]

            logging.info(f"Consumer with card id {cart_id} added product {product} to cart")
            self.consumer_lock.release()
            return True


        logging.info(f"Consumer with card id {cart_id} failed to add product {product} to cart")
        self.consumer_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to its originating producer's pool.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered remove_from_cart method", cart_id)

        # Logic: Identifies the original producer from the cart's cached metadata.
        given_id = self.carts[cart_id][product].pop(0)
        if len(self.carts[cart_id][product]) == 0:
            del self.carts[cart_id][product]

        # State Sync: Restores the physical unit to the global registry.
        if product not in self.products_dict:
            self.products_dict[product] = [given_id]
        else:
            self.products_dict[product].append(given_id)
        logging.info(f"Consumer with card id {cart_id} removed product {product} from cart")
        self.consumer_lock.release()

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and updates producer occupancy metrics.
        Side Effect: Synchronizes global utilization by decrementing occupancy for each fulfilled item.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered place_order method", cart_id)
        
        products = []
        # Block Logic: Finalization sweep.
        for product in self.carts[cart_id]:
            for given_id in self.carts[cart_id][product]:
                # Invariant: Occupancy must be decremented at final sale to allow producers to replenish.
                self.size_per_producer[given_id] -= 1
                products.append(product)
        logging.info("Consumer with card id %d placed order", cart_id)
        self.consumer_lock.release()
        return products

    def get_consumer_lock(self):
        """
        @brief Accessor for the consumer-side mutex to allow external synchronization of shared resources (stdout).
        """
        logging.info("A consumer entered get_consumer_lock method")
        return self.consumer_lock


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace state transitions and transactional safety.
    """
    
    def setUp(self):
        
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 0)

    def test_true_publish(self):
        
        producer_id = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(producer_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 1)
        self.assertEqual(len(self.marketplace.products_dict["id1"]), 1)

    def test_false_publish(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.assertFalse(self.marketplace.publish(producer_id, "id1"))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)

    def test_true_add_to_cart(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        cart_id = self.marketplace.new_cart()

        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 0)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)
        self.assertEqual(len(self.marketplace.carts[cart_id]["id1"]), 1)

        self.marketplace.publish(producer_id, "id1")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))
        self.assertEqual(len(self.marketplace.carts[cart_id]["id1"]), 2)

    def test_false_add_to_cart(self):
        
        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, "id1"))

    def test_remove_from_cart(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id2")

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")
        self.marketplace.add_to_cart(cart_id, "id2")
        self.assertEqual(len(self.marketplace.products_dict), 1)

        self.marketplace.remove_from_cart(cart_id, "id1")
        self.assertEqual(len(self.marketplace.products_dict), 2)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)
        self.assertFalse("id1" in self.marketplace.carts[cart_id])

        self.marketplace.remove_from_cart(cart_id, "id2")
        self.assertEqual(len(self.marketplace.products_dict), 2)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 0)
        self.assertFalse("id2" in self.marketplace.carts[cart_id])

    def test_place_order(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.assertEqual(self.marketplace.size_per_producer[producer_id], 2)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")

        expected_products = ["id1"]
        products = self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.size_per_producer[producer_id], 1)
        self.assertEqual(expected_products, products)

    def test_get_consumer_lock(self):
        


        self.assertEqual(self.marketplace.consumer_lock, self.marketplace.get_consumer_lock())


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continually fulfills production quotas and publishes results to the Marketplace.
    Functional Utility: Models manufacturing latencies and handles supply-side flow control.
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
        @brief Main manufacturing loop for the producer execution context.
        """
        # Initialization: Registers as a supplier to obtain a persistent ID.
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                for _ in range(0, product[1]):
                    
                    # Block Logic: Publish-retry loop for backpressure management.
                    while self.marketplace.publish(producer_id, product[0]) is not True:
                        sleep(self.republish_wait_time)
                    # Logic: Simulated production duration.
                    sleep(product[2])
