"""
@d0ec8f5d-4b4b-4ef9-b02b-06d59ebb1f82/consumer.py
@brief Event-driven simulation of a retail marketplace using multi-threaded Producer and Consumer agents.
Architecture: Centralized Marketplace mediator manages shared state, coordinating autonomous threads for asynchronous commerce.
Functional Utility: Handles inventory management, virtual shopping cart persistence, and atomic transaction fulfillment.
Synchronization: Employs threading.Lock for serializing state transitions and cooperative yield patterns (sleep) for demand-supply flow control.
"""

import threading
import time

class Consumer(threading.Thread):
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
        self.carts = carts
        self.marketplace = marketplace
        # Initialization: Establishes a session-scoped inventory buffer in the marketplace.
        self.consumert_cart_id = self.marketplace.new_cart()
        self.retry_wait_time = retry_wait_time
        threading.Thread.__init__(self, **kwargs)


    def run(self):
        """
        @brief lifecycle manager for the consumer thread.
        Logic: Processes all assigned shopping lists, fulfilling each requested item before finalizing the transaction.
        """
        if self.marketplace is None:
            return

        for cart_entry in self.carts:
            for elem in cart_entry:
                # Block Logic: Fulfillment loop.
                # Invariant: Must continue retrying until the requested quantity is successfully acquired or returned.
                while elem['quantity'] > 0:
                    if elem['type'] == 'add':
                        valid_op = self.marketplace.add_to_cart(
                                                        self.consumert_cart_id,
                                                        elem['product'])
                    else:
                        valid_op = self.marketplace.remove_from_cart(
                                                        self.consumert_cart_id,
                                                        elem['product'])

                    if not valid_op:
                        # Synchronization: Yield execution to allow producers to replenish the global pool.
                        time.sleep(self.retry_wait_time)
                    else:
                        elem['quantity'] = elem['quantity'] - 1

            # Finalization: executes the transaction and flushes results.
            products = self.marketplace.place_order(self.consumert_cart_id)
            for product_types in products:
                for product in product_types:
                    # Logic: Serialized output of finalized acquisition.
                    print(f'{str(threading.currentThread().getName())} bought {str(product)}')


import collections
import json
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import unittest

class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer stock quotas, global product availability, and active carts.
    Synchronization: Uses distinct locks (register_producer_lock, new_cart_lock, add_to_cart_lock) to ensure atomicity.
    Observability: Integrates RotatingFileHandler for structured audit logging of all concurrent events.
    """
    
    logging.Formatter.converter = time.gmtime
    # Block Logic: Global audit logging configuration.
    # Functional Utility: Persistent log with rotation to prevent disk exhaustion.
    logging.basicConfig(
        handlers=[RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=5)],
        format='%(asctime)s - %(message)s',
        level=logging.INFO)

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on inventory per supplier for backpressure control.
        """
        self.max_products_allowed = queue_size_per_producer

        self.ticket_nr = 0
        self.products_nr = [] # Tracks unit count per producer pool.

        self.carts_nr = 0
        self.products = collections.defaultdict(list) # Global Registry: Product -> List of ProducerIDs.
        self.cart_products = {} # Mapping: CartID -> {Product -> List of ProducerIDs}.

        self.register_producer_lock = threading.Lock()
        self.new_cart_lock = threading.Lock()
        self.add_to_cart_lock = threading.Lock()

        logging.info('Started Marketplace process.')

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its inventory tracking.
        @return Unique producer identifier.
        """
        self.register_producer_lock.acquire()

        # Initialization: Scaffolds the occupancy metrics for the new producer.
        self.products_nr.append(0)
        self.ticket_nr = self.ticket_nr + 1

        self.register_producer_lock.release()

        logging.info('Registered producer with ID %s.', self.ticket_nr - 1)
        return self.ticket_nr - 1

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the global pool.
        Constraint: Rejects publication if the supplier's individual queue is saturated (backpressure).
        """

        # Block Logic: Threshold check for supply-side flow control.
        if self.products_nr[producer_id] >= self.max_products_allowed:
            logging.info('Producer %s published too many products.', producer_id)
            return False

        # Invariant: Updates both global product registry and per-producer occupancy.
        self.products[product].append(producer_id)
        self.products_nr[producer_id] += 1
        logging.info('Producer %s published product %s.', producer_id, product)
        return True

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """
        
        self.new_cart_lock.acquire()
        self.carts_nr = self.carts_nr + 1
        self.new_cart_lock.release()

        logging.info('Created new cart with ID %s.', self.carts_nr - 1)
        return self.carts_nr - 1

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        Logic: Acquires product from the global registry and caches the source producer ID for potential returns.
        @return Boolean indicating acquisition success.
        """
        
        # Block Logic: Acquisition check.
        if product not in self.products:
            logging.info('Product %s does not exist on marketplace.', product)
            return False
        if len(self.products[product]) <= 0:
            logging.info('Product %s does not exist on marketplace.', product)
            return False

        
        self.add_to_cart_lock.acquire()

        # Logic: Pulls the first available unit from the registry.
        producer_picked = self.products[product][0]
        self.products_nr[producer_picked] -= 1

        if cart_id not in self.cart_products:
            # Initialization: establishes session storage on first acquisition.
            self.cart_products[cart_id] = {}

        
        # Logic: maps the unit to the specific consumer session.
        if product in self.cart_products[cart_id]:
            self.cart_products[cart_id][product].append(producer_picked)
        else:
            self.cart_products[cart_id][product] = [producer_picked]
        
        # Invariant: Item must be removed from global pool.
        self.products[product].pop(0)

        self.add_to_cart_lock.release()

        logging.info('Product %s was added to cart %s.', product, cart_id)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to its originating producer's pool.
        """

        # Block Logic: Validation of session and product existence in cart.
        if cart_id > self.carts_nr:
            logging.info('Cart %s does not exist.', cart_id)
            return False
        if cart_id not in self.cart_products:
            logging.info('Cart %s does not exist.', cart_id)
            return False
        if product not in self.cart_products[cart_id]:
            logging.info('Product %s does not exist in cart %s.', product, cart_id)
            return False
        if len(self.cart_products[cart_id][product]) <= 0:
            logging.info('Product %s does not exist in cart %s.', product, cart_id)
            return False

        
        self.add_to_cart_lock.acquire()

        # Logic: identifies original producer from cached metadata.
        removed_product = self.cart_products[cart_id][product][0]
        # State Sync: Restores unit to global availability and supplier pool.
        self.products[product].append(removed_product)
        self.products_nr[removed_product] += 1

        
        # Finalization: Prunes unit from the session cart.
        self.cart_products[cart_id][product].pop(0)

        self.add_to_cart_lock.release()

        return True

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and returns grouped results.
        """
        if cart_id not in self.cart_products:
            return []

        ans = []
        # Block Logic: Finalization loop.
        for product in self.cart_products[cart_id]:
            product_nr = len(self.cart_products[cart_id][product])
            products_repeat = []
            for _ in range(product_nr):
                products_repeat.append(product)
            ans.append(products_repeat)

        # Finalization: Resets the session state.
        self.cart_products[cart_id] = {}
        return ans

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace state transitions and transactional integrity.
    """

    max_queue = 3
    def setUp(self):
        self.marketplace = Marketplace(self.max_queue)

    def test_register_producer(self):
        
        producer_id = self.marketplace.register_producer()
        self.assertEqual(producer_id, 0)

        producer_id = self.marketplace.register_producer()
        self.assertEqual(producer_id, 1)

        producer_id = self.marketplace.register_producer()
        self.assertEqual(producer_id, 2)

    def test_publish(self):
        
        product_sample = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        producer_id = self.marketplace.register_producer()

        
        self.assertTrue(self.marketplace.publish(producer_id,
                                                 json.dumps(product_sample)))
        self.assertTrue(self.marketplace.publish(producer_id,
                                                 json.dumps(product_sample)))
        self.assertTrue(self.marketplace.publish(producer_id,
                                                 json.dumps(product_sample)))

        
        self.assertFalse(self.marketplace.publish(producer_id, json.dumps(product_sample)))

    def test_new_cart(self):
        
        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, 0)

        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, 1)

        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, 2)

    def test_add_to_cart(self):
        
        product_sample = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        
        self.assertFalse(self.marketplace.add_to_cart(cart_id, json.dumps(product_sample)))

        
        self.marketplace.publish(producer_id, json.dumps(product_sample))
        self.assertTrue(self.marketplace.add_to_cart(cart_id, json.dumps(product_sample)))

        
        self.assertFalse(self.marketplace.add_to_cart(cart_id, json.dumps(product_sample)))

    def test_remove_from_cart(self):
        
        product_sample = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, json.dumps(product_sample))

        
        self.assertFalse(self.marketplace.remove_from_cart(cart_id, json.dumps(product_sample)))

        
        self.marketplace.add_to_cart(cart_id, json.dumps(product_sample))
        self.assertTrue(self.marketplace.remove_from_cart(cart_id, json.dumps(product_sample)))

        
        self.assertFalse(self.marketplace.remove_from_cart(cart_id, json.dumps(product_sample)))

    def test_place_order(self):
        
        product_sample = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, json.dumps(product_sample))
        self.marketplace.publish(producer_id, json.dumps(product_sample))
        self.marketplace.publish(producer_id, json.dumps(product_sample))

        
        self.assertEqual(self.marketplace.place_order(cart_id), [])

        
        self.marketplace.add_to_cart(cart_id, json.dumps(product_sample))
        self.assertEqual(self.marketplace.place_order(cart_id), [[json.dumps(product_sample)]])

        
        self.marketplace.add_to_cart(cart_id, json.dumps(product_sample))
        self.marketplace.add_to_cart(cart_id, json.dumps(product_sample))

        arr_product = self.marketplace.place_order(cart_id)[0]
        self.assertEqual(arr_product[0], json.dumps(product_sample))
        self.assertEqual(arr_product[1], json.dumps(product_sample))


import threading
import time

class Producer(threading.Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continually fulfills production quotas and publishes results to the Marketplace mediator.
    Functional Utility: Models manufacturing latencies and handles supply-side flow control.
    """
    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) production metrics.
        @param marketplace Shared resource mediator.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        threading.Thread.__init__(self, **kwargs)
        # Initialization: Registers as a supplier to obtain a persistent ID.
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        @brief Main manufacturing loop for the producer execution context.
        """
        while True:
            for (typep, remaning_quantity, timep) in self.products:
                # Block Logic: Quota fulfillment.
                while remaning_quantity > 0:
                    # Synchronization: Publish-retry loop for backpressure management.
                    if not self.marketplace.publish(self.producer_id, typep):
                        time.sleep(self.republish_wait_time)
                    else:
                        # Logic: Simulated industrial processing duration.
                        time.sleep(timep)
                        remaning_quantity = remaning_quantity - 1
