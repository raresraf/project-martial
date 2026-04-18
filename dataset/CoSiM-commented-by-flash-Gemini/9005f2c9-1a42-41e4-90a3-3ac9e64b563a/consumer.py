"""
@9005f2c9-1a42-41e4-90a3-3ac9e64b563a/consumer.py
@brief Distributed simulation of a retail ecosystem using multi-threaded Producer and Consumer agents.
Architecture: Decoupled producer-consumer design coordinated by a centralized Marketplace object for state management.
Functional Utility: Manages inventory replenishment, session-persistent shopping carts, and concurrent transactional updates.
Synchronization: Employs threading.Lock for critical sections and a polling sleep mechanism for demand-supply flow control.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping lists.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested products from the Marketplace.
    Error Handling: Implements a spin-lock retry pattern when the requested inventory is out of stock.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Hierarchical list of shopping requests (operation type, product, quantity).
        @param marketplace Shared resource mediator.
        @param retry_wait_time Temporal duration to yield when inventory is depleted.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Lifecycle manager for the consumer thread.
        Logic: Allocates a unique session (cart_id) and fulfills all queued requests before finalizing the transaction.
        """
        for cart in self.carts:
            # Initialization: Establishes a session-scoped inventory buffer in the marketplace.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Multi-pass execution of shopping requests.
            for action in cart:

                for _ in range(action['quantity']):
                    if action['type'] == "add":
                        return_value = self.marketplace.add_to_cart(cart_id, action['product'])
                    else:
                        return_value = self.marketplace\
                            .remove_from_cart(cart_id, action['product'])
                    
                    # Logic: Initial yield to allow for concurrent state transitions.
                    time.sleep(self.retry_wait_time)

                    # Block Logic: Fulfillment barrier.
                    # Invariant: Must continue retrying until the requested state change (acquisition/return) is successful.
                    while return_value == False:
                        time.sleep(self.retry_wait_time)

                        if action['type'] == "add":
                            return_value = self.marketplace\
                                .add_to_cart(cart_id, action['product'])
                        else:
                            return_value = self.marketplace\
                                .remove_from_cart(cart_id, action['product'])

            # Finalization: Commits the transaction and releases the session resources.
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock


import unittest
import time
import logging
from logging.handlers import RotatingFileHandler

# Block Logic: Global logging infrastructure.
# Functional Utility: Persistent audit trail for multi-threaded transactions with rotation to manage disk space.
logging.basicConfig(
        handlers=[RotatingFileHandler('./marketplace.log', maxBytes=150000, backupCount=15)],
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = time.gmtime

class Producer:
    """
    @brief State container for a registered supplier entity.
    """
    def __init__(self, producer_id, nr_items):
        
        self.producer_id = producer_id 
        self.nr_items = nr_items # Current inventory occupancy count.

class Product:
    """
    @brief Internal representation of a commodity unit.
    """
    def __init__(self, details, producer_id, quantity):
        
        self.details = details # Product metadata or unique identifier.
        self.producer_id = producer_id # Reference to the originating supplier.
        self.quantity = quantity 

class Cart:
    """
    @brief Session-bound container for reserved products.
    """
    def __init__(self, cart_id, products):
        
        self.cart_id = cart_id 
        self.products = products 

def get_index_of_product(product, list_of_products):
    """
    @brief Search primitive for locating a product within a list of Product entities.
    """
    idx = 0
    for element in list_of_products:
        if product == element.details:
            return idx
        idx += 1
    return -1

class Marketplace:
    """
    @brief Centralized resource manager and concurrency coordinator.
    State Management: Tracks global producer capacity, aggregated product inventory, and active consumer sessions.
    Synchronization: Uses fine-grained locks (new_producer_lock, new_cart_lock, add_to_cart_lock, etc.) to minimize thread contention.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on inventory per supplier for flow control.
        """
        self.limit_per_producer = queue_size_per_producer  
        self.producers = [] 
        self.products = [] # Global product registry.
        self.carts = [] # Global cart registry.
        self.new_producer_lock = Lock()
        self.new_cart_lock = Lock()
        self.add_to_cart_lock = Lock()
        self.remove_from_cart_lock = Lock()
        self.publish_lock = Lock()
        self.pint_lock = Lock() # Serializes console output.



    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its inventory tracking.
        """
        with self.new_producer_lock:
            
            self.producers.append(Producer(len(self.producers), 0))
            logging.info(f'FROM "register_producer" ->'
                         f' output: producer_id = {len(self.producers) - 1}')
            return len(self.producers) - 1

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add inventory to the marketplace.
        Constraint: Rejects publication if the supplier's quota is exhausted (backpressure).
        """
        logging.info(f'FROM "publish" ->'
                     f' input: producer_id = {producer_id}, product = {product}')

        
        # Block Logic: Quota verification.
        if self.producers[producer_id].nr_items >= self.limit_per_producer:
            logging.info(f'FROM "publish" ->'
                         f' output: False')
            return False

        with self.publish_lock:
            
            self.producers[producer_id].nr_items += 1

            # Logic: Updates global product registry. Increments quantity if existing; otherwise adds new entry.
            idx_product = get_index_of_product(product, self.products)
            if idx_product == -1:
                self.products.append(Product(product, producer_id, 1))
            else:
                self.products[idx_product].quantity += 1
                self.products[idx_product].producer_id = producer_id

        logging.info(f'FROM "publish" ->'
                     f' output: True')
        return True


    def new_cart(self):
        """
        @brief Allocates a new shopping session for a consumer.
        """
        with self.new_cart_lock:
            
            self.carts.append(Cart(len(self.carts), []))
            logging.info(f'FROM "new_cart" ->'
                         f' output: cart_id: {len(self.carts) - 1}')
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        @return Boolean indicating if the product was available and acquired.
        """
        logging.info(f'FROM "add_to_cart" ->'
                     f' input: cart_id = {cart_id}, product = {product}')

        idx_product = get_index_of_product(product, self.products)
        
        # Condition: Failure if product is unknown or stock is zero.
        if idx_product == -1:
            logging.info(f'FROM "add_to_cart" ->'
                         f' output: False')
            return False
        
        if self.products[idx_product].quantity == 0:
            logging.info(f'FROM "add_to_cart" ->'
                         f' output: False')
            return False

        with self.add_to_cart_lock:
            
            # Invariant: Decrements occupancy for the specific source producer.
            idx_producer = self.products[idx_product].producer_id
            self.producers[idx_producer].nr_items -= 1

            
            self.products[idx_product].quantity -= 1

        
        self.carts[cart_id].products.append(product)

        logging.info(f'FROM "add_to_cart" ->'
                     f' output: True')
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the item to its originating producer's quota.
        """
        logging.info(f'FROM "remove_from_cart" ->'
                     f' input: cart_id = {cart_id}, product = {product}')

        
        self.carts[cart_id].products.remove(product)
        with self.remove_from_cart_lock:
            
            # Logic: Updates global availability and source producer occupancy.
            idx_product = get_index_of_product(product, self.products)
            self.products[idx_product].quantity += 1

            
            idx_producer = self.products[idx_product].producer_id
            self.producers[idx_producer].nr_items += 1

    def place_order(self, cart_id):
        """
        @brief Finalizes the shopping session and flushes results to standard output.
        """
        logging.info(f'FROM "place_order" ->'
                     f' input: cart_id = {cart_id}')

        # Synchronization: Serializes output to prevent interleaved log lines from concurrent consumers.
        self.pint_lock.acquire()
        for product in self.carts[cart_id].products:
            print(f'{currentThread().name} bought {product}')
        self.pint_lock.release()

        logging.info(f'FROM "place_order" ->'
                     f' input: cart_id = {self.carts[cart_id].products}')
        return self.carts[cart_id].products

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for validating Marketplace state transitions and concurrency safety.
    """
    def setUp(self):
        
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), len(self.marketplace.producers) - 1)
        
        self.assertEqual(len(self.marketplace.producers), 1)

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), len(self.marketplace.carts) - 1)
        
        self.assertEqual(len(self.marketplace.carts), 1)

    def test_publish_success(self):
        
        self.marketplace.register_producer()
        product = "product1"
        self.assertEqual(self.marketplace.publish(0, product), True)
        
        self.assertEqual(len(self.marketplace.products), 1)

    def test_publish_fail(self):
        
        self.marketplace.register_producer()
        self.marketplace.producers[0].nr_items = 5
        product = "product1"
        self.assertEqual(self.marketplace.publish(0, product), False)

    def test_add_to_cart_success(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.assertEqual(self.marketplace.add_to_cart(0, product), True)

    def test_add_to_cart_fail_case1(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        nonexistent_product = "nonexistent_product"
        self.assertEqual(self.marketplace.add_to_cart(0, nonexistent_product), False)

    def test_add_to_cart_fail_case2(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.marketplace.products[0].quantity = 0
        self.assertEqual(self.marketplace.add_to_cart(0, product), False)

    def test_remove_from_cart(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.marketplace.add_to_cart(0, product)
        self.marketplace.remove_from_cart(0, product)
        self.assertEqual(self.marketplace.carts[0].products, [])

    def test_place_order(self):
        
        self.marketplace.new_cart()
        products_sample = ["prod1", "prod2", "prod3"]
        self.marketplace.carts[0].products = products_sample
        self.assertEqual(self.marketplace.place_order(0), products_sample)


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Producer agent responsible for resource manufacturing and inventory replenishment.
    Logic: Continuously fulfills production quotas and publishes to the Marketplace mediator.
    Functional Utility: Models the supply chain with simulated processing delays.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) production metrics.
        @param marketplace Shared resource mediation system.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Main manufacturing loop for the producer thread.
        """
        # Initialization: Onboards as a supplier for the lifetime of the thread.
        id_producer = self.marketplace.register_producer()
        while True:
            for i in range(len(self.products)):
                product_id = self.products[i][0]
                how_many = self.products[i][1]
                wait_time = self.products[i][2]
                for _ in range(how_many):
                    # Block Logic: Publish-retry loop for supply-side flow control.
                    while True:
                        return_value = self.marketplace.publish(id_producer, product_id)
                        if return_value:
                            # Logic: Simulated production latency.
                            time.sleep(wait_time)
                            break
                        # Synchronization: Yield execution when the marketplace queue is full.
                        time.sleep(self.republish_wait_time)
