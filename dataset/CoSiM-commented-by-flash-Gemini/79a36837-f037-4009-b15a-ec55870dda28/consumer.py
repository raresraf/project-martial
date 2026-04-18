"""
@79a36837-f037-4009-b15a-ec55870dda28/consumer.py
@brief Distributed simulation of an e-commerce platform using multi-threaded Producer-Consumer agents.
Architecture: Decoupled design where a centralized Marketplace mediator coordinates interactions between autonomous Producers and Consumers.
Functional Utility: Manages concurrent access to inventory, virtual shopping carts, and order fulfillment workflows.
Synchronization: Employs threading.Lock for critical state transitions and cooperative yield patterns (sleep) for inventory flow control.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping strategies.
    Logic: Iterates through multiple shopping lists (carts), fulfilling product requests via the Marketplace interface.
    Fault Tolerance: Implements a spin-lock retry pattern when attempting to acquire out-of-stock items.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Hierarchical list of product acquisition requests.
        @param marketplace Shared resource management system.
        @param retry_wait_time Temporal duration to wait when acquisition fails.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        pass

    def run(self):
        """
        @brief Lifecycle manager for the consumer execution context.
        Logic: Allocates a session-scoped cart and processes all assigned product commands (add/remove).
        """
        id_cart = self.marketplace.new_cart()

        for cart_list in self.carts:
            for cart in cart_list:
                type_command = cart.get("type")
                prod = cart.get("product")
                quantity = cart.get("quantity")
                
                # Block Logic: Dispatcher for marketplace operations.
                if type_command == "add":
                    # Synchronization: Fulfillment barrier. Ensures the thread blocks until all units are acquired.
                    while quantity > 0:
                        ret = self.marketplace.add_to_cart(id_cart, prod)

                        if ret:
                            quantity -= 1
                        else:
                            # Logic: Yields execution to allow producers time to replenish the global pool.
                            time.sleep(self.retry_wait_time)
                else:
                    # Logic: Returns reserved commodities to the marketplace inventory.
                    while quantity > 0:
                        quantity -= 1
                        self.marketplace.remove_from_cart(id_cart, prod)

        # Finalization: Executes the transaction and outputs acquisition results.
        list_prod = self.marketplace.place_order(id_cart)

        for p in list_prod:
            print(self.name, "bought", p)

        pass

import unittest
from threading import Lock
from tema.product import Coffee, Tea
import logging


class Marketplace:
    """
    @brief Central state manager for inventory, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer-owned product pools and active consumer carts.
    Synchronization: Uses distinct locks for producer registration (lock_reg_producer) and cart allocation (lock_cart).
    """

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Capacity limit for each producer's supply queue to prevent buffer bloat.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_consumer = 0
        self.id_producer = 0
        
        self.carts = {} # Mapping: CartID -> List of [Product, ProducerID] tuples.
        self.products = {} # Mapping: ProducerID -> List of available Products.

        self.lock_reg_producer = Lock() # Protects global producer ID sequence.
        self.lock_cart = Lock() # Protects global consumer/cart ID sequence.

        pass

    def register_producer(self):
        """
        @brief Onboards a new supplier into the marketplace ecosystem.
        @return Unique producer identifier.
        """
        logging.info('Entered in register_producer')
        with self.lock_reg_producer:
            self.id_producer += 1
            # Initialization: Allocates an isolated inventory pool for the new producer.
            self.products[self.id_producer] = []
        
        logging.info('Returned id_prod from register_producer')
        return self.id_producer

    def publish(self, producer_id, product):
        """
        @brief Accepts a product from a producer and adds it to their available pool.
        Constraint: Enforces per-producer inventory limits for flow control.
        """
        logging.info('Entered publish')
        add_product = False

        # Block Logic: Threshold check for producer capacity.
        if len(self.products.get(producer_id)) < self.queue_size_per_producer:
            add_product = True
            self.products.get(producer_id).append(product)
            
        logging.info('Returned from publish')
        return add_product

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """
        logging.info('Entered new_cart')

        with self.lock_cart:
            self.id_consumer += 1
            self.carts[self.id_consumer] = []
            
        logging.info('Returned from new_cart')
        return self.id_consumer

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a commodity from a producer pool to a consumer cart.
        Strategy: Exhaustive search across all producers to fulfill the request.
        @return Boolean indicating if the product was successfully located and acquired.
        """
        logging.info('Entered in add_to_cart')

        id_producer = 0
        producer_found = False
        
        # Block Logic: Search pass to identify a supplier with the requested stock.
        for key in list(self.products.keys()):
            for prod in self.products.get(key):
                if prod == product:
                    producer_found = True
                    id_producer = key
                    break
            if producer_found: break

        if producer_found:
            # Invariant: Item is removed from the global pool and associated with the cart and its source ID.
            self.products.get(id_producer).remove(product)
            self.carts.get(cart_id).append([product, id_producer])
            
        logging.info('Returned from add_to_cart')
        return producer_found

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the product to its source producer's inventory.
        """
        logging.info('Entered in remove_from_cart')

        for prod, id_producer in self.carts.get(cart_id):
            if prod == product:
                # Logic: Uses the cached ProducerID in the cart tuple to route the return.
                self.carts.get(cart_id).remove([product, id_producer])
                self.products.get(id_producer).append(product)
                break
        logging.info('Exit from remove_from_cart')

    def place_order(self, cart_id):
        """
        @brief Finalizes the shopping session and flushes the cart contents.
        """
        logging.info('Entered in place_order')

        products_list = []
        for prod, id_prod in self.carts.get(cart_id):
            products_list.append(prod)

        logging.info('Returned from place_order')
        return products_list


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for validating Marketplace transactional integrity and concurrency safety.
    """

    def setUp(self):
        self.marketplace = Marketplace(2)

        self.coffee1 = Coffee(name="Indonesia", acidity="5.05", roast_level="MEDIUM", price=1)
        self.coffee2 = Coffee(name="Brasil", acidity="5.09", roast_level="MEDIUM", price=7)
        self.tea1 = Tea(name="Linden", type="Herbal", price=7)
        self.tea2 = Tea(name="Cactus fig", type="Green", price=5)

    def test_register_producer(self):
        self.setUp()
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        self.setUp()
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(1, self.coffee1))
        self.assertTrue(self.marketplace.publish(1, self.coffee1))
        self.assertFalse(self.marketplace.publish(1, self.coffee1))

    def test_new_cart(self):
        self.setUp()
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        self.setUp()
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.coffee1)
        self.marketplace.publish(1, self.coffee1)

        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))
        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))
        self.assertFalse(self.marketplace.add_to_cart(1, self.coffee1))

    def test_remove_from_cart(self):
        self.setUp()
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.coffee1)
        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))
        self.marketplace.remove_from_cart(1, self.coffee1)

    def test_place_order(self):
        self.setUp()
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.tea1)
        self.marketplace.add_to_cart(1, self.tea1)
        self.assertEqual(len(self.marketplace.place_order(1)), 1)


# Note: The following section was originally in producer.py
# >>>> file: producer.py

class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation.
    Logic: Continuously registers as a supplier and attempts to fulfill production quotas.
    Functional Utility: Models the supply chain with simulated manufacturing delays.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) production metrics.
        @param marketplace Shared resource mediator.
        @param republish_wait_time Duration to wait during marketplace saturation.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        pass

    def run(self):
        """
        @brief Main manufacturing loop for the producer thread.
        """
        while True:
            # Initialization: Onboards as a new entity for every production cycle.
            producer_id = self.marketplace.register_producer()

            for prod in self.products:
                product = prod[0]
                quantity = prod[1]
                wait_time = prod[2]

                # Block Logic: Quota fulfillment with backpressure handling.
                while quantity > 0:
                    ret = self.marketplace.publish(producer_id, product)
                    if ret:
                        quantity -= 1
                        # Logic: Simulated production latency.
                        time.sleep(wait_time)
                    else:
                        # Synchronization: Exponential wait during marketplace inventory overflow.
                        time.sleep(self.republish_wait_time)
        pass


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base representation of a marketable commodity.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized commodity with botanical classification.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized commodity with sensory and acidity profiles.
    """
    acidity: str
    roast_level: str
