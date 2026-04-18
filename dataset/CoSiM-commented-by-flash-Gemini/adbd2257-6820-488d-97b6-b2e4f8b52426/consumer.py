"""
@adbd2257-6820-488d-97b6-b2e4f8b52426/consumer.py
@brief multi-threaded simulation of a retail marketplace using autonomous Producer and Consumer agents.
Architecture: Centralized mediator (Marketplace) manages shared state between independent execution threads.
Functional Utility: Facilitates asynchronous inventory replenishment, session-scoped shopping carts, and concurrent transactional updates.
Synchronization: Employs threading.Lock for critical sections and cooperative yield patterns (sleep) for demand-supply flow control.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing high-level shopping workflows.
    Logic: Sequentially processes assigned carts, interacting with the Marketplace to acquire or return commodities.
    Error Handling: Implements a polling retry loop when requested inventory is out of stock.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Hierarchical list of shopping requests (type, product, quantity).
        @param marketplace Shared resource management interface.
        @param retry_wait_time Temporal duration to yield when inventory is depleted.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        @brief lifecycle manager for the consumer thread.
        Logic: Allocates a unique session (id_cart) and fulfills all commands before finalizing the order.
        """
        for cart in self.carts:
            # Initialization: Establishes a session-scoped inventory buffer.
            id_cart = self.marketplace.new_cart()
            for curr_op in cart:
                # Block Logic: Dispatcher for marketplace operations.
                for _ in range(curr_op["quantity"]):
                    if curr_op["type"] == "add":
                        # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                        while True:
                            check = self.marketplace.add_to_cart(id_cart, curr_op["product"])
                            if not check:
                                sleep(self.retry_wait_time)
                            else:
                                break
                    elif curr_op["type"] == "remove":
                        # Logic: Returns reserved commodities to the marketplace inventory.
                        self.marketplace.remove_from_cart(id_cart, curr_op["product"])
            
            # Finalization: executes the transaction and flushes results.
            self.marketplace.place_order(id_cart)

import threading
import unittest
import logging
import logging.handlers
from tema.product import Product

LOG_FILE_NAME = 'marketplace.log'
LOGGING_LEVEL = logging.DEBUG

class Marketplace:
    """
    @brief Shared resource manager coordinating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for per-producer occupancy, global product availability, and active carts.
    Synchronization: Uses distinct locks for producer registration (reg_p_lock), publication (publish_lock), and cart allocation (new_cart_lock) to minimize contention.
    Observability: Integrates TimedRotatingFileHandler for structured audit logging of all concurrent events.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum inventory allowed per supplier for backpressure control.
        """
        self.allowed_quantity = queue_size_per_producer
        self.producers = {} # Mapping: ProducerID -> Current stock count.
        self.last_prod_id = 0
        self.last_cart_id = 0
        self.consumers = {} # Mapping: CartID -> List of Products.
        self.market = {} # Global Registry: Product -> SourceProducerID.
        self.all_products = [] # List of all available Product units.

        self.reg_p_lock = threading.Lock()
        self.publish_lock = threading.Lock()
        self.new_cart_lock = threading.Lock()
        self.add_to_cart_lock = threading.Lock()
        
        # Block Logic: Audit logging configuration.
        self.logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE_NAME,
                                                            interval=30, backupCount=10)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(LOGGING_LEVEL)



    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its inventory tracking.
        @return Unique producer identifier.
        """
        self.reg_p_lock.acquire()
        self.logger.info("-- Intrare in metoda register_producer")
        self.last_prod_id += 1
        # Initialization: Scaffolds the occupancy metrics for the new producer.
        self.producers[self.last_prod_id] = 0
        self.reg_p_lock.release()
        self.logger.info("-- Iesire din metoda register_producer")
        return self.last_prod_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the global pool.
        Constraint: Rejects publication if the supplier's individual queue is saturated (backpressure).
        """

        self.publish_lock.acquire()
        self.logger.info("-- Intrare in metoda publish cu param producer_id = %s si product = %s",
                         str(producer_id), str(product))
        
        # Logic: Updates global product registry.
        self.market[product] = producer_id
        self.all_products.append(product)
        old_val = self.producers.get(producer_id)
        self.producers[producer_id] = old_val + 1
        self.publish_lock.release()

        # Block Logic: Threshold check for supply-side flow control.
        if self.producers[producer_id] > self.allowed_quantity:
            # Reversion: Decrements count if limit was exceeded.
            self.producers[producer_id] = old_val
            self.logger.info("-- Iesire din metoda publish cu rezultatul False")
            return False

        self.logger.info("-- Iesire din metoda publish cu rezultatul True")
        return True

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        """
        self.new_cart_lock.acquire()
        self.logger.info("-- Intrare in metoda new_cart")
        self.last_cart_id += 1
        self.consumers[self.last_cart_id] = []
        self.logger.info("-- Iesire din metoda new_cart")
        self.new_cart_lock.release()
        return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a unit from producer inventory to a specific cart.
        Logic: Acquires product from the global registry and decrements the originating producer's count.
        @return Boolean indicating acquisition success.
        """
        self.add_to_cart_lock.acquire()
        self.logger.info("-- Intrare in metoda add_to_cart cu param cart_id = %s si product = %s",
                         str(cart_id), str(product))
        
        # Block Logic: Acquisition check.
        if product not in self.market or product not in self.all_products:
            self.logger.info("-- Iesire din metoda add_to_cart -> Nu exista produsul")
            self.add_to_cart_lock.release()
            return False

        # Invariant: Item must be removed from global pool before being assigned to a session.
        self.consumers[cart_id].append(product)
        prod = self.market[product]
        self.all_products.remove(product)
        self.producers[prod] -= 1


        self.logger.info("-- Iesire triumfatoare din add_to_cart")
        self.add_to_cart_lock.release()
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to its source producer's pool.
        """
        self.logger.info("-- Intrare in metoda remove_from_cart cu param cart_if = %s si %s",
                         str(cart_id), str(product))
        self.consumers[cart_id].remove(product)
        
        # Logic: Restores unit to global availability.
        prod = self.market.get(product)
        self.all_products.append(product)
        self.producers[prod] += 1
        self.logger.info("-- Iesire din metoda remove_from_cart")

    def place_order(self, cart_id):
        """
        @brief Finalizes the shopping session and flushes results to standard output.
        """
        self.logger.info("-- Intrare in metoda place_order cu param cart_id = %s ", str(cart_id))
        
        # Serialization: Outputs acquisition logs to console.
        for prod in self.consumers[cart_id]:
            print(threading.currentThread().getName() + " bought " + str(prod))
        self.logger.info("-- Iesire din metoda place_order")
        return self.consumers[cart_id]

class TestMarketPlace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and state transitions.
    """
    
    def test_register_producer(self):
        
        market = Marketplace(12)
        self.assertEqual(1, market.register_producer())

    def test_publish(self):
        
        market = Marketplace(2)
        market.register_producer()
        self.assertEqual(True, market.publish(1, Product("TeaName", 23.0)))

    def test_new_cart(self):
        
        market = Marketplace(2)
        first_cart = market.new_cart()
        second_cart = market.new_cart()
        self.assertEqual(1, first_cart)
        self.assertEqual(2, second_cart)

    def test_add_to_cart(self):
        
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutz", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutz", 5.0))
        self.assertEqual(0, len(market.all_products))

    def test_remove_from_cart(self):
        
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.remove_from_cart(cart_id, Product("Cafelutzaaa", 5.0))
        self.assertEqual(1, len(market.all_products))

    def test_place_order(self):
        
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.publish(1, Product("Ceiut", 3.0))
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.publish(1, Product("Ceiut", 3.0))
        market.publish(1, Product("Altceva", 3.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Ceiut", 3.0))
        self.assertEqual(3, len(market.place_order(cart_id)))


from threading import Thread
import time


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
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.published_products = 0
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        @brief Main manufacturing loop for the producer execution context.
        """
        # Initialization: Registers as a supplier to obtain a persistent ID.
        id_prod = self.marketplace.register_producer()

        while 1:
            for (product, quantity, production_time) in self.products:
                for i in range(quantity):
                    # Block Logic: Publish-retry loop for backpressure management.
                    check = self.marketplace.publish(id_prod, product)
                    if check:
                        # Logic: Simulated industrial processing duration.
                        time.sleep(production_time)
                    else:
                        # Synchronization: Exponential wait during marketplace inventory overflow.
                        time.sleep(self.republish_wait_time)
                        i += 1
                    i -= 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base representation of a marketable commodity unit.
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
