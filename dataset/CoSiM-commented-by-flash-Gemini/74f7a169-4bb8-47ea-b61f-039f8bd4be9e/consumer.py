"""
@74f7a169-4bb8-47ea-b61f-039f8bd4be9e/consumer.py
@brief Event-driven simulation of a retail marketplace using concurrent Producer and Consumer agents.
Architecture: Centralized mediator (Marketplace) manages shared state between independent execution threads.
Functional Utility: Models asynchronous inventory flow, transaction processing, and multi-cart session management.
Synchronization: Employs threading.Lock for critical sections and cooperative multitasking (sleep) for flow control.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing high-level purchase requests.
    Logic: Processes a batch of carts, each containing a sequence of inventory operations (add/remove).
    Functional Utility: Abstracts lower-level marketplace interactions into high-level 'add_product' and 'remove_product' routines.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Structured list of shopping sessions and product requests.
        @param marketplace Shared resource management interface.
        @param retry_wait_time Duration to wait when a requested product is out of stock.
        """
        Thread.__init__(self, **kwargs)


        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def add_product(self, cart_id, product, quantity):
        """
        @brief Iterative acquisition loop for a specific product.
        Block Logic: Implements a polling retry mechanism. If 'add_to_cart' fails, the thread yields to allow producers to replenish.
        Invariant: All requested units must be acquired before the method returns.
        """
        for _ in range(quantity):
            while not self.marketplace.add_to_cart(cart_id, product):
                sleep(self.retry_wait_time)

    def remove_product(self, cart_id, product, quantity):
        """
        @brief Iterative removal of products from a specific cart.
        Logic: Returns the product to the marketplace pool for other consumers.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """
        @brief Lifecycle manager for the consumer thread.
        Logic: Iterates through assigned carts, fulfills all internal requests, and finalizes the order.
        """
        for cart in self.carts:
            # Logic: Initializes a new isolated session in the marketplace.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Dispatches commands based on the request type ('add' or 'remove').
            for request in cart:
                command = request["type"]
                product = request["product"]
                quantity = request["quantity"]

                if command == "add":
                    self.add_product(cart_id, product, quantity)
                elif command == "remove":
                    self.remove_product(cart_id, product, quantity)

            # Functional Utility: Transitions the cart from 'active' to 'finalized' state.
            order = self.marketplace.place_order(cart_id)

            # Logic: Serialized output of the finalized transaction.
            self.marketplace.print_order(order, self.name)

import logging
import time
import unittest
from logging.handlers import RotatingFileHandler
from threading import Lock, current_thread


class Marketplace:
    """
    @brief Shared resource mediator managing inventory, carts, and thread registration.
    State Management: Maintains mappings for producers, active carts, and product ownership.
    Synchronization: Uses a central mutex (Lock) to protect non-thread-safe dictionary operations and ID increments.
    """

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum allowed inventory per producer for flow control.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0  
        self.cart_id = 0  
        self.producers = {} # Mapping: ID -> List of available products.
        self.carts = {} # Mapping: ID -> List of reserved products.
        self.products = {} # Reverse mapping: Product -> Owner Producer ID.
        self.mutex = Lock() # Central synchronization primitive.

        # Observability: Structured logging to track concurrent state transitions and audit trails.
        self.logger = logging.getLogger("Logger")
        self.handler = RotatingFileHandler("marketplace.log", maxBytes=25000, backupCount=10)
        self.handler.setLevel(logging.INFO)
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s: %(message)s"))
        logging.Formatter.converter = time.gmtime
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def register_producer(self):
        """
        @brief Generates a unique producer ID and initializes its inventory pool.
        """
        with self.mutex:
            self.producers[self.producer_id] = []
            self.producer_id += 1
            self.logger.info("Thread %s has producer_id=%d", current_thread().name,
                             self.producer_id - 1)
            return str(self.producer_id - 1)

    def publish(self, producer_id, product):
        """
        @brief Adds a product to the marketplace inventory on behalf of a producer.
        Constraint: Enforces the queue_size_per_producer limit as a backpressure mechanism.
        """
        self.logger.info("Thread %s has producer_id=%s, product=%s", current_thread().name,
                         producer_id, product)
        producer_index = int(producer_id)

        # Block Logic: Boundary check for producer capacity.
        if len(self.producers[producer_index]) == self.queue_size_per_producer:
            return False

        # State Transition: Product becomes available for consumer acquisition.
        self.producers[producer_index].append(product)
        self.products[product] = producer_index

        return True

    def new_cart(self):
        """
        @brief Allocates a new shopping cart identifier for a consumer.
        """
        with self.mutex:
            self.carts[self.cart_id] = []
            self.cart_id += 1
            self.logger.info("Thread %s has cart_id=%s", current_thread().name, self.cart_id - 1)
            return self.cart_id - 1

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers a product from producer inventory to a consumer cart.
        Logic: Performs a global search across all producers for the first available unit.
        """
        self.logger.info("Thread %s has cart_id=%d, product=%s", current_thread().name,
                         cart_id, product)

        # Block Logic: Scanning loop to locate and acquire requested resource.
        for list_of_products in self.producers.values():
            if product in list_of_products:
                # Invariant: Product must be removed from inventory before being assigned to a cart.
                list_of_products.remove(product)
                self.carts[cart_id].append(product)

                return True

        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverses an acquisition, returning the product to its original producer's pool.
        """
        self.logger.info("Thread %s has cart_id=%d, product=%s", current_thread().name,
                         cart_id, product)

        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            # Logic: Uses the stored product-producer mapping to route the return correctly.
            self.producers[self.products[product]].append(product)

            return True

        return False

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and flushes the cart state.
        """
        self.logger.info("Thread %s has cart_id=%d", current_thread().name, cart_id)

        # Functional Utility: Atomic retrieval and deletion of cart contents.
        cart_content = self.carts[cart_id]
        self.carts.pop(cart_id)

        self.logger.info("Thread %s has cart_content=%s", current_thread().name, cart_content)
        return cart_content

    def print_order(self, order, name):
        """
        @brief Serializes and outputs order details to the console.
        Synchronization: Protects stdout to prevent interleaved lines from multiple threads.
        """
        with self.mutex:
            self.logger.info("Thread %s has order=%s, name=%s", current_thread().name,
                             order, name)
            for product in order:
                print(f"{name} bought {product}")


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying core Marketplace transactional logic.
    """
    def setUp(self):
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        
        for producer_id in range(100):
            self.assertEqual(self.marketplace.register_producer(), str(producer_id), "wrong id")

    def test_publish(self):
        
        prod_id = self.marketplace.register_producer()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(product1 in self.marketplace.producers[int(prod_id)],
                        "product is not on the marketplace")
        self.assertEqual(self.marketplace.products[product1], int(prod_id),
                         "don't recognize the product")

        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")
        self.assertTrue(product2 in self.marketplace.producers[int(prod_id)],
                        "product is not on the marketplace")
        self.assertEqual(self.marketplace.products[product2], int(prod_id),
                         "don't recognize the product")

        self.assertFalse(self.marketplace.publish(prod_id, product3), "failed not to publish")

    def test_new_cart(self):
        
        for cart_id in range(100):
            self.assertEqual(self.marketplace.new_cart(), cart_id, "wrong id")

    def test_add_to_cart(self):
        
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")
        self.assertFalse(self.marketplace.publish(prod_id, product3), "failed not to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(product1 in self.marketplace.carts[cart_id], "is not in the cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")
        self.assertTrue(product2 in self.marketplace.carts[cart_id], "is not in the cart")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product3),
                         "product should not be in the market")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product1),
                         "product should be already in the cart")

    def test_remove_from_cart(self):
        
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")

        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product1), "not in the cart")
        self.assertTrue(product1 not in self.marketplace.carts[cart_id], "is in cart")
        self.assertTrue(product1 in self.marketplace.producers[int(prod_id)],
                        "not in producer's list")
        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product2), "not in the cart")
        self.assertTrue(product2 not in self.marketplace.carts[cart_id], "is in cart")
        self.assertTrue(product2 in self.marketplace.producers[int(prod_id)],
                        "not in producer's list")
        self.assertFalse(self.marketplace.remove_from_cart(cart_id, product3), "not in the cart")

    def test_place_order(self):
        
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")

        self.assertEqual(self.marketplace.place_order(cart_id), ["coffee", "tea"],
                         "not the same order")


# Note: The following section was originally in producer.py
# >>>> file: producer.py

class Producer(Thread):
    """
    @brief Producer agent responsible for resource generation.
    Logic: Cycles through a list of production targets, publishing them to the marketplace.
    Functional Utility: Models the supply side of the economy with simulated production delays.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ProductionTime) tuples.
        @param marketplace Shared resource mediator.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Logic: Obtains credentials from the marketplace during initialization.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Continuous production loop for the producer thread.
        """
        while True:
            for product in self.products:
                product_id = product[0]
                quantity = product[1]
                production_time = product[2]
                
                for _ in range(quantity):
                    # Block Logic: Publish-retry loop for flow control.
                    while not self.marketplace.publish(self.producer_id, product_id):
                        sleep(self.republish_wait_time)

                    # Logic: Simulated duration required to 'create' the resource.
                    sleep(production_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable representation of a commodity.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized commodity with beverage-specific attributes.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized commodity with roasting and acidity profile.
    """
    acidity: str
    roast_level: str
