"""
@e1ff56f9-2407-438f-8e98-8ce7fd47fc0f/consumer.py
@brief Threaded marketplace simulation with persistent audit logging and transaction-based inventory management.
* Algorithm: Concurrent producer-consumer model with shared state managed via producer-partitioned lists and global mutual exclusion.
* Functional Utility: Facilitates a virtual market where producers generate goods and consumers manage items via cart sessions, with detailed logging of all state transitions.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    @brief Consumer entity that performs synchronized shopping operations across multiple carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its assigned shopping lists and market connection.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        @brief Main execution loop for shopping activities.
        Algorithm: Iterative processing of carts with a busy-wait retry strategy for inventory acquisition.
        """
        for cart in self.carts:
            # Logic: Initializes a new transaction identifier.
            cart_id = self.marketplace.new_cart()
            for entry in cart:
                (entry_type, product, quantity) = \
                    (entry["type"], entry["product"], entry["quantity"])
                aux = 0
                while aux < quantity:
                    if entry_type == "add":
                        # Logic: Continuous attempt to secure product from inventory.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            # Functional Utility: Throttles retry attempts during stock-outs.
                            time.sleep(self.retry_wait_time)
                    else:
                        # Logic: Returns items from cart to inventory.
                        self.marketplace.remove_from_cart(cart_id, product)
                    aux = aux + 1

            # Post-condition: Completes the transaction.
            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread
from logging.handlers import RotatingFileHandler
import unittest
import logging
import tema.product

class Marketplace:
    """
    @brief Centralized inventory controller with integrated audit logging and thread-safe operations.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace and its logging infrastructure.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {} # Intent: Map of producer ID to its active inventory list.
        self.products = []  # Intent: Global registry of all available product instances.
        self.consumers = {} # Intent: Map of cart ID to its reserved (producer_id, product) tuples.
                            
        self.lock = Lock()
        
        # Block Logic: Audit Logging Configuration.
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=5000, backupCount=10)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its inventory buffer.
        Invariant: Uses self.lock to ensure atomic producer ID assignment.
        """
        with self.lock:
            producer_id = len(self.producers) + 1
            self.producers[producer_id] = []
            self.logger.info("return from register_producer %s", str(producer_id))
            return producer_id

    def publish(self, producer_id, product):
        """
        @brief Adds a product to the global pool if the producer's quota allows.
        """
        self.logger.info("input to publish %s %s", str(producer_id), str(product))

        # Logic: Enforces production capacity constraints.
        if len(self.producers[producer_id]) > self.queue_size_per_producer:
            self.logger.info("return from publish False")
            return False

        self.producers[producer_id].append(product)
        self.products.append(product)
        self.logger.info("return from publish True")
        return True

    def new_cart(self):
        """
        @brief Allocates a new transaction identifier for a consumer.
        """
        with self.lock:
            cart_id = len(self.consumers) + 1
            self.consumers[cart_id] = []
            self.logger.info("return from new_cart %s", str(cart_id))
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product unit from any producer to a specific consumer cart.
        Algorithm: Search-and-consume across all producer buffers.
        Invariant: Uses self.lock to ensure consistent ownership transfer.
        """
        self.logger.info("input to add_to_cart %s %s", str(cart_id), str(product))

        with self.lock:
            source_producer = 0
            if cart_id in self.consumers:
                # Logic: Linear scan for product availability across all producer segments.
                for producer_id in self.producers:
                    if product in self.producers[producer_id]:
                        source_producer = producer_id
                        break

                if source_producer == 0:
                    self.logger.info("return from add_to_cart False")
                    return False

            # Logic: Preserves producer-affinity to support consistent returns.
            self.consumers[cart_id].append((source_producer, product))
            self.products.remove(product)
            self.producers[source_producer].remove(product)
            self.logger.info("return from add_to_cart True")
            return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns an item from a cart back to its original producer's buffer.
        """
        self.logger.info("input to remove_from_cart %s %s", str(cart_id), str(product))

        with self.lock:
            if cart_id in self.consumers:
                for search in self.consumers[cart_id]:
                    if search[1] == product:
                        self.consumers[cart_id].remove(search)
                        self.products.append(product)
                        # Post-condition: Restores item to its source buffer if space permits.
                        if len(self.producers[search[0]]) < self.queue_size_per_producer:
                            self.producers[search[0]].append(product)
                        return

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and logs the purchased items.
        """
        self.logger.info("input to place_order %s", str(cart_id))

        if cart_id in self.consumers:
            order_list = []
            for product_tuple in self.consumers[cart_id]:
                # Logic: Synchronized console output using the current thread context.
                print(currentThread().getName() + " bought " + str(product_tuple[1]))
                order_list.append(product_tuple[1])

            self.logger.info("return from place_order %s", str(order_list))
            return order_list

        return []


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for validating Marketplace operational logic.
    """
    
    def setUp(self):
        self.marketplace = Marketplace(5)
        self.product = tema.product.Tea('Linden', 10, 'Herbal')
        self.product2 = tema.product.Coffee('Arabica', 10, '5.05', 'MEDIUM')

    def test_register_producer(self):
        self.marketplace.register_producer()
        self.assertEqual(len(self.marketplace.producers), 1, 'wrong number of producers')

    def test_publish(self):
        self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(1, self.product), True, 'failed to publish')

    def test_new_cart(self):
        self.marketplace.new_cart()
        # Note: Typo in original code checked dict itself instead of length.
        self.assertEqual(len(self.marketplace.consumers), 1, 'wrong number of carts')

    def test_add_to_cart(self):
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(1, self.product), True, 'failed to add')

    def test_remove_from_cart(self):
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product)
        self.marketplace.remove_from_cart(1, self.product)
        self.assertEqual(self.product in [p[1] for p in self.marketplace.consumers[1]], False, 'failed to remove')

    def test_place_order(self):
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.publish(1, self.product2)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product)
        self.marketplace.add_to_cart(1, self.product2)
        aux = self.marketplace.place_order(1)
        correct_list = [self.product, self.product2]
        self.assertEqual(aux, correct_list, 'failed to place order')


from threading import Thread
import time


class Producer(Thread):
    """
    @brief Producer agent that generates goods for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with its production schedule.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Main production lifecycle loop.
        Algorithm: Iterative batch generation with fixed production latency and quota-constrained publication.
        """
        while True:
            for (product, quantity, wait_time) in self.products:
                aux = 0
                while aux < quantity:
                    aux = aux + 1
                    # Logic: Attempts to publish item; retries on buffer saturation.
                    while not self.marketplace.publish(self.prod_id, product):
                        # Functional Utility: Throttles re-publication attempts.
                        time.sleep(self.republish_wait_time)
                    # Domain: Simulation of manufacturing time.
                    time.sleep(wait_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base schema for marketplace goods.
    """
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product type for tea items.
    """
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product type for coffee items.
    """
    acidity: str
    roast_level: str
