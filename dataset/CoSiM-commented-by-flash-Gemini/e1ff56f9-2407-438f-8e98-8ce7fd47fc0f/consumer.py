"""
@e1ff56f9-2407-438f-8e98-8ce7fd47fc0f/consumer.py
@brief multi-threaded electronic marketplace with centralized session and inventory management.
This module implements a coordinated trading environment where Producers supply goods 
and Consumers execute shopping sequences. The system uses a centralized locking 
model to ensure atomic updates to producer buffers and consumer carts, with 
integrated unit tests to validate state transition correctness. A rotating log 
system tracks all marketplace operations for auditing.

Domain: Concurrent State Management, Producer-Consumer Simulation, Integrated Unit Testing.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Consumer entity simulating a shopper.
    Functional Utility: Manages multiple shopping sessions (carts) and performs 
    automated transactions using a polling-based backoff strategy.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: Nested list of shopping operation batches.
        @param marketplace: Central trading coordinator.
        @param retry_wait_time: delay between failed acquisition attempts.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        Main execution loop for shopper actions.
        Logic: Orchestrates session creation and sequential fulfillment of 
        'add' and 'remove' tasks.
        """
        for cart in self.carts:
            # Atomic creation of a new transaction context.
            cart_id = self.marketplace.new_cart()
            for entry in cart:
                (entry_type, product, quantity) =\
                    (entry["type"], entry["product"], entry["quantity"])
                aux = 0
                while aux < quantity:
                    # Block Logic: Workload fulfillment.
                    if entry_type == "add":
                        # Busy-wait polling for item availability.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    else:
                        # Transaction reversal: restore item to global supply.
                        self.marketplace.remove_from_cart(cart_id, product)
                    aux = aux + 1

            # Commit: finalize the transaction.
            self.marketplace.place_order(cart_id)

from threading import Lock, currentThread
from logging.handlers import RotatingFileHandler
import unittest
import logging
import tema.product

class Marketplace:
    """
    Central coordinator for the trading simulation.
    Functional Utility: Manages supply lines (producers) and shopper sessions 
    (consumers) using a centralized mutex for all state mutations.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace hub.
        @param queue_size_per_producer: Capacity limit per supply line.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {} 
        self.products = []
        self.consumers = {} 
        
        # Centralized Synchronization Primitive.
        self.lock = Lock()
        
        # System Audit Configuration.
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=5000, backupCount=10)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Allocates a new supply line ID.
        Logic: uses the current count of producers to generate a unique index.
        """
        with self.lock:
            producer_id = len(self.producers) + 1
            self.producers[producer_id] = []
            self.logger.info("return from register_producer %s", str(producer_id))
            return producer_id

    def publish(self, producer_id, product):
        """
        Accepts a product from a producer.
        Logic: verifies producer buffer capacity before publication.
        @return: True if accepted, False if full.
        """
        self.logger.info("input to publish %s %s", str(producer_id), str(product))

        # Capacity Guard.
        if len(self.producers[producer_id]) > self.queue_size_per_producer:
            self.logger.info("return from publish False")
            return False

        # Atomic state update.
        self.producers[producer_id].append(product)
        self.products.append(product)
        self.logger.info("return from publish True")
        return True

    def new_cart(self):
        """Creates a new shopper session context."""
        with self.lock:
            cart_id = len(self.consumers) + 1
            self.consumers[cart_id] = []
            self.logger.info("return from new_cart %s", str(cart_id))
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers a product from a supply line to a consumer cart.
        Logic: performs a linear search across all producer buffers to find 
        the target item.
        """
        self.logger.info("input to add_to_cart %s %s", str(cart_id), str(product))

        with self.lock:
            aux = 0
            if cart_id in self.consumers:
                # Block Logic: Global Inventory Search.
                for producer_id in self.producers:
                    if product in self.producers[producer_id]:
                        aux = producer_id
                        break # Optimization: item found.

                if aux == 0:
                    self.logger.info("return from add_to_cart False")
                    return False

            # State Transition: move item from producer to consumer.
            self.consumers[cart_id].append((aux, product))
            self.products.remove(product)
            self.producers[aux].remove(product)
            self.logger.info("return from add_to_cart True")
            return True

    def remove_from_cart(self, cart_id, product):
        """Restores an item from a cart back to its originating producer."""
        self.logger.info("input to remove_from_cart %s %s", str(cart_id), str(product))

        with self.lock:
            if cart_id in self.consumers:
                for search in self.consumers[cart_id]:
                    if search[1] == product:
                        # Transaction Reversal.
                        self.consumers[cart_id].remove(search)
                        self.products.append(product)
                        # restore to producer buffer if space permits.
                        if len(self.producers[search[0]]) < self.queue_size_per_producer:
                            self.producers[search[0]].append(product)
                        return

    def place_order(self, cart_id):
        """Finalizes the purchase and prints the manifest."""
        self.logger.info("input to place_order %s", str(cart_id))

        if cart_id in self.consumers:
            order_list = []
            for product in self.consumers[cart_id]:
                print(currentThread().getName() + " bought " + str(product[1]))
                order_list.append(product[1])

            self.logger.info("return from place_order %s", str(order_list))
            return order_list

        return []



class TestMarketplace(unittest.TestCase):
    """
    Unit testing suite for validating the marketplace's state management logic.
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
        # Verify item is no longer in cart.
        self.assertEqual(any(p[1] == self.product for p in self.marketplace.consumers[1]), False, 'failed to remove')

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

import time
from threading import Thread


class Producer(Thread):
    """
    Simulation thread representing a manufacturing unit.
    Functional Utility: Manages the continuous production cycle of items and 
    observes marketplace backpressure.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the producer and secures a supply line ID.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        Main production cycle.
        Logic: Cycles through product quotas, observing production times and hub capacity.
        """
        while True:
            for (product, quantity, wait_time) in self.products:
                aux = 0
                while aux < quantity:
                    aux = aux + 1
                    # Block Logic: Publication Retry loop.
                    while not self.marketplace.publish(self.prod_id, product):
                        time.sleep(self.republish_wait_time)
                    # Simulate manufacturing overhead.
                    time.sleep(wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """Core data model for marketplace items."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Beverage specialization."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Beverage specialization."""
    acidity: str
    roast_level: str
