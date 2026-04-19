
"""
@file consumer.py
@brief Thread-safe implementation of a Producer-Consumer marketplace simulation.

This module provides a multi-threaded architectural framework for a simulated 
marketplace. It utilizes synchronization primitives (RLocks) to manage shared 
state across producers (who publish products) and consumers (who manage carts 
and place orders).

Domain: Multi-threaded Production Systems, Synchronization Patterns.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Acts as a client entity in the marketplace, executing cart operations in a dedicated thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread with task instructions.

        :param carts: A list of carts, each containing a sequence of instructions.
        :param marketplace: Reference to the shared Marketplace instance.
        :param retry_wait_time: Interval to wait when a marketplace operation is throttled.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_to_cart(self, cart_id, product, quantity):
        """
        Sequentially adds multiple units of a product to a specific cart.
        Logic: Retries with backoff if the marketplace cannot immediately fulfill the request.
        """
        while quantity > 0:
            while not self.marketplace.add_to_cart(cart_id, product):
                time.sleep(self.retry_wait_time)
            quantity = quantity - 1

    def remove_from_cart(self, cart_id, product, quantity):
        """
        Sequentially removes multiple units of a product from a specific cart.
        Logic: Retries with backoff if the operation is temporarily blocked.
        """
        while quantity > 0:
            while not self.marketplace.remove_from_cart(cart_id, product):
                time.sleep(self.retry_wait_time)
            quantity = quantity - 1

    def order_cart(self, cart_id):
        """
        Finalizes the purchase of all items in the specified cart.
        """
        products = self.marketplace.place_order(cart_id)
        for prod in products:
            print(f"{self.name} bought {prod}")

    def run(self):
        """
        Main execution loop for the consumer thread.
        Logic: Processes all carts and their associated instructions (add/remove) 
        before placing final orders.
        """
        cart_id = self.marketplace.new_cart()
        for cart in self.carts:
            for instruction in cart:
                instr_type = instruction.get("type")
                product = instruction.get("product")
                quantity = instruction.get("quantity")
                # Block Logic: Operation routing based on instruction type.
                if instr_type == "add":
                    self.add_to_cart(cart_id, product, quantity)
                elif instr_type == "remove":
                    self.remove_from_cart(cart_id, product, quantity)
            self.order_cart(cart_id)

import logging
import unittest
from logging.handlers import RotatingFileHandler
from threading import RLock


class Marketplace:
    """
    Central hub for product transactions, managing shared resource access through fine-grained locking.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Sets up the marketplace state with isolated locks for producers, carts, and products.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producers = 0
        self.producer_queue_size = {}
        self.producer_lock = RLock()

        self.carts = 0
        self.cart_lock = RLock()

        self.cart_id_data = {}
        self.cart_id_lock = {}

        self.products = []
        self.product_lock = RLock()

        # Monitoring: Configures persistent logging for transaction auditing.
        self.logger = logging.getLogger("Marketplace")
        self.logger.addHandler(RotatingFileHandler("marketplace.log", maxBytes=10000, backupCount=5))
        self.logger.setLevel(logging.INFO)

    def register_producer(self):
        """
        Assigns a unique identifier to a new producer and initializes their publishing quota.
        """
        self.producer_lock.acquire()
        self.producers = self.producers + 1
        producer_id = self.producers


        self.producer_queue_size[producer_id] = self.queue_size_per_producer
        self.producer_lock.release()

        self.logger.info(f"Producer {producer_id} has been registered")

        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product in the marketplace.
        Logic: Enforces publishing limits per producer using atomic state updates.
        """
        if self.producer_queue_size[producer_id] > 0:
            self.product_lock.acquire()
            self.products.append((product, producer_id))
            self.product_lock.release()

            self.producer_lock.acquire()
            self.producer_queue_size[producer_id] = self.producer_queue_size[producer_id] - 1
            self.producer_lock.release()

            self.logger.info(f"Product {product} has been published by producer {producer_id}")
            return True
        return False

    def new_cart(self):
        """
        Initializes a new shopping session for a consumer.
        """
        self.cart_lock.acquire()

        self.carts = self.carts + 1
        cart_id = self.carts
        self.cart_id_data[cart_id] = []
        self.cart_id_lock[cart_id] = RLock()



        self.cart_lock.release()

        self.logger.info(f"Customer created cart {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers a product from the global listing to a specific consumer cart.
        Logic: Uses multi-lock acquisition to ensure atomic transfer between 
        the global product pool and the private cart.
        """
        self.product_lock.acquire()
        for prod in self.products:
            if prod[0] == product:
                # Critical Section: Removes from global inventory.
                self.products.remove(prod)
                self.product_lock.release()

                # Critical Section: Adds to consumer cart.
                self.cart_id_lock[cart_id].acquire()
                self.cart_id_data[cart_id].append(prod)
                self.cart_id_lock[cart_id].release()

                self.logger.info(f"Customer added product {product} to cart {cart_id}")
                return True



        self.product_lock.release()
        self.logger.info(f"Customer failed to add product {product} to cart {cart_id}")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Returns a product from a cart back to the global listing.
        """
        self.cart_id_lock[cart_id].acquire()
        self.product_lock.acquire()

        for prod in self.cart_id_data[cart_id]:
            if prod[0] == product:
                self.cart_id_data[cart_id].remove(prod)
                self.products.append(prod)



                self.cart_id_lock[cart_id].release()
                self.product_lock.release()

                self.logger.info(f"Customer removed product {prod} from cart {cart_id}")
                return True

    def place_order(self, cart_id):
        """
        Executes a final transaction for all items in a cart, replenishing 
        producer quotas for future listings.
        """
        products = []
        self.cart_id_lock[cart_id].acquire()
        for prod, producer_id in self.cart_id_data[cart_id]:
            products.append(prod)

            # Logic: Restores quota to the original producer.
            self.producer_lock.acquire()
            self.producer_queue_size[producer_id] = self.producer_queue_size[producer_id] + 1
            self.producer_lock.release()

        self.cart_id_data[cart_id].clear()
        self.cart_id_lock[cart_id].release()
        self.logger.info(f"Customer placed an order and emptied cart {cart_id}")
        return products


class TestMarketplace(unittest.TestCase):
    def setUp(self) -> None:
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        producers = []
        for _ in range(1, 50):
            producers.append(self.marketplace.register_producer())
        self.assertEqual(len(producers), len(list(dict.fromkeys(producers))))

    def test_publish(self):
        producer = self.marketplace.register_producer()

        product = "Coffee"
        self.assertEqual(self.marketplace.publish(producer, product), True)
        self.assertEqual(self.marketplace.products[0], (product, producer))

        product = "Tea"
        self.assertEqual(self.marketplace.publish(producer, product), True)
        self.assertEqual(self.marketplace.products[1], (product, producer))

        self.assertEqual(self.marketplace.publish(producer, product), False)

    def test_add_to_cart(self):
        cart_id = self.marketplace.new_cart()
        product = "Coffee"
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), False)

        producer = self.marketplace.register_producer()
        self.marketplace.publish(producer, product)

        self.assertEqual(self.marketplace.products[0], (product, producer))
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)
        self.assertEqual(self.marketplace.cart_id_data[cart_id], [(product, producer)])
        self.assertEqual(self.marketplace.products, [])

    def test_remove_from_cart(self):
        cart_id = self.marketplace.new_cart()
        product = "Tea"
        producer = self.marketplace.register_producer()

        self.marketplace.publish(producer, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), True)
        self.assertEqual(self.marketplace.products[0], (product, producer))

    def test_place_order(self):
        cart_id = self.marketplace.new_cart()
        product = "Coffee"
        producer = self.marketplace.register_producer()

        self.marketplace.publish(producer, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.place_order(cart_id), product)


from threading import Thread
import time


class Producer(Thread):
    """
    Supply-side entity that populates the marketplace with products in a background thread.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        :param products: Catalog of products to be produced.
        :param marketplace: Shared marketplace structure.
        :param republish_wait_time: Cooldown after a failed publish attempt (quota reached).
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def publish(self, product, quantity, publish_wait_time, producer_id):
        """
        Background task to incrementally publish units of a specific product.
        Logic: Throttles production based on marketplace capacity and fixed wait times.
        """
        while quantity > 0:
            while not self.marketplace.publish(producer_id, product):
                time.sleep(self.republish_wait_time)
            time.sleep(publish_wait_time)
            quantity = quantity - 1

    def run(self):
        """
        Main lifecycle loop for the producer.
        Logic: Infinite production cycle over the assigned product catalog.
        """
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                self.publish(product[0], product[1], product[2], producer_id)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Immutable representation of a generic product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Specialization for tea products.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Specialization for coffee products.
    """
    acidity: str
    roast_level: str
