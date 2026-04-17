"""
@bf7e78c0-af21-48be-9d51-edc8876d4b81/consumer.py
@brief Simulation of a multi-threaded electronic marketplace with Producers and Consumers.
This module implements a coordinated trading environment where Producers supply goods 
to a central Marketplace, and Consumers perform transactional operations (adding/removing 
items) asynchronously. The system uses a backoff-and-retry strategy to handle resource 
limitations such as full queues or unavailable stock.

Domain: Concurrent Systems, Producer-Consumer Pattern, Backoff Algorithms.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Simulates a network-attached client performing shopping operations.
    Functional Utility: Manages multiple shopping carts and interacts with the 
    marketplace to fulfill a predefined set of transaction requests.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: A list of shopping task lists (each list contains operations).
        @param marketplace: The central coordinator for products and transactions.
        @param retry_wait_time: Interval to wait before retrying a failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Main execution loop for the consumer.
        Logic: Iterates through each cart and its associated operations (add/remove), 
        then finalize orders.
        """
        for cart in self.carts:
            # Atomic creation of a new session context in the marketplace.
            cart_id = self.marketplace.new_cart()
            for oper in cart:
                type_of_operation = oper["type"]
                prod = oper["product"]
                quantity = oper["quantity"]
                if type_of_operation == "add":
                    self.add_cart(cart_id, prod, quantity)
                elif type_of_operation == "remove":
                    self.remove_cart(cart_id, prod, quantity)
            # Finalize the transaction and receive a manifest of successfully purchased items.
            p_purchased = self.marketplace.place_order(cart_id)
            for prod in p_purchased:
                print(f"{self.getName()} bought {prod}")

    def add_cart(self, cart_id, product_id, quantity):
        """
        Attempts to move items from the marketplace pool to a specific cart.
        Algorithm: Iterative try-wait loop ensuring the desired quantity is fulfilled.
        """
        for _ in range(quantity):
            while True:
                added = self.marketplace.add_to_cart(cart_id, product_id)
                if added:
                    break
                # Functional Utility: Implements a fixed-interval backoff to prevent busy-waiting.
                sleep(self.retry_wait_time)

    def remove_cart(self, cart_id, product_id, quantity):
        """
        Restores items from a cart back to the marketplace availability pool.
        Algorithm: Iterative try-wait loop for quantity reversal.
        """
        for _ in range(quantity):
            while True:
                removed = self.marketplace.remove_from_cart(cart_id, product_id)
                if removed:
                    break
                sleep(self.retry_wait_time)


from threading import Lock
import unittest
import sys
sys.path.insert(1, './tema')
import product as produs

class Marketplace:
    """
    The central coordinator managing product visibility and thread-safe transactions.
    Functional Utility: Acts as a broker between producers (supply) and consumers (demand) 
    using per-producer queues and per-consumer carts.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        @param queue_size_per_producer: Maximum buffer size for each producer's supply line.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        self.queues = []
        self.carts = []
        self.mutex = Lock()
        # Mapping from product instances to their originating producer queue index.
        self.products_dict = {}

    def register_producer(self):
        """
        Registers a new supply entity.
        Logic: Atomically assigns a unique ID and initializes a corresponding supply queue.
        """
        self.mutex.acquire()
        producer_id = self.producer_id
        self.producer_id += 1
        self.queues.append([])
        self.mutex.release()
        return str(producer_id)

    def publish(self, producer_id, product):
        """
        Accepts a product from a producer and places it in the available supply.
        @return: True if successful, False if the producer's buffer is full.
        """
        index_prod = int(producer_id)
        if len(self.queues[index_prod]) == self.queue_size_per_producer:
            return False
        self.queues[index_prod].append(product)
        self.products_dict[product] = index_prod
        return True

    def new_cart(self):
        """
        Creates a new shopping cart for a consumer.
        Logic: Generates a unique cart ID and initializes its storage list.
        """
        self.mutex.acquire()
        cart_id = self.cart_id
        self.cart_id += 1
        self.mutex.release()
        self.carts.append([])
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers a product from any available producer queue to a consumer's cart.
        Logic: Implements a global search across all producer buffers.
        @return: True if item was found and moved, False otherwise.
        """
        prod_in_queue = False
        for queue in self.queues:
            if product in queue:
                prod_in_queue = True
                queue.remove(product)
                break
        if not prod_in_queue:
            return False
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to its original producer queue.
        Precondition: The original producer queue must have space available.
        @return: True if the operation succeeded, False if cart doesn't have item or queue is full.
        """
        if product not in self.carts[cart_id]:
            return False
        index_producer = self.products_dict[product]
        if len(self.queues[index_producer]) == self.queue_size_per_producer:
            return False
        self.carts[cart_id].remove(product)
        self.queues[index_producer].append(product)
        return True

    def place_order(self, cart_id):
        """
        Finalizes the shopping session.
        Logic: Flushes the cart and returns the list of purchased products.
        """
        cart_product_list = self.carts[cart_id]
        self.carts[cart_id] = []
        return cart_product_list

class TestMarketplace(unittest.TestCase):
    """
    Unit testing suite for validating the marketplace's state transition logic.
    """
    
    def setUp(self):
        self.marketplace = Marketplace(4)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertEqual(self.marketplace.register_producer(), str(2))
        self.assertNotEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertNotEqual(self.marketplace.register_producer(), str(2))

    def test_publish(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertNotEqual(self.marketplace.new_cart(), 3)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertNotEqual(self.marketplace.new_cart(), 0)
        self.assertNotEqual(self.marketplace.new_cart(), 3)
        self.assertNotEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.assertTrue(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")))

    def test_remove_from_cart(self):
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))
        self.assertTrue(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal")))

    def test_place_order(self):
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))
        self.assertEqual([produs.Tea("Linden", 9, "Herbal")], self.marketplace.place_order(0))


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    Simulates a manufacturing unit providing goods to the marketplace.
    Functional Utility: Manages the continuous production cycle of items and 
    handles publication bottlenecks.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the producer.
        @param products: List of product specifications (product, quantity, time).
        @param marketplace: The central trading hub.
        @param republish_wait_time: Interval to wait when the supply queue is full.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Self-registration upon initialization to secure a supply line.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Infinite production loop.
        Logic: Cycles through its product catalog and attempts to publish items 
        according to specified quantities.
        """
        while True:
            for product in self.products:
                quantity = product[1]
                for _ in range(0, quantity):
                    self.publish_product(product[0], product[2])

    def publish_product(self, product, production_time):
        """
        Executes the publication of a single unit.
        Logic: Incorporates production time and handles marketplace congestion 
        via a backoff strategy.
        """
        while True:
            published = self.marketplace.publish(self.producer_id, product)
            if published:
                # Simulation of manufacturing overhead.
                sleep(production_time)
                break
            # Backoff: wait for consumer demand to clear queue space.
            sleep(self.republish_wait_time)
