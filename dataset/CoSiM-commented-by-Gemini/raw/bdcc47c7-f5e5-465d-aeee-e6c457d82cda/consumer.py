"""
Module providing a multi-threaded simulation of a retail marketplace.
Uses specialized worker threads (Producers and Consumers) coordinated by a 
central Marketplace entity that ensures data integrity through explicit locking mechanisms.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents an automated shopper thread that executes a predefined shopping list.
    Orchestrates the acquisition and removal of products across multiple shopping sessions (carts).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer with shopping tasks.
        
        Args:
            carts (list): List of operations per cart (e.g., [{'product': p, 'type': 'add', 'quantity': q}]).
            marketplace (Marketplace): The central broker for transactions.
            retry_wait_time (float): Seconds to wait if a desired product is out of stock.
            **kwargs: Thread configuration options (requires 'name').
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Executes the shopping lifecycle for each assigned cart.
        Logic: Sequential processing of operations (add/remove) with busy-wait retries for stock availability.
        """
        # Block Logic: Iterates through each independent shopping session.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            # Block Logic: Processes individual product operations within the current cart.
            for ops in cart:
                if ops['type'] == "add":
                    for _ in range(0, ops['quantity']):
                        # Block Logic: Indefinite retry loop for product acquisition.
                        # Invariant: Consumer waits for the marketplace to fulfill the request before moving to the next item.
                        while self.marketplace.add_to_cart(cart_id, ops['product']) is not True:
                            sleep(self.retry_wait_time)
                else:
                    # Action: Return reserved products back to the marketplace.
                    for _ in range(0, ops['quantity']):
                        self.marketplace.remove_from_cart(cart_id, ops['product'])
            
            # Finalization: Converts the cart contents into a finalized transaction log.
            products = self.marketplace.place_order(cart_id)

            # Functional Utility: synchronized console output using the shared consumer lock.
            lock = self.marketplace.get_consumer_lock()
            lock.acquire()
            for product in products:
                print(self.kwargs['name'] + " bought " + str(product))
            lock.release()


import logging
from logging.handlers import RotatingFileHandler
import time
from threading import Lock
import unittest

# Telemetry Configuration: Rotating logs for audit trails.
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO,
                    handlers=[RotatingFileHandler('marketplace.log',
                                                  maxBytes=20000, backupCount=10)])
logging.Formatter.converter = time.gmtime

class Marketplace:
    """
    Acts as a thread-safe mediator between Producers and Consumers.
    Manages global product availability, producer registration, and consumer cart state.
    Utilizes fine-grained locks to protect internal registries while minimizing contention.
    """

    def __init__(self, queue_size_per_producer):
        """
        Args:
            queue_size_per_producer (int): Maximum items allowed in a producer's public listing.
        """
        self.queue_size_per_producer = queue_size_per_producer
        
        # Synchronization: Separate locks for producer and consumer logic paths.
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        
        self.producer_id = -1
        self.cart_id = -1
        
        # Data Structures: Tracking occupancy and availability.
        self.size_per_producer = {} # producer_id -> count
        self.carts = {} # cart_id -> {product -> [producer_id, ...]}
        self.products_dict = {} # product -> [producer_id, ...] (Global availability list)

    def register_producer(self):
        """
        Registers a new manufacturing source in the system.
        
        Returns:
            int: Unique identifier for the producer.
        """
        self.producer_lock.acquire()
        logging.info("New producer entered register_producer method")
        self.producer_id += 1
        self.size_per_producer[self.producer_id] = 0
        self.producer_lock.release()
        logging.info("New producer registered with id %d", self.producer_id)
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Lists a product in the marketplace inventory, respecting quota limits.
        
        Args:
            producer_id (int): source of the item.
            product: The item object to list.
            
        Returns:
            bool: True if listing succeeded, False if producer buffer is full.
        """
        logging.info("Producer with id %d entered publish method", producer_id)

        self.producer_lock.acquire()
        
        # Quota Enforcement: Prevents overflow of the producer's allotted queue space.
        if self.size_per_producer[producer_id] == self.queue_size_per_producer:
            logging.info(f"Producer with id {producer_id} failed to publish product {product}")
            self.producer_lock.release()
            return False
            
        # Logic: Append producer ID to the product's availability queue (FIFO).
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
        Allocates a new transaction context for a consumer.
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
        Transfers an item from global availability to a specific consumer cart.
        
        Returns:
            bool: Success status based on item availability.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered add_to_cart method", cart_id)
        
        # Block Logic: Atomic acquisition.
        if product in self.products_dict:
            # Logic: Pull from the global availability pool and assign to the cart.
            producer_id = self.products_dict[product].pop(0)
            if product in self.carts[cart_id]:
                self.carts[cart_id][product].append(producer_id)
            else:
                self.carts[cart_id][product] = [producer_id]
            
            # Maintenance: Remove key if no more instances are available.
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
        Returns a reserved item back to the global marketplace.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered remove_from_cart method", cart_id)

        # Logic: Extract from cart and return to the global availability queue.
        given_id = self.carts[cart_id][product].pop(0)
        if len(self.carts[cart_id][product]) == 0:
            del self.carts[cart_id][product]

        if product not in self.products_dict:
            self.products_dict[product] = [given_id]
        else:
            self.products_dict[product].append(given_id)
        logging.info(f"Consumer with card id {cart_id} removed product {product} from cart")
        self.consumer_lock.release()

    def place_order(self, cart_id):
        """
        Finalizes the transaction by clearing the cart and updating producer stock levels.
        
        Returns:
            list: Collection of purchased products.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered place_order method", cart_id)
        
        # Block Logic: Final accounting.
        # Logic: Iterates through cart contents, decrementing producer quotas permanently.
        products = []
        for product in self.carts[cart_id]:
            for given_id in self.carts[cart_id][product]:
                self.size_per_producer[given_id] -= 1
                products.append(product)
        
        logging.info("Consumer with card id %d placed order", cart_id)
        self.consumer_lock.release()
        return products

    def get_consumer_lock(self):
        """
        Functional Utility: Accessor for the consumer lock to coordinate external actions (like printing).
        """
        logging.info("A consumer entered get_consumer_lock method")
        return self.consumer_lock


class TestMarketplace(unittest.TestCase):
    """
    Unit test suite for validating Marketplace operations.
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
    Represents a manufacturing source thread that supplies products to the marketplace.
    Operates in a continuous cycle based on its production capacity and item-specific costs.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Args:
            products (list): Collection of (product, quantity, production_time) tuples.
            marketplace (Marketplace): Target for product listings.
            republish_wait_time (float): Back-pressure delay if marketplace is full.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        Continuous production loop.
        Logic: Sequential manufacturing of assigned product batches with congestion-aware backoff.
        """
        producer_id = self.marketplace.register_producer()
        while True:
            # Block Logic: Iterates through the production catalog.
            for product in self.products:
                for _ in range(0, product[1]):
                    # Action: Attempt to list the item.
                    # Invariant: Retries until the marketplace accepting the listing.
                    while self.marketplace.publish(producer_id, product[0]) is not True:
                        sleep(self.republish_wait_time)
                    # Logic: Production time represents the physical cost of generating the item.
                    sleep(product[2])
