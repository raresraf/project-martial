
"""
@file __init__.py
@brief Concurrent marketplace simulation implementing the Producer-Consumer pattern.

Functional Intent: Provides a thread-safe environment for multiple Producers to 
publish products and Consumers to acquire them via virtual shopping carts. 
Features multi-level locking for independent shared resources (Producers, 
Consumers, and Inventory Queue). Implements a soft-reservation strategy where 
items are marked as 'unavailable' in the marketplace upon being added to a cart 
but only physically removed from the producer's quota during the final order placement.

Domain: Production Systems, Concurrency and Synchronization.
"""


from threading import Thread
import time
import sys


class Consumer(Thread):
    """
    @brief Represents a consumer entity that operates in its own execution thread.
    
    Logic: Iteratively processes assigned shopping carts, performing batch 'add' 
    and 'remove' operations with polling-based retry for inventory acquisition.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with specific shopping tasks.
        @param carts List of carts containing operation requests.
        @param marketplace Reference to the central Marketplace instance.
        @param retry_wait_time Interval to wait when the marketplace is depleted.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        @brief Core execution loop for the consumer thread.
        
        Algorithm: Sequential cart processing with nested polling retries.
        """
        for cart in self.carts:
            # Block Logic: Session initialization.
            cart_id = self.marketplace.new_cart()

            for action in cart:
                action_type = action['type']
                product = action['product']
                quantity = action['quantity']

                if action_type == "add":
                    # Block Logic: Persistent acquisition loop.
                    for _ in range(quantity):
                        # Invariant: Retries acquisition until the marketplace grants the reservation.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            # Optimization: Back-off wait to prevent CPU thrashing.
                            time.sleep(self.retry_wait_time)
                else:
                    # Block Logic: Batch item restoration pass.
                    for _ in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

            # Functional Intent: Finalize order and log results to standard output.
            for order in self.marketplace.place_order(cart_id):
                sys.stdout.flush()
                print(f"{self.name} bought {order}")


from threading import Lock
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Counter
import unittest
from tema.product import Coffee, Tea


def set_logger():
    """
    @brief Configures a rotating file logger for marketplace activity tracking.
    """
    formatter = logging.Formatter(
        '[%(asctime)s] --> %(levelname)s: %(message)s')
    formatter.converter = time.gmtime

    handler = RotatingFileHandler(
        'marketplace.log', maxBytes=100000, backupCount=10)
    handler.setFormatter(formatter)

    logger = logging.getLogger('marketplace info logger')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


class Marketplace:
    """
    @brief Central coordinator for thread-safe item publishing and transactions.
    
    Functional Utility: Manages producer quotas, consumer sessions, and 
    global inventory. Utilizes discrete mutexes for distinct data planes to 
    maximize concurrency and prevent race conditions.
    """
    
    logger = set_logger()

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace with capacity limits.
        """
        self.logger.info(
            "Marketplace initialized with maximum queue size: %s.", queue_size_per_producer)

        self.queue_size_per_producer = queue_size_per_producer

        self.producer_number = 0
        self.consumer_cart_number = 0

        self.producers_products = {} # Logic: Tracks current item count per producer ID.
        self.consumers_carts = {} # Logic: Maps cart IDs to reserved item indices.
        self.products_queue = {} # Logic: Primary inventory map (ProductName -> List of (ProducerID, AvailableFlag)).

        # Synchronization: Discrete locks for high-granularity concurrency control.
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.queue_lock = Lock()

        self.logger.info("Initiated marketplace parameters.")

    def register_producer(self):
        """
        @brief Onboards a new producer and assigns a unique identifier.
        """
        self.logger.info("A new producer tries to obtain an id.")

        # Synchronization: Critical section for global producer count.
        with self.producer_lock:
            producer_id = self.producer_number
            self.producer_number = self.producer_number + 1

        self.producers_products[producer_id] = 0

        self.logger.info(
            "Generated the producer id number %s.", producer_id)

        return str(producer_id)

    def publish(self, producer_id, product):
        """
        @brief Exposes a product to the marketplace if the producer's quota allows.
        """
        self.logger.info(
            "Producer with id %s wants to publish %s.", producer_id, product)

        producer_id = int(producer_id)

        # Block Logic: Quota verification.
        if self.producers_products[producer_id] >= self.queue_size_per_producer:
            self.logger.info("Reached max queue for producer %s on product %s.", producer_id, product)
            return False

        self.producers_products[producer_id] += 1

        # Synchronization: Protects the shared inventory mapping.
        with self.queue_lock:
            if not product in self.products_queue:
                self.products_queue[product] = []

            # Invariant: Items are initially marked as 'Available' (True).
            self.products_queue[product].append((producer_id, True))

        self.logger.info(
            "Producer with id %s published %s.", producer_id, product)

        return True

    def new_cart(self):
        """
        @brief Allocates a new shopping cart identifier for a consumer.
        """
        self.logger.info("A consumer tries to obtain a new cart id.")

        # Synchronization: Mutex for thread-safe cart ID generation.
        with self.consumer_lock:
            consumer_cart_id = self.consumer_cart_number
            self.consumer_cart_number = self.consumer_cart_number + 1

        self.consumers_carts[consumer_cart_id] = {}

        self.logger.info(
            "Generated the cart id number %s.", consumer_cart_id)

        return consumer_cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Performs a soft-reservation of a product for a specific consumer.
        
        Algorithm: First-available scanning within the product list.
        Logic: Atomically flips the availability flag of an item in the global 
        queue and temporarily releases the producer's quota slot.
        """
        self.logger.info(
            "Consumer with cart id %s wants to add %s.", cart_id, product)

        # Synchronization: Mutual exclusion for inventory status modification.
        with self.queue_lock:
            if not product in self.products_queue:
                self.logger.info("Inexistent product %s for cart %s.", product, cart_id)
                return False

            # Block Logic: Search for an unreserved instance of the product.
            for index in range(len(self.products_queue[product])):
                product_queue = self.products_queue[product][index]

                if product_queue[1] is True:
                    # Logic: Successfully reserved an item. Mark as unavailable.
                    self.products_queue[product][index] = (
                        product_queue[0], False)

                    if not product in self.consumers_carts[cart_id]:
                        self.consumers_carts[cart_id][product] = []

                    self.consumers_carts[cart_id][product].append(index)

                    # Logic: Optimistically releases the producer's quota slot 
                    # as the item has left their active pool.
                    with self.producer_lock:
                        self.producers_products[self.products_queue[product]
                                                [index][0]] -= 1

                    self.logger.info(
                        "Consumer with cart id %s added %s.", cart_id, product)

                    return True

        self.logger.info("Product %s currently unavailable for cart %s.", product, cart_id)
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Cancels a reservation and restores the product to the global pool.
        """
        self.logger.info(
            "Consumer with cart id %s wants to remove %s.", cart_id, product)

        with self.queue_lock:
            if len(self.consumers_carts[cart_id][product]) == 0:
                raise Exception("No product to be removed from cart")

            # Block Logic: Restoration pass.
            index = self.consumers_carts[cart_id][product].pop()
            product_queue = self.products_queue[product][index]
            
            # Logic: Restore availability flag.
            self.products_queue[product][index] = (product_queue[0], True)

            # Logic: Re-consume the producer's quota slot.
            with self.producer_lock:
                self.producers_products[self.products_queue[product]
                                        [index][0]] += 1

        self.logger.info(
            "Consumer with cart id %s removed %s.", cart_id, product)

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and clears the session state.
        """
        self.logger.info(
            "Consumer with cart id %s wants to place order.", cart_id)

        order = []
        consumer_cart = self.consumers_carts[cart_id]
        
        # Block Logic: Aggregation of reserved items.
        for product in consumer_cart.keys():
            for _ in consumer_cart[product]:
                order.append(product)

        # Finalization: Deletes the cart session.
        self.consumers_carts[cart_id] = {}

        self.logger.info(
            "Consumer with cart id %s placed order: %s.", cart_id, order)

        return order


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for verifying thread-safe marketplace logic.
    """
    def setUp(self):
        queue_size_per_producer = 5
        self.marketplace = Marketplace(queue_size_per_producer)
        self.teas = []
        self.teas.append(Tea("Tabiets", 5, "Black"))
        self.teas.append(Tea("Aroma Tea", 7.5, "Mint"))
        self.teas.append(Tea("Honey", 5, "Green"))
        self.coffees = []
        self.coffees.append(Coffee("Davidoff", 10, 4.5, "STRONG"))
        self.coffees.append(Coffee("Romantique", 6, 3.0, "MILD"))
        self.coffees.append(Coffee("Costa", 8, 5.0, "EXTRA STRONG"))

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(),
                         '0', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(),
                         '1', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(),
                         '2', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(),
                         '3', "Incorrect producer id.")

    def test_publish(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.assertTrue(self.marketplace.publish(
            '0', self.teas[0]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '0', self.teas[1]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '0', self.teas[0]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '0', self.teas[2]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '0', self.coffees[0]), "Did not publish coffee")
        self.assertFalse(self.marketplace.publish(
            '0', self.coffees[1]), "Reached max queue")
        self.marketplace.add_to_cart(0, self.coffees[0])
        self.assertTrue(self.marketplace.publish(
            '0', self.coffees[1]), "Did not publish coffee")
        self.assertTrue(self.marketplace.publish(
            '1', self.teas[2]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '1', self.coffees[2]), "Did not publish coffee")

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 0, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 1, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 2, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 3, "Incorrect cart id.")

    def test_add_to_cart(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(
            0, self.coffees[1]), "Inexistent coffee")
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])
        self.assertTrue(self.marketplace.add_to_cart(
            1, self.teas[0]), "Did not add tea")
        self.assertFalse(self.marketplace.add_to_cart(
            0, self.teas[0]), "Inexistent tea")
        self.marketplace.publish('1', self.coffees[1])
        self.marketplace.publish('1', self.coffees[2])
        self.assertFalse(self.marketplace.add_to_cart(
            1, self.coffees[0]), "Inexistent coffee")
        self.assertTrue(self.marketplace.add_to_cart(
            0, self.coffees[1]), "Did not add coffee")
        self.assertTrue(self.marketplace.add_to_cart(
            0, self.coffees[2]), "Did not add coffee")

    def test_remove_from_cart(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.assertRaises(Exception, self.marketplace.remove_from_cart,
                          0, self.coffees[0], "No product to be removed from cart")
        self.marketplace.publish('1', self.coffees[0])
        self.marketplace.add_to_cart(1, self.coffees[0])
        self.assertFalse(self.marketplace.add_to_cart(
            0, self.coffees[0]), "Inexistent coffee")
        self.marketplace.remove_from_cart(1, self.coffees[0])
        self.assertTrue(self.marketplace.add_to_cart(
            0, self.coffees[0]), "Did not add coffee")
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])
        self.marketplace.publish('0', self.teas[2])
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])
        self.marketplace.add_to_cart(0, self.teas[0])
        self.marketplace.publish('0', self.teas[2])
        self.marketplace.add_to_cart(0, self.teas[1])
        self.marketplace.remove_from_cart(0, self.teas[1])
        self.assertFalse(self.marketplace.publish(
            '0', self.coffees[0]), "Reached max queue")

    def test_place_order(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('1', self.coffees[0])
        self.marketplace.publish('0', self.teas[2])
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('1', self.coffees[1])
        self.marketplace.add_to_cart(0, self.teas[0])
        self.marketplace.add_to_cart(0, self.coffees[1])
        self.assertEqual(Counter(self.marketplace.place_order(0)),
                         Counter([self.teas[0], self.coffees[1]]))
        self.marketplace.add_to_cart(0, self.coffees[0])
        self.assertEqual(Counter(self.marketplace.place_order(0)),
                         Counter([self.coffees[0]]))
        self.marketplace.add_to_cart(1, self.teas[0])
        self.marketplace.remove_from_cart(1, self.teas[0])
        self.assertEqual(self.marketplace.place_order(1), [])


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    @brief Represents a producer entity that generates items for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with catalog and manufacturing schedule.
        """
        Thread.__init__(self)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        
        self.producer_id = self.marketplace.register_producer()
        self.daemon = True

    def run(self):
        """
        @brief Core manufacturing loop.
        
        Logic: Continuously iterates through catalog, manufacturing items with 
        specified delays and implementing back-off if the marketplace is full.
        """
        while True:
            for product in self.products:
                
                product_data = product[0]
                quantity = product[1]
                wait_time = product[2]

                # Block Logic: Batch production.
                for _ in range(quantity):
                    # Invariant: Polls until the marketplace accepts the item for publication.
                    while not self.marketplace.publish(self.producer_id, product_data):
                        # Optimization: Back-off wait.
                        sleep(self.republish_wait_time)

                # Optimization: Simulation of device-level production cycle time.
                sleep(wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base immutable representation of a market item.
    """
    name: str
    price: int

    def __hash__(self):
        return hash((self.name, self.price))

    def __eq__(self, other):
        return (self.name, self.price) == (other.name, other.price)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product type for tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product type for coffee.
    """
    acidity: str
    roast_level: str
