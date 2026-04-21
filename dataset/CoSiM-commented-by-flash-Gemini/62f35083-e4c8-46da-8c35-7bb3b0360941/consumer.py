"""
@62f35083-e4c8-46da-8c35-7bb3b0360941/consumer.py
@brief Multi-threaded marketplace simulation implementing a producer-consumer pattern.
Functional Utility: Orchestrates a concurrent environment where multiple Producers 
publish items and Consumers purchase them via a centralized Marketplace mediator. 
Includes a comprehensive unit testing suite for behavioral validation.
Domain: Concurrent Systems, Design Patterns (Mediator).
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    @brief Consumer entity that interacts with the marketplace to fulfill shopping carts.
    Functional Utility: Iterates through a set of predefined shopping tasks, performing 
    blocking 'add' and 'remove' operations with exponential backoff on failure.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.
        @param carts List of carts, where each cart is a list of operation dictionaries.
        @param marketplace Central marketplace instance.
        @param retry_wait_time Duration to wait before retrying a failed marketplace operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_product(self, cart_id, product, quantity):
        """
        Block Logic: Iteratively attempts to add a specific product to the cart.
        Invariant: Continues retrying until the requested quantity is successfully reserved.
        """
        added = 0
        while added < quantity:
            status = self.marketplace.add_to_cart(cart_id, product)
            if status:
                added += 1
            else:
                time.sleep(self.retry_wait_time)

    def remove_product(self, cart_id, product, quantity):
        """
        Block Logic: Removes a specified quantity of a product from the consumer's cart.
        """
        removed = 0
        while removed < quantity:
            status = self.marketplace.remove_from_cart(cart_id, product)
            if status:
                removed += 1

    def run(self):
        """
        @brief Main execution lifecycle for the consumer.
        Logic: Spawns new carts, executes all planned operations (add/remove), 
        and finalizes orders.
        """
        for carts_elem in self.carts:
            cart_id = self.marketplace.new_cart()
            for elem in carts_elem:
                op_type = elem.get("type")
                product = elem.get("product")
                quantity = elem.get("quantity")
                if op_type == 'add':
                    self.add_product(cart_id, product, quantity)
                elif op_type == 'remove':
                    self.remove_product(cart_id, product, quantity)
            items = self.marketplace.place_order(cart_id)
            for item in items:
                print(f"{self.name} bought {item}")

import random
from threading import Lock
import unittest
import logging
from logging import handlers

from tema.product import Tea, Coffee


class TestMarketPlace(unittest.TestCase):
    """
    @brief Unit testing suite for the Marketplace coordination logic.
    Functional Utility: Validates thread-safe ID generation, product publishing 
    constraints, and cart manipulation logic.
    """

    def setUp(self):
        """
        Pre-condition: Fresh Marketplace instance initialized with a fixed queue capacity.
        """
        self.marketplace = Marketplace(15)

    def test_register_producer(self):
        """
        @brief Validates atomic producer registration and ID assignment.
        """
        self.assertEqual(self.marketplace.producer_id_generator, 0,
                         "Wrong initial producer id generator value")
        self.assertEqual(self.marketplace.register_producer(), 1,
                         "Generated wrong id for producer 1")
        self.assertIn(1, self.marketplace.producers.keys(),
                      "Producer with id 1 not in producers dictionary")
        self.assertEqual(self.marketplace.register_producer(), 2,
                         "Generated wrong id for producer 2")
        self.assertIn(2, self.marketplace.producers.keys(),
                      "Producer with id 2 not in producers dictionary")
        self.assertNotIn(3, self.marketplace.producers.keys(),
                         "Producer with id 3 not generated but in producers dictionary")

    def test_new_cart(self):
        """
        @brief Validates thread-safe cart initialization.
        """
        self.assertEqual(self.marketplace.cart_id_generator, 0,
                         "Wrong initial cart id generator value")
        self.assertEqual(self.marketplace.new_cart(), 1,
                         "Generated wrong id for cart 1")
        self.assertIn(1, self.marketplace.carts.keys(),
                      "Cart with id 1 not in carts dictionary")
        self.assertEqual(self.marketplace.new_cart(), 2,
                         "Generated wrong id for cart 2")
        self.assertIn(2, self.marketplace.carts.keys(),
                      "Cart with id 2 not in carts dictionary")
        self.assertNotIn(3, self.marketplace.carts.keys(),
                         "Cart with id 3 not generated but in carts dictionary")

    def test_publish(self):
        """
        @brief Validates producer product publishing with capacity constraints.
        """
        self.assertEqual(len(self.marketplace.producers), 0,
                         "Wrong producers number")
        self.assertEqual(self.marketplace.register_producer(), 1,
                         "Generated wrong id for producer 1")
        self.assertEqual(self.marketplace.register_producer(), 2,
                         "Generated wrong id for producer 2")
        self.assertFalse(self.marketplace.publish(3, None),
                         "Producer with id 3 does not exist")

        tea = Tea(name='Linden', price=9, type='Herbal')
        coffee = Coffee(name='Indonesia', price=1,
                        acidity='5.05', roast_level='MEDIUM')

        self.assertTrue(self.marketplace.publish(1, tea),
                        "Problem adding product for producer 1")
        self.assertTrue(self.marketplace.publish(1, coffee),
                        "Problem adding product for producer 1")
        self.assertTrue(self.marketplace.publish(2, coffee),
                        "Problem adding product for producer 2")
        self.assertFalse(self.marketplace.publish(3, coffee),
                         "Adding product for nonexistent producer")

        self.assertIn(tea, self.marketplace.producers.get(1),
                      "Tea does not exist in producer's 1 list")
        self.assertIn(coffee, self.marketplace.producers.get(1),
                      "Coffee does not exist in producer's 1 list")
        self.assertIn(coffee, self.marketplace.producers.get(2),
                      "Coffee does not exist in producer's 2 list")

        self.assertEqual(len(self.marketplace.producers.get(1)), 2,
                         "More/less elements in producer 1 list")
        self.assertEqual(len(self.marketplace.producers.get(2)), 1,
                         "More/less elements in producer 1 list")

        test_id = self.marketplace.register_producer()
        for _ in range(15):
            self.assertTrue(self.marketplace.
                            publish(test_id, random.choice([tea, coffee])),
                            "Problem adding new product")
        self.assertFalse(self.marketplace.
                         publish(test_id, random.choice([tea, coffee])),
                         "Problem adding more products than the queue capacity")

    def test_add_to_cart(self):
        """
        @brief Validates item reservation from producers to consumer carts.
        """
        producer_id1 = self.marketplace.register_producer()
        producer_id2 = self.marketplace.register_producer()
        self.assertEqual(producer_id1, 1, "Generated wrong id for producer 1")
        self.assertEqual(producer_id2, 2, "Generated wrong id for producer 2")

        tea = Tea(name='Linden', price=9, type='Herbal')
        coffee = Coffee(name='Indonesia', price=1,
                        acidity='5.05', roast_level='MEDIUM')

        cart_id1 = self.marketplace.new_cart()
        cart_id2 = self.marketplace.new_cart()
        self.assertEqual(cart_id1, 1, "Generated wrong id for cart 1")
        self.assertEqual(cart_id2, 2, "Generated wrong id for cart 2")

        self.marketplace.publish(producer_id1, tea)
        self.marketplace.publish(producer_id1, tea)
        self.marketplace.publish(producer_id1, tea)
        self.marketplace.publish(producer_id2, coffee)
        self.marketplace.publish(producer_id2, coffee)

        self.assertTrue(self.marketplace.add_to_cart(cart_id1, tea),
                        "Error adding produce to cart 1")
        self.assertTrue(self.marketplace.add_to_cart(cart_id1, tea),
                        "Error adding produce to cart 1")
        self.assertTrue(self.marketplace.add_to_cart(cart_id1, coffee),
                        "Error adding produce to cart 1")
        self.assertEqual(len(self.marketplace.carts.get(cart_id1)), 3,
                         "More/less elements in cart 1 list")

        self.assertTrue(self.marketplace.add_to_cart(cart_id2, tea),
                        "Error adding produce to cart 1")
        self.assertTrue(self.marketplace.add_to_cart(cart_id2, coffee),
                        "Error adding produce to cart 1")
        self.assertEqual(len(self.marketplace.carts.get(cart_id2)), 2,
                         "More/less elements in cart 1 list")

        self.assertFalse(self.marketplace.add_to_cart(cart_id2, tea),
                         "Added to cart nonexistent element")

        self.marketplace.publish(producer_id2, coffee)
        self.assertFalse(self.marketplace.add_to_cart(3, coffee),
                         "Add element to nonexistent cart")

        new_coffee = Coffee(name='India', price=2,
                            acidity='5.05', roast_level='ROASTED')
        self.assertFalse(self.marketplace.add_to_cart(1, new_coffee),
                         "Add nonexistent element to cart")

    def test_remove_from_cart(self):
        """
        @brief Validates item return logic from carts back to producer queues.
        """
        producer_id1 = self.marketplace.register_producer()
        producer_id2 = self.marketplace.register_producer()

        tea = Tea(name='Linden', price=9, type='Herbal')
        coffee = Coffee(name='Indonesia', price=1,
                        acidity='5.05', roast_level='MEDIUM')

        cart_id1 = self.marketplace.new_cart()
        cart_id2 = self.marketplace.new_cart()

        self.marketplace.publish(producer_id1, tea)
        self.marketplace.publish(producer_id1, tea)
        self.marketplace.publish(producer_id1, coffee)
        self.marketplace.publish(producer_id2, coffee)
        self.marketplace.publish(producer_id2, coffee)
        self.marketplace.publish(producer_id2, tea)

        self.marketplace.add_to_cart(cart_id1, tea)
        self.marketplace.add_to_cart(cart_id1, coffee)
        self.marketplace.add_to_cart(cart_id1, coffee)
        self.marketplace.add_to_cart(cart_id2, tea)
        self.marketplace.add_to_cart(cart_id2, tea)

        self.assertTrue(self.marketplace.remove_from_cart(cart_id1, tea),
                        "Error removing product from cart 1")
        self.assertTrue(self.marketplace.remove_from_cart(cart_id1, coffee),
                        "Error removing product from cart 1")
        self.assertTrue(self.marketplace.remove_from_cart(cart_id1, coffee),
                        "Error removing product from cart 1")
        self.assertEqual(len(self.marketplace.carts.get(cart_id1)), 0,
                         "Error removing all products from cart 1")

        self.assertTrue(self.marketplace.remove_from_cart(cart_id2, tea),
                        "Error removing product from cart 2")
        self.assertTrue(self.marketplace.remove_from_cart(cart_id2, tea),
                        "Error removing product from cart 2")
        self.assertEqual(len(self.marketplace.carts.get(cart_id2)), 0,
                         "Error removing all products from cart 2")

        self.assertFalse(self.marketplace.remove_from_cart(3, tea),
                         "Error removing product from nonexistent cart")

        new_coffee = Coffee(name='India', price=2,
                            acidity='5.05', roast_level='ROASTED')
        self.assertFalse(self.marketplace.remove_from_cart(1, new_coffee),
                         "Error removing nonexistent product from cart")

    def test_place_order(self):
        """
        @brief Validates order finalization and item retrieval.
        """
        cart_id = self.marketplace.new_cart()
        self.assertEqual(len(self.marketplace.place_order(cart_id)), 0,
                         "Error ordering nonexistent items")

        tea = Tea(name='Linden', price=9, type='Herbal')
        coffee = Coffee(name='Indonesia', price=1,
                        acidity='5.05', roast_level='MEDIUM')

        test_id = self.marketplace.register_producer()
        for i in range(15):
            if i % 2 == 0:
                self.assertTrue(self.marketplace.publish(test_id, tea),
                                "Problem adding new product")
            else:
                self.assertTrue(self.marketplace.publish(test_id, coffee),
                                "Problem adding new product")
        for i in range(7):
            self.assertTrue(self.marketplace.
                            add_to_cart(cart_id, random.choice([tea, coffee])),
                            "Problem adding product to cart")

        self.assertEqual(len(self.marketplace.place_order(cart_id)), 7,
                         "Error placing order")


class Marketplace:
    """
    @brief Central mediator for managing producers, consumers, and inventory.
    Functional Utility: Provides thread-safe primitives for registering producers, 
    publishing products, and managing consumer carts. Uses coarse-grained locking 
    to ensure data consistency in a concurrent environment.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.
        @param queue_size_per_producer Maximum capacity for each producer's internal buffer.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id_generator = 0
        self.cart_id_generator = 0
        self.carts = {} # Maps cart_id to list of [producer_id, product] pairs.
        self.producers = {} # Maps producer_id to list of available products.
        self.producers_lock = Lock() # Mutex for producer-side operations.
        self.consumers_lock = Lock() # Mutex for consumer-side operations.

        # Functional Utility: Configures audit logging for all marketplace transitions.
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                handlers.RotatingFileHandler("marketplace.log",
                                                             maxBytes=1000000,
                                                             backupCount=10)],
                            format='%(asctime)s %(levelname)s : %(message)s')

    def register_producer(self):
        """
        @brief Registers a new producer and assigns a unique ID.
        Functional Utility: Atomically increments the ID generator and initializes a producer buffer.
        """
        logging.info("Entered method %s", self.register_producer.__name__)
        self.producers_lock.acquire()
        self.producer_id_generator += 1
        self.producers[self.producer_id_generator] = []
        self.producers_lock.release()
        logging.info("Leaved method %s and returned %d",
                     self.register_producer.__name__,
                     self.producer_id_generator)
        return self.producer_id_generator

    def publish(self, producer_id, product):
        """
        @brief Adds a product to a producer's queue if capacity allows.
        Functional Utility: Implements a bounded buffer check before accepting new products.
        """
        logging.info('Entered method %s '
                     'with arguments producer_id = %d, product = %s',
                     self.publish.__name__, producer_id, product)
        self.producers_lock.acquire()
        products = self.producers.get(producer_id)
        if products is not None:
            if len(products) < self.queue_size_per_producer:
                self.producers.get(producer_id).append(product)
                self.producers_lock.release()
                logging.info(
                    'Leaved method %s and returned '
                    '%s', self.publish.__name__, True)
                return True

        self.producers_lock.release()
        logging.info(
            'Leaved method %s and returned '
            '%s', self.publish.__name__, False)
        return False

    def new_cart(self):
        """
        @brief Allocates a new shopping cart for a consumer.
        """
        logging.info(
            'Entered method %s ', self.new_cart.__name__)

        self.consumers_lock.acquire()
        self.cart_id_generator += 1
        self.carts[self.cart_id_generator] = []
        logging.info(
            'Leaved method %s and returned '
            '%d', self.new_cart.__name__, self.cart_id_generator)
        self.consumers_lock.release()
        return self.cart_id_generator

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product from any producer's queue to a specific consumer cart.
        Functional Utility: Implements the 'checkout' phase where products are removed 
        from global availability and reserved for a specific consumer.
        Logic: Scans all producers for the requested item and performs an atomic transfer.
        """
        logging.info(
            f'Entered method %s '
            f'with arguments cart_id = %d, product = %s',
            self.add_to_cart.__name__, cart_id, product)

        self.consumers_lock.acquire()
        product_key = None

        # Block Logic: Searches for the product across all active producers.
        for key in self.producers.keys():
            for prod in self.producers.get(key):
                if prod == product:
                    product_key = key
                    break

        # Block Logic: Atomic transfer of product from producer to cart.
        if product_key is not None:
            if cart_id in self.carts.keys():
                self.producers.get(product_key).remove(product)
                self.carts.get(cart_id).append([product_key, product])
                self.consumers_lock.release()
                logging.info(
                    'Leaved method %s and returned '
                    '%s', self.add_to_cart.__name__, True)
                return True

        self.consumers_lock.release()
        logging.info(
            'Leaved method %s and returned '
            '%s', self.add_to_cart.__name__, False)
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a previously reserved product from a cart back to the producer's queue.
        Functional Utility: Supports cart cancellation or item removal with inventory restoration.
        """
        logging.info(
            'Entered method %s '
            'with arguments cart_id = %d, product = %s',
            self.remove_from_cart.__name__, cart_id, product)

        product_key = None
        products = self.carts.get(cart_id)
        if products is None:
            return False

        # Block Logic: Locates the item and its origin producer within the cart.
        for key, prod in products:
            if prod == product:
                product_key = key
                break

        if product_key is not None:
            products.remove([product_key, product])
            self.carts.update({cart_id: products})
            self.producers.get(product_key).append(product) # Inventory restoration.
            logging.info(
                f'Leaved method %s and returned '
                f'%s', self.remove_from_cart.__name__, True)
            return True
        logging.info(
            'Leaved method %s and returned '
            '%s', self.remove_from_cart.__name__, False)
        return False

    def place_order(self, cart_id):
        """
        @brief Finalizes the order and retrieves all items in the cart.
        """
        logging.info(
            'Entered method %s '
            'with arguments cart_id = %d', self.place_order.__name__, cart_id)

        items = []
        for _, prod in self.carts.get(cart_id):
            items.append(prod)
        logging.info(
            'Leaved method %s and returned '
            'items = %s', self.place_order.__name__, items)
        return items

import time
from threading import Thread


class Producer(Thread):
    """
    @brief Producer entity responsible for manufacturing and publishing products.
    Functional Utility: Continuously generates products and attempts to publish 
    them to the marketplace, respecting buffer capacity limits.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.
        @param products List of product specifications (id, quantity, production_time).
        @param marketplace Mediator instance.
        @param republish_wait_time Duration to wait when the marketplace buffer is full.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def post_product(self, id_producer, product):
        """
        Block Logic: Orchestrates the production and publishing sequence.
        Logic: Respects per-item production times and implements wait logic for full buffers.
        """
        id_prod, quantity, wait_time = product[0], product[1], product[2]
        posted = 0
        while posted < quantity:
            status = self.marketplace.publish(id_producer, id_prod)
            if status:
                posted += 1
                time.sleep(wait_time) # Simulation of manufacturing latency.
            else:
                time.sleep(self.republish_wait_time) # Wait for consumer activity to free space.

    def run(self):
        """
        @brief Infinite production cycle.
        """
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                self.post_product(producer_id, product)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base entity for marketplace goods.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Domain-specific specialization for Tea products.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Domain-specific specialization for Coffee products.
    """
    acidity: str
    roast_level: str
