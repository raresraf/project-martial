
"""
@6ade1122-0765-4300-b197-1f921230f31e/consumer.py
@brief Thread-safe marketplace architecture with hierarchical resource management.
This file implements a concurrent system for product exchange using a broker 
pattern. It features Consumer and Producer threads interacting through a 
central Marketplace and specialized Cart objects. It ensures thread-safe 
identity assignment and uses granular locks to manage global and per-agent 
capacities, accompanied by a comprehensive unit test suite.

Domain: Concurrent Systems, Synchronization, Resource Accounting.
"""

import logging
from threading import Thread, Lock
from time import sleep
from tema.marketplace import Marketplace


class Consumer(Thread):
    """
    Functional Utility: Represent a consumer agent that processes a sequence of orders.
    Logic: Orchestrates cart-based shopping by adding or removing items. It 
    utilizes a shared static counter (protected by a class lock) to assign 
    globally unique consumer IDs for logging and reporting.
    """

    __id = 1
    __id_lock = Lock()

    def __init__(self, carts, marketplace: Marketplace, retry_wait_time: int, **kwargs):
        """
        Constructor: Initializes the consumer with its shopping carts and sets its unique ID.
        """
        Thread.__init__(self, **kwargs)

        with Consumer.__id_lock:
            self.__id = Consumer.__id
            Consumer.__id += 1

        self.__marketplace = marketplace
        self.__retry_wait_time = retry_wait_time
        self.__carts = carts
        self.__cart_id = None

    def add_product(self, product):
        """
        Block Logic: Product acquisition loop with retry backoff.
        Logic: Attempts to add a product to the current cart. If rejected 
        (insufficient stock), it sleeps before re-trying the operation.
        """
        status = self.__marketplace.add_to_cart(self.__cart_id, product)
        while not status:
            sleep(self.__retry_wait_time)
            status = self.__marketplace.add_to_cart(self.__cart_id, product)

    def add_products(self, product, quantity):
        """
        Block Logic: Batch acquisition.
        """
        for _ in range(quantity):
            self.add_product(product)

    def remove_product(self, product):
        """
        Functional Utility: Removes a single item from the cart.
        """
        self.__marketplace.remove_from_cart(self.__cart_id, product)

    def remove_products(self, product, quantity):
        """
        Block Logic: Batch removal.
        """
        for _ in range(quantity):
            self.remove_product(product)

    def buy_cart(self, cart):
        """
        Execution Logic: Atomic processing of a single shopping cart.
        Invariant: All add/remove operations are performed before the 
        marketplace order is finalized and results printed.
        """
        self.__cart_id = self.__marketplace.new_cart()
        logging.info(cart)

        for action in cart:
            if action['type'] == 'add':
                self.add_products(action['product'], action['quantity'])
            else:
                self.remove_products(action['product'], action['quantity'])

        products = self.__marketplace.place_order(self.__cart_id)
        for product in products:
            print(f'cons{self.__id} bought {product}'.strip())

        self.__cart_id = None

    def run(self):
        """
        Execution Logic: Main thread entry point for cart processing.
        """
        for cart in self.__carts:
            self.buy_cart(cart)

import logging
from logging.handlers import RotatingFileHandler
from queue import Queue
from threading import Lock
from typing import Dict, List
from unittest import TestCase
from tema.product import Product, Tea

LOGGER = logging.getLogger('MARKETPLACE')
HANDLER = RotatingFileHandler('marketplace.log', maxBytes=102, backupCount=10)
LOGGER.addHandler(HANDLER)


class Cart:
    """
    Functional Utility: Encapsulates consumer shopping state.
    Logic: Tracks products and their quantities. It uses a class-level lock 
    to ensure unique cart ID assignment across all instances.
    """

    __last_id = 1
    __id_lock = Lock()

    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.amount: Dict[str, int] = {}

        with Cart.__id_lock:
            self.__id = Cart.__last_id
            Cart.__last_id += 1

    def add(self, product):
        """
        Functional Utility: Updates internal quantity tracking for a product.
        """
        if product.name not in self.amount:
            self.amount[product.name] = 0

        self.amount[product.name] += 1
        self.products[product.name] = product

    def remove(self, product):
        """
        Functional Utility: Decrements internal quantity for a product.
        """
        self.amount[product.name] -= 1

    def list(self) -> List:
        """
        Functional Utility: Flattens the quantity map into a sequence of products.
        """
        result = []
        for product_name in self.amount:
            result += [self.products[product_name]] * self.amount[product_name]
        return result

    def get_id(self):
        return self.__id


class Marketplace:
    """
    Functional Utility: Centralized broker for concurrent product publishing and acquisition.
    Logic: Manages producer identity, stock levels (total and reserved), and 
    consumer carts. It uses a global lock to protect the consistency of 
    multiple internal mappings during transactional operations.
    """

    __producer_id = 0
    __producer_id_lock = Lock()

    def __init__(self, queue_size_per_producer: int):
        self.producer_capacity = queue_size_per_producer
        self.producers = {}
        self.prod_queue = {}
        self.all_products = {}
        self.reserved_products = {}
        self.carts = {}
        self.lock = Lock()

    def register_producer(self) -> str:
        """
        Functional Utility: Issues a unique identifier to a new producer.
        """
        with Marketplace.__producer_id_lock:
            producer_id = f"producer{Marketplace.__producer_id}"
            Marketplace.__producer_id += 1

        self.producers[producer_id] = {}
        LOGGER.info("register_producer -> %s", producer_id)
        return producer_id

    def publish(self, producer_id: str, product) -> bool:
        """
        Functional Utility: Adds a product instance to a producer's stock.
        Logic: Enforces per-producer limits. It tracks the specific producer 
        ID in a queue to maintain FIFO-like fairness for acquisitions.
        """
        with self.lock:
            if product.name not in self.all_products:
                self.all_products[product.name] = 0

            if product.name not in self.producers[producer_id]:
                self.producers[producer_id][product.name] = 0

            if self.producers[producer_id][product.name] == self.producer_capacity:
                LOGGER.info("publish(%s, %s) -> False", producer_id, product)
                return False

            if product.name not in self.prod_queue:
                self.prod_queue[product.name] = Queue()

            self.producers[producer_id][product.name] += 1
            self.all_products[product.name] += 1
            self.prod_queue[product.name].put(producer_id)

        LOGGER.info("publish(%s, %s) -> True", producer_id, product)
        return True

    def new_cart(self) -> int:
        """
        Functional Utility: Creates and registers a new shopping cart.
        """
        cart = Cart()
        self.carts[cart.get_id()] = cart
        LOGGER.info("new_cart -> %s", cart.get_id())
        return cart.get_id()

    def add_to_cart(self, cart_id: int, product) -> bool:
        """
        Functional Utility: Reserves a product instance for a specific cart.
        Logic: Checks global availability by comparing total stock vs 
        currently reserved stock. If available, updates the cart and 
        the reservation count.
        """
        with self.lock:
            if product.name not in self.all_products:
                self.all_products[product.name] = 0
            if product.name not in self.reserved_products:
                self.reserved_products[product.name] = 0

            if self.all_products[product.name] == self.reserved_products[product.name]:
                LOGGER.info("add_to_cart(%s, %s) -> False", cart_id, product.name)
                return False

            cart = self.carts[cart_id]
            cart.add(product)
            self.reserved_products[product.name] += 1

        LOGGER.info("add_to_cart(%s, %s) -> True", cart_id, product.name)
        return True

    def remove_from_cart(self, cart_id: int, product):
        """
        Functional Utility: Removes an item from the cart without releasing the reservation.
        """
        LOGGER.info("remove_from_cart(%s, %s)", cart_id, product.name)
        with self.lock:
            cart = self.carts[cart_id]
            cart.remove(product)

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes the order and reconciles global stock.
        Logic: Decrements global and reserved stock counts. It pops the 
        producer IDs from the product queue to identify which producer's 
        capacity is now truly consumed.
        """
        cart = self.carts[cart_id]
        products = cart.list()
        amount = cart.amount

        with self.lock:
            for product_id in cart.amount:
                self.all_products[product_id] -= amount[product_id]
                self.reserved_products[product_id] -= amount[product_id]

                while amount[product_id] > 0:
                    producer_id = self.prod_queue[product_id].get()
                    self.producers[producer_id][product_id] -= 1
                    amount[product_id] -= 1

        LOGGER.info("place_order(%s) -> %s", cart_id, products)
        return products


class TestMarketplace(TestCase):
    """
    Functional Utility: Integrity validation suite for the Marketplace.
    """
    PRODUCER_COUNT = 10
    PRODUCER_QUEUE_SIZE = 10

    def setUp(self) -> None:
        self.marketplace = Marketplace(TestMarketplace.PRODUCER_QUEUE_SIZE)

    def test_register_producer(self):
        producers = set()
        for _ in range(TestMarketplace.PRODUCER_COUNT):
            producer_id = self.marketplace.register_producer()
            producers.add(producer_id)
        self.assertEqual(len(producers), TestMarketplace.PRODUCER_COUNT)

    def test_publish(self):
        producer_id = self.marketplace.register_producer()
        product = Tea('test_tea', 10, 'test_type')
        self.marketplace.publish(producer_id, product)
        self.assertEqual(self.marketplace.all_products[product.name], 1)

    def test_add_to_cart(self):
        producer_id = self.marketplace.register_producer()
        product = Tea('test_tea', 10, 'test_type')
        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product))
        self.marketplace.publish(producer_id, product)
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product))


class Producer(Thread):
    """
    Functional Utility: Represent a production agent that supplies the marketplace.
    Logic: Iterates through its inventory, simulating production time before 
    attempting to publish. It implements a retry mechanism for full queues.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.__orders = products
        self.__marketplace: Marketplace = marketplace
        self.__wait_time = republish_wait_time
        self.__id = marketplace.register_producer()

    @staticmethod
    def prepare_product(product, prepare_time):
        """
        Block Logic: Simulated manufacturing delay.
        """
        sleep(prepare_time)
        return product

    def publish_product(self, product, prepare_time):
        """
        Block Logic: Product publication with backoff.
        """
        self.prepare_product(product, prepare_time)
        status = self.__marketplace.publish(self.__id, product)
        while not status:
            sleep(self.__wait_time)
            status = self.__marketplace.publish(self.__id, product)

    def publish_products(self, product, quantity, prepare_time):
        for _ in range(quantity):
            self.publish_product(product, prepare_time)

    def run(self):
        while True:
            for order in self.__orders:
                product, quantity, prepare_time = order
                self.publish_products(product, quantity, prepare_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Functional Utility: Immutable base data carrier for products.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Functional Utility: Specialized product for tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Functional Utility: Specialized product for coffee.
    """
    acidity: str
    roast_level: str
