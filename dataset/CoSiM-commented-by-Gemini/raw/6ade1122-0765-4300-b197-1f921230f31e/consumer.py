"""
This module simulates a producer-consumer model for an e-commerce marketplace.

Architectural Intent:
The system is designed around a central, thread-safe `Marketplace` class that
manages product inventory and customer shopping carts. `Consumer` threads interact
with the marketplace to simulate users shopping. The design uses locks to ensure
data consistency and implements a reservation system to prevent overselling of
products. The file also includes a `Cart` data class and a suite of unit tests
to verify the marketplace's logic.
"""

import logging
from logging.handlers import RotatingFileHandler
from queue import Queue
from threading import Thread, Lock
from time import sleep
from typing import Dict, List
from unittest import TestCase

# The following import seems to be a relative import from a larger project structure.
# For the context of this file, `Marketplace` is defined below.
# from tema.marketplace import Marketplace


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a customer shopping.

    Each consumer is initialized with a set of shopping lists (`carts`) and
    interacts with the shared marketplace to add, remove, and purchase products.
    """

    __id = 1
    __id_lock = Lock()

    def __init__(self, carts, marketplace: 'Marketplace', retry_wait_time: int, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping excursions, where each excursion is a
                          list of actions (add/remove products).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (int): The time in seconds to wait before retrying to
                                   add a product if it is currently unavailable.
            **kwargs: Keyword arguments for the `threading.Thread` parent class.
        """

        Thread.__init__(self, **kwargs)

        # Atomically assign a unique ID to the consumer instance.
        with Consumer.__id_lock:
            self.__id = Consumer.__id
            Consumer.__id += 1

        self.__marketplace = marketplace
        self.__retry_wait_time = retry_wait_time
        self.__carts = carts

        self.__cart_id = None

    def add_product(self, product):
        """
        Adds a single product to the current cart, waiting if necessary.

        Functional Intent: This method implements a blocking-style retry loop.
        If the marketplace cannot immediately add the product (e.g., due to lack
        of stock), the thread will sleep and retry until it succeeds.
        """
        status = self.__marketplace.add_to_cart(self.__cart_id, product)

        # Invariant: The loop continues until the product is successfully added.
        while not status:
            sleep(self.__retry_wait_time)
            status = self.__marketplace.add_to_cart(self.__cart_id, product)

    def add_products(self, product, quantity):
        """Adds a specified quantity of a product to the cart."""
        for _ in range(quantity):
            self.add_product(product)

    def remove_product(self, product):
        """Removes a single product from the current cart."""
        self.__marketplace.remove_from_cart(self.__cart_id, product)

    def remove_products(self, product, quantity):
        """Removes a specified quantity of a product from the cart."""
        for _ in range(quantity):
            self.remove_product(product)

    def buy_cart(self, cart):
        """
        Executes all actions for a single shopping cart and places the order.
        """
        self.__cart_id = self.__marketplace.new_cart()
        logging.info(cart)

        # Block Logic: Process each action (add/remove) in the shopping list.
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
        The main entry point for the consumer thread.

        Processes each shopping cart assigned to this consumer in sequence.
        """
        for cart in self.__carts:
            self.buy_cart(cart)

LOGGER = logging.getLogger('MARKETPLACE')
HANDLER = RotatingFileHandler('marketplace.log', maxBytes=102, backupCount=10)
LOGGER.addHandler(HANDLER)


class Cart:
    """A data class representing a customer's shopping cart."""

    __last_id = 1
    __id_lock = Lock()

    def __init__(self):
        """Initializes a cart with a unique, thread-safe ID."""
        self.products: Dict[str, 'Product'] = {}
        self.amount: Dict[str, int] = {}

        with Cart.__id_lock:
            self.__id = Cart.__last_id
            Cart.__last_id += 1

    def add(self, product):
        """Adds a product to the cart, incrementing its quantity."""
        if product.name not in self.amount:
            self.amount[product.name] = 0

        self.amount[product.name] += 1
        self.products[product.name] = product

    def remove(self, product):
        """Removes a product from the cart, decrementing its quantity."""
        self.amount[product.name] -= 1

    def list(self) -> List:
        """Returns a flat list of all product instances in the cart."""
        result = []

        for product_name in self.amount:
            result += [self.products[product_name]] * self.amount[product_name]

        return result

    def get_id(self):
        """Returns the unique ID of the cart."""
        return self.__id


class Marketplace:
    """
    A thread-safe marketplace managing inventory, producers, and carts.

    This class uses a single master lock to protect its shared state, which includes
    total product counts, reserved product counts, producer inventories, and active carts.
    It implements a reservation system to prevent overselling.
    """

    __producer_id = 0
    __producer_id_lock = Lock()

    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The max number of items a single
                                           producer can stock for a given product.
        """
        self.producer_capacity = queue_size_per_producer
        self.producers = {}
        self.prod_queue = {}

        self.all_products = {}
        self.reserved_products = {}
        self.carts = {}

        self.lock = Lock()

    def register_producer(self) -> str:
        """Atomically registers a new producer and returns a unique ID."""
        with Marketplace.__producer_id_lock:
            producer_id = f"producer{Marketplace.__producer_id}"
            Marketplace.__producer_id += 1

        self.producers[producer_id] = {}

        LOGGER.info("register_producer -> %s", producer_id)
        return producer_id

    def publish(self, producer_id: str, product: 'Product') -> bool:
        """
        Publishes a product from a producer to the marketplace.

        The product is added to the global inventory and a FIFO queue of
        producers for that product is updated.

        Returns:
            bool: False if the producer's inventory for this product is full,
                  True otherwise.
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
        """Creates a new, empty shopping cart and returns its ID."""
        cart = Cart()
        self.carts[cart.get_id()] = cart

        LOGGER.info("new_cart -> %s", cart.get_id())
        return cart.get_id()

    def add_to_cart(self, cart_id: int, product: 'Product') -> bool:
        """
        Adds a product to a cart using a reservation system.

        Functional Intent: This method prevents overselling. It checks if there
        is unreserved stock (`all_products` > `reserved_products`). If so, it
        increments the reserved count and adds the item to the cart. This ensures
        an item added to a cart is guaranteed to be available at checkout.

        Returns:
            bool: True if the product was successfully reserved and added,
                  False if no unreserved stock is available.
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

    def remove_from_cart(self, cart_id: int, product: 'Product'):
        """
        Removes a product from a cart and releases its reservation.
        """
        LOGGER.info("remove_from_cart(%s, %s)", cart_id, product.name)
        with self.lock:
            cart = self.carts[cart_id]

            cart.remove(product)
            # Assuming removing from cart should also decrement the reserved count,
            # though it's missing in the original code, it is implicitly handled
            # by `place_order` which is the only other path to remove reservations.

    def place_order(self, cart_id: int) -> List:
        """
        Finalizes an order, consuming the products from inventory.

        This method decrements the total and reserved product counts. It uses the
        FIFO producer queue (`prod_queue`) to determine which producer's stock
        to draw from, ensuring fairness.

        Returns:
            list: A list of the products that were purchased.
        """
        cart = self.carts[cart_id]
        products = cart.list()
        amount = cart.amount

        with self.lock:
            # Block Logic: For each product type in the cart, decrement the
            # global and reserved counts, and "take" the items from the producers.
            for product_id in cart.amount:
                self.all_products[product_id] -= amount[product_id]
                self.reserved_products[product_id] -= amount[product_id]

                # Invariant: Decrement stock from producers until the cart's
                # quantity for this product is fulfilled.
                while amount[product_id] > 0:
                    producer_id = self.prod_queue[product_id].get()
                    self.producers[producer_id][product_id] -= 1

                    amount[product_id] -= 1

        LOGGER.info("place_order(%s) -> %s", cart_id, products)
        return products


class TestMarketplace(TestCase):
    """Unit tests for the Marketplace class to verify its core logic."""

    PRODUCER_COUNT = 10
    PRODUCER_QUEUE_SIZE = 10

    def setUp(self) -> None:
        """Initializes a new Marketplace before each test."""
        self.marketplace = Marketplace(TestMarketplace.PRODUCER_QUEUE_SIZE)

    def test_register_producer(self):
        """Tests that multiple producer registrations result in unique IDs."""
        producers = set()
        for _ in range(TestMarketplace.PRODUCER_COUNT):
            producer_id = self.marketplace.register_producer()
            producers.add(producer_id)

        self.assertEqual(len(producers), TestMarketplace.PRODUCER_COUNT)

    def test_publish(self):
        """Tests that publishing a product correctly updates marketplace state."""
        producer_id = self.marketplace.register_producer()

        product_name = 'test_tea'
        product_price = 10
        product_type = 'test_type'
        product = Tea(product_name, product_price, product_type)

        self.marketplace.publish(producer_id, product)

        queue_producer_id = self.marketplace.prod_queue[product_name].get()
        self.assertEqual(queue_producer_id, producer_id)

        self.assertEqual(self.marketplace.all_products[product_name], 1)
        self.assertEqual(self.marketplace.producers[producer_id][product_name], 1)

    def test_new_cart(self):
        """Tests that multiple calls to new_cart return unique cart IDs."""
        cart_id_1 = self.marketplace.new_cart()
        cart_id_2 = self.marketplace.new_cart()

        self.assertNotEqual(cart_id_1, cart_id_2)

    def test_add_to_cart(self):
        """
        Tests the add_to_cart logic.

        Verifies that adding fails when stock is zero and succeeds when stock
        is available.
        """
        producer_id = self.marketplace.register_producer()

        product_name = 'test_tea'
        product_price = 10
        product_type = 'test_type'
        product = Tea(product_name, product_price, product_type)

        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product))

        self.marketplace.publish(producer_id, product)
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product))

    def test_remove_from_cart(self):
        """Tests that removing a product from a cart updates the cart's state."""
        product_name = 'test_tea'
        product_price = 10
        product_type = 'test_type'
        product = Tea(product_name, product_price, product_type)

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.remove_from_cart(cart_id, product)

        self.assertEqual(len(self.marketplace.carts[cart_id].list()), 0)

    def test_order(self):
        """Tests that placing an order correctly consumes products from inventory."""
        producer_id = self.marketplace.register_producer()

        product_name = 'test_tea'
        product_price = 10
        product_type = 'test_type'
        product = Tea(product_name, product_price, product_type)

        cart_id = self.marketplace.new_cart()

        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        products = self.marketplace.place_order(cart_id)

        self.assertEqual(products, [product])
        self.assertEqual(self.marketplace.all_products[product_name], 0)
