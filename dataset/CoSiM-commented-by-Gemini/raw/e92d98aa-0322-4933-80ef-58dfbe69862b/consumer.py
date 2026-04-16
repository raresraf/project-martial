"""
This module implements a multi-threaded producer-consumer simulation of a marketplace.

It defines the following main classes:
- Marketplace: A thread-safe marketplace where producers can publish products and
  consumers can purchase them. It manages inventory, producers, and consumer carts.
- Producer: A thread that generates and publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places orders from the marketplace.

The simulation uses threading primitives like Locks to ensure data consistency
in a concurrent environment. It also includes a suite of unit tests to verify the
marketplace's functionality.
"""


from threading import Thread
from time import sleep

ADD_OPTION = "add"
NAME_ARG = "name"
QUANTITY = "quantity"
PRODUCT_ARG = "product"
TYPE = "type"


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer simulates the behavior of a customer who browses products,
    adds them to a shopping cart, and eventually places an order. The consumer's
    actions are defined in a list of cart operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts: A list of shopping operations for the consumer to perform.
            marketplace: The Marketplace instance to interact with.
            retry_wait_time (float): Time in seconds to wait before retrying to add a product
                                     if it's not available.
            **kwargs: Keyword arguments for the Thread constructor, including the consumer's name.
        """

        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cons_name = kwargs[NAME_ARG]

    def run(self):
        """
        The main execution logic for the consumer thread.

        It processes a series of cart operations, such as adding or removing products.
        When a product is unavailable, it waits for a specified time and retries.
        After processing the cart operations, it places the final order.
        """
        cons_id = self.marketplace.new_cart()

        for entries in self.carts:
            for cart in entries:
                for _ in range(cart[QUANTITY]):
                    if cart[TYPE] == ADD_OPTION:
                        while True:
                            done = self.marketplace.add_to_cart(cons_id, cart[PRODUCT_ARG])

                            if done:
                                break

                            
                            sleep(self.retry_wait_time)
                    else:
                        self.marketplace.remove_from_cart(cons_id, cart[PRODUCT_ARG])

            groceries = self.marketplace.place_order(cons_id)

            for product in groceries:
                print(f"{self.cons_name} bought {product}")


from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import time
import unittest
import tema.product as p


MARKETPLACE_LOGGER = logging.getLogger("market_logger")

MARKETPLACE_LOGGER.setLevel(logging.INFO)

HANDLER = RotatingFileHandler("marketplace.log", maxBytes=100000, backupCount=16)

FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
FORMATTER.converter = time.gmtime

HANDLER.setFormatter(FORMATTER)
MARKETPLACE_LOGGER.addHandler(HANDLER)

TEST_QUEUE_SIZE = 3


class ProductEntry:
    """
    Represents a single product instance within the marketplace's inventory.

    This class wraps a product with its producer's ID and availability status,
    allowing the marketplace to track individual items from their creation by a
    producer to their purchase by a consumer.
    """

    def __init__(self, product, producer_id):
        """
        Initializes a ProductEntry instance.

        Args:
            product: The product object.
            producer_id: The ID of the producer who published the product.
        """

        self.product = product
        self.producer_id = producer_id
        self.is_available = True

    def __str__(self):
        return f"prod = {self.product}, id = {self.producer_id}, av = {self.is_available}"


class Marketplace:
    """
    Implements the central marketplace for producers and consumers.

    This class is thread-safe and manages the registration of producers,
    the publication of products, and the entire consumer shopping experience,
    from creating a cart to placing an order. It uses locks to protect shared
    data structures from race conditions.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at one time.
        """

        MARKETPLACE_LOGGER.info("start __init__ with args: %d", queue_size_per_producer)

        self.queue_size = queue_size_per_producer
        self.producers = {}
        self.consumers = {}
        self.curr_prod_id = 0
        self.curr_cons_id = 0
        self.reg_prod_lock = Lock()
        self.reg_cart_lock = Lock()
        self.producer_locks = {}



        MARKETPLACE_LOGGER.info("exit __init__")

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Returns:
            int: The unique ID assigned to the new producer.
        """

        MARKETPLACE_LOGGER.info("start register_producer")

        
        
        with self.reg_prod_lock:
            new_prod_id = self.curr_prod_id
            self.producers[new_prod_id] = []
            self.producer_locks[new_prod_id] = Lock()
            self.curr_prod_id += 1

            MARKETPLACE_LOGGER.info("exit register_producer")

            return new_prod_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer.
            product: The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue is full.
        """

        MARKETPLACE_LOGGER.info("start publish with args: %d; %s", producer_id, product)

        if len(self.producers[producer_id]) < self.queue_size:
            self.producers[producer_id].append(ProductEntry(product, producer_id))

            MARKETPLACE_LOGGER.info("exit publish")

            return True

        MARKETPLACE_LOGGER.info("exit publish")

        return False

    def new_cart(self):
        """
        Creates a new shopping cart for a consumer.

        Returns:
            int: The unique ID assigned to the new cart.
        """

        MARKETPLACE_LOGGER.info("start new_cart")

        
        
        with self.reg_cart_lock:
            new_cart_id = self.curr_cons_id
            self.consumers[new_cart_id] = []
            self.curr_cons_id += 1

            MARKETPLACE_LOGGER.info("exit new_cart")

            return new_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's shopping cart.

        Args:
            cart_id (int): The ID of the shopping cart.
            product: The product to add.

        Returns:
            bool: True if the product was added successfully, False if the
                  product is not available.
        """

        MARKETPLACE_LOGGER.info("start add_to_cart with args: %id; %s", cart_id, product)

        for id_producer, product_entries in self.producers.items():
            
            with self.producer_locks[id_producer]:
                for product_entry in product_entries:
                    if product == product_entry.product and product_entry.is_available:
                        product_entry.is_available = False
                        self.consumers[cart_id].append(product_entry)
                        return True

        MARKETPLACE_LOGGER.info("exit add_to_cart")

        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's shopping cart.

        Args:
            cart_id (int): The ID of the shopping cart.
            product: The product to remove.
        """

        MARKETPLACE_LOGGER.info("start remove_from_cart with args: %d; %s", cart_id, product)

        to_remove = None

        for product_entry in self.consumers[cart_id]:
            if product == product_entry.product:
                
                with self.producer_locks[product_entry.producer_id]:
                    to_remove = product_entry
                    product_entry.is_available = True
                    break



        self.consumers[cart_id].remove(to_remove)

        MARKETPLACE_LOGGER.info("exit remove_from_cart")

    def place_order(self, cart_id):
        """
        Places an order for all items in a consumer's shopping cart.

        This action removes the items from the marketplace inventory permanently.

        Args:
            cart_id (int): The ID of the shopping cart.

        Returns:
            list: A list of products that were successfully purchased.
        """

        MARKETPLACE_LOGGER.info("start place_order with args: %d", cart_id)

        
        groceries = []

        for product_entry in self.consumers[cart_id]:
            
            with self.producer_locks[product_entry.producer_id]:
                groceries.append(product_entry.product)
                self.producers[product_entry.producer_id].remove(product_entry)

        
        self.consumers[cart_id].clear()

        MARKETPLACE_LOGGER.info("exit place_order")

        return groceries


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    """

    def setUp(self):
        """
        Set up the test environment for each test case.
        """
        self.marketplace = Marketplace(TEST_QUEUE_SIZE)
        self.coffee_product_0 = p.Coffee("coffee0", 0, "0", "l")
        self.coffee_product_1 = p.Coffee("coffee1", 1, "1", "m")
        self.coffee_product_2 = p.Coffee("coffee2", 2, "2", "h")
        self.tea_product_0 = p.Tea("tea0", 3, "t0")
        self.tea_product_1 = p.Tea("tea1", 4, "t1")
        self.producer_0 = self.marketplace.register_producer()
        self.producer_1 = self.marketplace.register_producer()
        self.consumer_0 = self.marketplace.new_cart()
        self.consumer_1 = self.marketplace.new_cart()

    def test_register_producer(self):
        """
        Test the registration of new producers.
        """

        self.assertEqual(0, self.producer_0, "wrong id for producer_0")
        self.assertEqual(1, self.producer_1, "wrong id for producer_1")

        id_test = []

        for _ in range(1024):
            id_test.append(self.marketplace.register_producer())

        for i in range(1024):
            self.assertEqual(i + 2, id_test[i], f"wrong id: got {id_test[i]}, should be {i + 2}")

    def test_publish(self):
        """
        Test publishing products to the marketplace.
        """

        
        result = self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][0].product,
                         self.coffee_product_0,
                         "wrong product")

        
        result = self.marketplace.publish(self.producer_0, self.coffee_product_1)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][1].product,
                         self.coffee_product_1,
                         "wrong product")

        result = self.marketplace.publish(self.producer_0, self.coffee_product_2)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][2].product,
                         self.coffee_product_2,
                         "wrong product")

        
        result = self.marketplace.publish(self.producer_0, self.tea_product_0)
        self.assertEqual(result, False, "should be False")

    def test_new_cart(self):
        """
        Test the creation of new consumer carts.
        """

        self.assertEqual(0, self.consumer_0, "wrong id for consumer_0")
        self.assertEqual(1, self.consumer_1, "wrong id for consumer_1")

        id_test = []

        for _ in range(1024):
            id_test.append(self.marketplace.new_cart())

        for i in range(1024):
            self.assertEqual(i + 2, id_test[i], f"wrong id: got {id_test[i]}, should be {i + 2}")

    def test_add_to_cart(self):
        """
        Test adding products to a shopping cart.
        """

        
        self.marketplace.publish(self.producer_1, self.tea_product_1)
        result = self.marketplace.add_to_cart(self.consumer_0, self.tea_product_1)
        self.assertEqual(result, True, "should be True")

        
        result = self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_0)
        self.assertEqual(result, False, "should be False")

        
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_1)
        self.assertEqual(result, False, "should be False")

        
        self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.marketplace.publish(self.producer_1, self.coffee_product_1)
        self.marketplace.publish(self.producer_1, self.coffee_product_2)
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_0)
        self.assertEqual(result, True, "should be True")
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_2)
        self.assertEqual(result, True, "should be True")
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_1)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(3,
                         len(self.marketplace.consumers[self.consumer_1]),
                         "wrong number of products in cart")

    def test_remove_from_cart(self):
        """
        Test removing products from a shopping cart.
        """

        
        self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_0)
        self.marketplace.remove_from_cart(self.consumer_0, self.coffee_product_0)
        self.assertEqual(0,
                         len(self.marketplace.consumers[self.consumer_0]),
                         "wrong number of products in cart")

        
        self.marketplace.publish(self.producer_1, self.tea_product_0)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_0)
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_0)
        self.assertEqual(result, False, "should be False")
        self.marketplace.remove_from_cart(self.consumer_0, self.tea_product_0)
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_0)
        self.assertEqual(result, True, "should be True")

    def test_place_order(self):
        """
        Test the process of placing an order.
        """

        self.marketplace.publish(self.producer_0, self.tea_product_0)
        self.marketplace.publish(self.producer_1, self.coffee_product_1)
        self.marketplace.publish(self.producer_1, self.coffee_product_2)
        self.marketplace.publish(self.producer_0, self.coffee_product_2)
        self.marketplace.publish(self.producer_1, self.tea_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_2)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_2)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_1)
        self.marketplace.remove_from_cart(self.consumer_0, self.coffee_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_0)

        
        self.assertEqual(4,
                         len(self.marketplace.consumers[self.consumer_0]),
                         "wrong number of products in cart")

        groceries_list = self.marketplace.place_order(self.consumer_0)

        
        self.assertEqual(groceries_list,
                         [self.coffee_product_2, self.coffee_product_2,
                          self.tea_product_1, self.tea_product_0],
                         "wrong products in cart")

        
        self.assertEqual(0, len(self.marketplace.producers[self.consumer_0]), "should be empty")


from threading import Thread
from time import sleep

PRODUCTS = 0
QUANTITY = 1
PRODUCE_TIME = 2


class Producer(Thread):
    """
    Represents a producer thread that generates and publishes products.

    Each producer continuously creates a predefined set of products and attempts
    to publish them to the marketplace. If the marketplace's queue for that
    producer is full, it waits and retries.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products: A list of products for the producer to generate.
            marketplace: The Marketplace instance to interact with.
            republish_wait_time (float): Time in seconds to wait before retrying to
                                         publish a product if the queue is full.
            **kwargs: Keyword arguments for the Thread constructor.
        """

        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer thread.

        It registers with the marketplace and then enters an infinite loop to
        produce and publish its assigned products, respecting production times
        and marketplace capacity.
        """
        prod_id = self.marketplace.register_producer()

        while True:
            for product in self.products:

                for _ in range(product[QUANTITY]):
                    while True:
                        done = self.marketplace.publish(prod_id, product[PRODUCTS])

                        if done:
                            break

                        
                        sleep(self.republish_wait_time)

                    
                    sleep(product[PRODUCE_TIME])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A data class representing a generic product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class representing Tea, a specific type of Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class representing Coffee, a specific type of Product."""
    acidity: str
    roast_level: str
