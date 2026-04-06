"""
This module simulates a multi-threaded producer-consumer model for an e-commerce
marketplace.

It defines `Consumer`, `Producer`, and `Marketplace` classes. This implementation
has a significant design flaw in its inventory management, where only one producer
can be associated with a given product type at a time, regardless of stock from
other producers.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that processes a list of shopping carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts", where each cart is a list of
                          operations (add/remove).
            marketplace (Marketplace): A reference to the central marketplace object.
            retry_wait_time (int): Time in seconds to wait before retrying an 'add'
                                   operation if the product is not available.
            **kwargs: Keyword arguments for the Thread parent class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread. For each assigned cart,
        it creates a new cart in the marketplace, processes all add/remove
        operations, and finally places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for cart_op in cart:
                quantity = cart_op["quantity"]
                if cart_op["type"] == "add":
                    # Block Logic: Repeatedly try to add the desired quantity of a product.
                    # This simulates waiting for a product to be restocked.
                    while quantity > 0:
                        res = self.marketplace.add_to_cart(cart_id, cart_op["product"])
                        while not res:
                            sleep(self.retry_wait_time)
                            res = self.marketplace.add_to_cart(cart_id, cart_op["product"])
                        quantity -= 1
                elif cart_op["type"] == "remove":
                    # Block Logic: Remove the desired quantity of a product.
                    while quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, cart_op["product"])
                        quantity -= 1
            
            # Place the order and print the items.
            # Note: Printing is not synchronized, so output may be interleaved with other threads.
            order = self.marketplace.place_order(cart_id)
            for element in order:
                print(self.name + ' bought ' + str(element))


import time
import unittest
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Tea, Coffee


class Marketplace:
    """
    Coordinates producers and consumers in a simulated e-commerce environment.

    This implementation uses dictionaries to manage producers and carts. Its primary
    inventory logic is flawed, as it only allows one producer to be the source
    for any given product type at a time.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        
        Args:
            queue_size_per_producer (int): The max number of products a single
                                           producer can have published.
        """
        self.queue_size_per_producers = queue_size_per_producer
        self.producers_dict = {}  # {producer_id: [product1, product2, ...]}
        self.producer_id_seed = 0
        self.producer_id_lock = Lock()
        self.carts_dict = {}  # {cart_id: [(producer_id, product), ...]}
        self.cart_id_lock = Lock()
        self.cart_id_seed = 0

        # Flawed inventory map: This maps a product to a SINGLE producer ID.
        # If multiple producers sell the same product, this will be overwritten.
        self.product_to_producer_id = {}
        
        self.producers_queue_sizes = {}
        self.producer_queue_lock = Lock()

        # --- Logging Setup ---
        logging.basicConfig(
            handlers=[RotatingFileHandler("marketplace.log",
                                          maxBytes=10000, backupCount=10)],
            level=logging.INFO,
            format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.Formatter.converter = time.gmtime
        logging.info('Marketplace start!')

    def register_producer(self):
        """Atomically registers a new producer and returns a unique ID."""
        with self.producer_id_lock:
            self.producer_id_seed += 1
        producer_id = self.producer_id_seed
        self.producers_dict[producer_id] = []
        self.producers_queue_sizes[producer_id] = 0
        logging.info('new producer id registered: %s', str(producer_id))
        return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer.

        BUG: This method overwrites the global `product_to_producer_id` entry,
        meaning only the last producer to publish a certain product will be
        recognized as its source.
        """
        logging.info('publish method arguments: (producer_id:%s), (product:%s)',
                     str(producer_id), str(product))
        if self.producers_queue_sizes[producer_id] >= self.queue_size_per_producers:
            logging.info('publish method return=%s', 'False')
            return False
        self.producers_dict[producer_id].append(product)
        # This is the source of the major design flaw.
        self.product_to_producer_id[product] = producer_id
        
        with self.producer_queue_lock:
            self.producers_queue_sizes[producer_id] += 1
        logging.info('publish method return=%s', 'True')
        return True

    def new_cart(self):
        """Atomically creates a new, empty cart and returns its ID."""
        with self.cart_id_lock:
            self.cart_id_seed += 1
            cart_id = self.cart_id_seed
        self.carts_dict[cart_id] = []
        logging.info('new_cart method return cart_id=%s', str(cart_id))
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        This logic is flawed because it relies on the globally-mapped producer
        ID for a product, which may not reflect all available stock.
        """
        logging.info('add_to_cart method arguments: (cart_id:%s), (product:%s)',
                     str(cart_id), str(product))
        
        producer_id = self.product_to_producer_id.get(product, None)
        if producer_id is None:
            logging.info('add_to_cart method return=%s', 'False')
            return False
        if product not in self.producers_dict[producer_id]:
            logging.info('add_to_cart method return=%s', 'False')
            return False
        
        self.producers_dict[producer_id].remove(product)
        self.carts_dict[cart_id].append((producer_id, product))
        logging.info('add_to_cart method return=%s', 'True')
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the producer's inventory."""
        logging.info('remove_from_cart method arguments: (cart_id=%s), (product=%s)',
                     cart_id, product)
        producer_id = -1
        list_cart = self.carts_dict[cart_id]

        # Find the product in the cart to identify its original producer.
        for _, cart_tuple in enumerate(list_cart):
            if product == cart_tuple[1]:
                producer_id = cart_tuple[0]
                break
        if producer_id == -1:
            logging.info('remove_from_cart method return=%s', 'False')
            return False

        self.carts_dict[cart_id].remove((producer_id, product))
        self.producers_dict[producer_id].append(product)
        logging.info('remove_from_cart method return=%s', 'True')
        return True

    def place_order(self, cart_id):
        """
        Finalizes an order, freeing up producer slots for the items purchased.
        """
        logging.info('place_order method argument: (cart_id:%s)', cart_id)
        res = self.carts_dict.pop(cart_id)
        final_list = []
        for _, cart_tuple in enumerate(res):
            with self.producer_queue_lock:
                self.producers_queue_sizes[cart_tuple[0]] -= 1
            final_list.append(cart_tuple[1])
        logging.info('place_order method return=%s', str(final_list))
        return final_list


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Prepares a new Marketplace instance and products for each test."""
        self.marketplace = Marketplace(3)
        self.product1 = Tea(name='Wild Cherry', price=5, type='Black')
        self.product2 = Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')

    def test_register_producer(self):
        """Tests that producer IDs are generated sequentially."""
        res = self.marketplace.register_producer()
        self.assertEqual(res, 1, "Register wrong!")
        res = self.marketplace.register_producer()
        self.assertEqual(res, 2, "Register wrong!")

    def test_publish(self):
        """Tests product publishing and queue limits."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(1, self.product1),
                         True, "incorrect return value for publish")
        self.assertEqual(self.marketplace.publish(1, self.product2),
                         True, "incorrect return value for publish")
        self.assertEqual(self.marketplace.publish(1, self.product1),
                         True, "incorrect return value for publish")
        self.assertEqual(self.marketplace.publish(1, self.product2),
                         False, "incorrect return value for publish")
        self.assertEqual(self.marketplace.publish(2, self.product1),
                         True, "incorrect return value for publish")

    def test_new_cart(self):
        """Tests that cart IDs are generated sequentially."""
        self.assertEqual(self.marketplace.new_cart(), 1, "Error in new cart!")
        self.assertEqual(self.marketplace.new_cart(), 2, "Error in new cart!")
        self.assertEqual(self.marketplace.new_cart(), 3, "Error in new cart!")

    def test_add_to_cart(self):
        """Tests adding products to a cart."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product1)
        self.marketplace.publish(2, self.product2)
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product1)
        self.assertEqual(len(self.marketplace.carts_dict[1]), 1, "Error in add to cart!")
        # This add should fail because product1 is already "locked" by the first add.
        self.marketplace.add_to_cart(2, self.product1)
        self.assertEqual(len(self.marketplace.carts_dict[2]), 0, "Error in add to cart!")
        self.marketplace.add_to_cart(2, self.product2)
        self.assertEqual(len(self.marketplace.carts_dict[2]), 1, "Error in add to cart!")

    def test_remove_from_cart(self):
        """Tests removing products from a cart."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()

        self.marketplace.publish(1, self.product1)
        self.marketplace.publish(2, self.product2)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product1)
        self.marketplace.add_to_cart(1, self.product2)
        self.marketplace.remove_from_cart(1, self.product2)
        self.assertEqual(len(self.marketplace.carts_dict[1]), 1, "Error in add to cart!")
        self.marketplace.remove_from_cart(1, self.product1)
        self.assertEqual(len(self.marketplace.carts_dict[1]), 0, "Error in add to cart!")

    def place_order(self):
        """Tests that placing an order correctly depletes producer inventory."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product1)
        self.marketplace.publish(2, self.product2)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product1)
        self.marketplace.add_to_cart(1, self.product2)
        self.marketplace.place_order(1)
        self.assertEqual(len(self.marketplace.producers_dict[1]),
                         0, "Error in add to cart!")
        self.assertEqual(len(self.marketplace.producers_dict[2]),
                         0, "Error in add to cart!")


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that continuously supplies products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of (product, quantity, production_time) tuples.
            marketplace (Marketplace): A reference to the central marketplace.
            republish_wait_time (int): Time to wait before retrying a publish.
            **kwargs: Keyword arguments for the Thread parent class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # A producer gets a unique ID from the marketplace upon creation.
        self.id = self.marketplace.register_producer()

    def run(self):
        """The main execution logic for the producer thread."""
        while True:
            for product in self.products:
                quantity = product[1]
                while quantity > 0:
                    # Attempt to publish a product.
                    res = self.marketplace.publish(self.id, product[0])
                    
                    # If publishing fails (queue is full), wait and retry.
                    while not res:
                        sleep(self.republish_wait_time)
                        res = self.marketplace.publish(self.id, product[0])
                    
                    # On success, wait for the production time.
                    sleep(product[2])
                    quantity -= 1





from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Represents a Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Represents a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
