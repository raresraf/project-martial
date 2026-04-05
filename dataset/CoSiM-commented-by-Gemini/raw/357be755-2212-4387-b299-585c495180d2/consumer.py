"""A multi-threaded producer-consumer marketplace simulation.

This script implements a marketplace system where multiple Producer threads can
publish products and multiple Consumer threads can purchase them. This version is
notable for its concurrency model, which uses a single, coarse-grained lock
external to the Marketplace methods, managed by the client threads (Consumer
and Producer).
"""
#
# =============================================================================
#
#                                CONSUMER
#
# =============================================================================
from threading import Thread
from time import sleep


class Consumer(Thread):
    """Represents a consumer thread that processes a list of shopping carts.

    This Consumer implementation uses a single, coarse-grained lock on the
    marketplace to serialize all its operations, including retries.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping lists to process.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an operation.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.id = kwargs["name"]

    def run(self):
        """The main execution logic for the consumer thread.

        This method demonstrates a coarse-grained locking strategy. The thread
        acquires the marketplace's single lock before almost every interaction,
        including the wait-and-retry loop, effectively blocking all other
        producers and consumers during its operation.
        """
        for cart in self.carts:
            # Acquire lock to get a new cart ID.
            self.marketplace.lock.acquire()
            cart_id = self.marketplace.new_cart()
            self.marketplace.lock.release()

            # Process each operation (add/remove) in the shopping list.
            for operation in cart:
                op_type = operation['type']
                product = operation['product']
                quantity = operation['quantity']
                for i in range(quantity):
                    self.marketplace.lock.acquire()
                    if op_type == "add":
                        # If adding to cart fails, the thread sleeps and retries,
                        # all while holding the marketplace lock, which will
                        # block all other threads.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            self.marketplace.lock.release()
                            sleep(self.retry_wait_time)
                            self.marketplace.lock.acquire()
                    else:
                        self.marketplace.remove_from_cart(cart_id, product)
                    self.marketplace.lock.release()

            products = self.marketplace.place_order(cart_id)

            # Print the final purchased items.
            for product in products:
                print(self.id + " bought " + str(product))

#
# =============================================================================
#
#                                MARKETPLACE & TESTS
#
# =============================================================================
import logging
import time
import unittest
from threading import Lock

# Assumes these classes are defined elsewhere in the original project structure.
# from tema.product import Coffee, Product, Tea


class Marketplace:
    """Manages product inventory and transactions.

    This implementation of the marketplace is NOT internally thread-safe. It
    relies entirely on client-side locking, exposing a single `lock` attribute
    for producers and consumers to manage concurrency.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The max number of products a
                single producer can have listed at once.
        """
        self.max_queue_size = queue_size_per_producer
        self.available_products = {} # {producer_id: {product: count}}
        self.no_available_products = {} # {producer_id: total_count}
        self.carts_in_use = {} # {cart_id: [(product, producer_id)]}
        self.last_cart_id = -1
        self.last_producer_id = 0

        # The single lock for clients to manage concurrency.
        self.lock = Lock()
        
        # Set up file-based logging for all marketplace events.
        logging.Formatter.converter = time.gmtime
        logging.basicConfig(filename="marketplace.log",
                            filemode="a",
                            format='%(asctime)s,%(msecs)d %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger()

    def add_available_product(self, producer_id, product):
        """Internal helper to add a product to a producer's available stock."""
        self.logger.info(
            'Entered function "add_available_product" with parameter: ' + str(producer_id) + ' and ' + str(product))
        if product not in self.available_products[producer_id]:
            self.available_products[producer_id][product] = 1
        else:
            self.available_products[producer_id][product] += 1
        self.no_available_products[producer_id] += 1
        self.logger.info('Exit function "add_available_product"')

    def register_producer(self):
        """Registers a new producer and returns a unique ID. NOT thread-safe."""
        self.logger.info('Entered function "register_producer"')
        self.last_producer_id += 1
        self.no_available_products[self.last_producer_id] = 0
        self.available_products[self.last_producer_id] = {}
        self.logger.info('Exit function "register_producer" with value: ' + str(self.last_producer_id))
        return self.last_producer_id

    def publish(self, producer_id, product):
        """Publishes a product from a producer. NOT thread-safe."""
        self.logger.info('Entered function "publish" with parameters ' + str(producer_id) + ' and  ' + str(product))
        if self.no_available_products[producer_id] >= self.max_queue_size:
            self.logger.info('Exit function "publish" with value: False')
            return False
        self.add_available_product(producer_id, product)
        self.logger.info('Exit function "publish" with value: True')
        return True

    def new_cart(self):
        """Creates a new cart. NOT thread-safe."""
        self.logger.info('Entered function "new_cart"')
        self.last_cart_id += 1
        self.carts_in_use[self.last_cart_id] = []
        self.logger.info('Exit function "new_cart" with value: ' + str(self.last_cart_id))
        return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        """Adds a product to a cart if available. NOT thread-safe."""
        self.logger.info('Entered function "add_to_cart" with parameters: ' + str(cart_id) + ' and  ' + str(product))
        for producer in self.available_products.keys():
            if product in self.available_products[producer]:
                # Decrement stock and move product to the cart.
                number_of_products = self.available_products[producer][product]
                number_of_products -= 1
                if number_of_products <= 0:
                    del self.available_products[producer][product]
                else:
                    self.available_products[producer][product] = number_of_products
                self.carts_in_use[cart_id].append((product, producer))
                self.no_available_products[producer] -= 1
                self.logger.info('Exit function "add_to_cart" with value: True')
                return True
        self.logger.info('Exit function "add_to_cart" with value: False')
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to stock. NOT thread-safe."""
        self.logger.info(
            'Entered function "remove_from_cart" with parameters: ' + str(cart_id) + ' and ' + str(product))
        if cart_id not in self.carts_in_use:
            return
        for product_producer in self.carts_in_use[cart_id]:
            current_product, current_producer = product_producer
            if current_product == product:
                self.add_available_product(current_producer, product)
                self.carts_in_use[cart_id].remove(product_producer)
                break
        self.logger.info('Exit function "remove_from_cart"')

    def place_order(self, cart_id):
        """Finalizes an order. NOT thread-safe."""
        self.logger.info('Entered function "place_order" + with parameter: ' + str(cart_id))
        if cart_id not in self.carts_in_use:
            self.logger.info('Exit function "place_order" with value: []')
            return []
        self.logger.info('Exit function "place_order" with value: ' + str(self.carts_in_use[cart_id]))
        product_list = [prod for prod, _ in self.carts_in_use[cart_id]]
        return product_list


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Set up a new Marketplace for each test."""
        self.marketplace = Marketplace(1)
        # Mock product classes for testing since they are not fully included.
        class Tea: pass
        class Coffee: pass
        self.coffee1 = Coffee()
        self.coffee2 = Coffee()
        self.tea = Tea()
        self.coffee1.name, self.coffee1.price = "Ethiopia", 10
        self.coffee2.name, self.coffee2.price = "China", 10
        self.tea.name, self.tea.price = "Linden", 9

    def test_register_producer(self):
        """Tests that producers receive sequential IDs starting from 1."""
        id = self.marketplace.register_producer()
        self.assertEqual(1, id)

    def test_publish(self):
        """Tests that a producer can publish up to their queue limit."""
        id = self.marketplace.register_producer()
        self.marketplace.publish(id, self.coffee1)
        # This second publish should fail as the queue size is 1.
        self.marketplace.publish(id, self.coffee2)
        self.assertEqual(1, self.marketplace.available_products[id][self.coffee1])
        self.assertEqual(False, self.coffee2 in self.marketplace.available_products[id])

    def test_new_cart(self):
        """Tests that new carts receive sequential IDs starting from 0."""
        id = self.marketplace.new_cart()
        self.assertEqual(id, 0)

    def test_add_to_cart(self):
        """Tests adding an available product to a cart."""
        id_cart = self.marketplace.new_cart()
        id_producer = self.marketplace.register_producer()
        self.marketplace.publish(id_producer, self.tea)
        self.marketplace.add_to_cart(id_cart, self.tea)
        self.assertEqual(True, (self.tea, id_producer) in self.marketplace.carts_in_use[id_cart])

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        id_cart = self.marketplace.new_cart()
        id_producer = self.marketplace.register_producer()
        self.marketplace.publish(id_producer, self.tea)
        self.marketplace.add_to_cart(id_cart, self.tea)
        self.marketplace.remove_from_cart(id_cart, self.tea)
        self.assertEqual(False, (self.tea, id_producer) in self.marketplace.carts_in_use[id_cart])

    def test_place_order(self):
        """Tests that placing an order returns the correct list of products."""
        id_cart = self.marketplace.new_cart()
        id_producer1 = self.marketplace.register_producer()
        id_producer2 = self.marketplace.register_producer()
        self.marketplace.publish(id_producer1, self.coffee1)
        self.marketplace.publish(id_producer2, self.coffee2)
        self.marketplace.add_to_cart(id_cart, self.coffee1)
        self.marketplace.add_to_cart(id_cart, self.coffee2)
        order_list = self.marketplace.place_order(id_cart)
        self.assertEqual([self.coffee1, self.coffee2], order_list)

    def test_add_available_product(self):
        """Tests the internal helper for adding a product."""
        id = self.marketplace.register_producer()
        self.marketplace.add_available_product(id, self.coffee1)
        self.assertEqual(1, self.marketplace.available_products[id][self.coffee1])


#
# =============================================================================
#
#                                PRODUCER
#
# =============================================================================
class Producer(Thread):
    """Represents a producer thread that generates and publishes products.

    This Producer implementation uses a single, coarse-grained lock on the
    marketplace to serialize all its operations.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer instance.

        Args:
            products (list): A list of products for the producer to generate.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait after a successful publish.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id = -1

    def run(self):
        """The main execution logic for the producer thread.

        The thread first registers itself with the marketplace, then enters an
        infinite loop to produce and publish items. It uses the marketplace's
        single lock to protect both registration and publishing actions.
        """
        self.marketplace.lock.acquire()
        self.id = self.marketplace.register_producer()
        self.marketplace.lock.release()

        while True:
            for product in self.products:
                real_product, quantity, time_to_produce = product
                for i in range(quantity):
                    sleep(time_to_produce) # Simulate production time.
                    self.marketplace.lock.acquire()
                    # If publishing fails, release the lock, wait, and retry.
                    while not self.marketplace.publish(self.id, real_product):
                        self.marketplace.lock.release()
                        sleep(self.republish_wait_time)
                        self.marketplace.lock.acquire()
                    self.marketplace.lock.release()

#
# =============================================================================
#
#                                PRODUCTS
#
# =============================================================================
from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing a Tea product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing a Coffee product."""
    acidity: str
    roast_level: str
