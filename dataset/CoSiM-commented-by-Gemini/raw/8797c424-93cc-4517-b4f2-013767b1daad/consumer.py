"""
This module contains a full simulation of a producer-consumer marketplace.

It includes all components for the simulation:
- The `Marketplace` class, which acts as the central, thread-safe hub.
- `Producer` and `Consumer` thread classes to simulate market participants.
- A `get_log` function for creating a configured logger.
- A `TestMarketplace` unit test suite for validation.
- Dataclass definitions for `Product`, `Coffee`, and `Tea`.
"""


from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer thread that purchases products from the marketplace.

    Each consumer is initialized with a set of shopping lists (carts) and
    executes the actions (add/remove) in each list before placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists. Each list contains
                          dictionaries specifying products, quantities, and actions.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying to
                                     add a product if it is unavailable.
            **kwargs: Keyword arguments for the parent Thread class, including 'name'.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs["name"]

    def run(self):
        """
        The main execution method for the consumer thread.

        Iterates through its assigned shopping lists, creates a cart for each,
        executes the add/remove operations, and finalizes by placing an order.
        """
        # Invariant: Process all shopping lists assigned to this consumer.
        for cart in self.carts:

            cart_id = self.marketplace.new_cart()

            for data in cart:
                i = 0
                item = data["product"]
                operation = data["type"]

                # Pre-condition: The operation must be performed for the specified quantity.
                while i < data["quantity"]:

                    if operation == "add":
                        available = self.marketplace.add_to_cart(cart_id, item)
                        if available:
                            i += 1
                        else:
                            # If the product is not available, wait and retry.
                            time.sleep(self.retry_wait_time)

                    if operation == "remove":
                        self.marketplace.remove_from_cart(cart_id, item)
                        i += 1

            order = self.marketplace.place_order(cart_id)

            # After placing the order, print the items that were bought.
            self.marketplace.print_list(order, self.consumer_name)

from logging.handlers import RotatingFileHandler
import logging
import time
import os
if not os.path.exists("Logs"):
    os.makedirs("Logs")

def get_log(name):
    """
    Factory function to create and configure a logger.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A configured logger instance with a rotating file handler.
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    
    logging.Formatter.converter = time.gmtime
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    
    handler = RotatingFileHandler('Logs/marketplace.log', maxBytes=2000, backupCount=20)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)

    return logger

from threading import Lock
import unittest
import io
import sys
sys.path.append("tema")
from logger import get_log

LOGGER = get_log('marketplace.log')

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Sets up the test fixture before each test method."""
        self.marketplace = Marketplace(15)
        self.product_1 = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        self.product_2 = {
            "product_type": "Tea",
            "name": "Bubble Tea",
            "price": 10
        }
        self.producer_id = self.marketplace.register_producer()
        self.cart_id = self.marketplace.new_cart()
        self.cart_id_2 = self.marketplace.new_cart()

    def test_register_producer(self):
        """Tests that producer registration works as expected."""
        print("\nTesting register_producer")
        self.assertIsNotNone(self.producer_id)
        self.assertEqual(self.producer_id, 0)
        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 0)
        self.assertEqual(self.marketplace.items[self.producer_id], [])

    def test_publish(self):
        """Tests the product publishing logic, including capacity limits."""
        print("\nTesting publish")
        self.assertEqual(self.marketplace.publish(self.producer_id, self.product_1), True)
        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 1)
        self.assertEqual(self.marketplace.items[self.producer_id], [self.product_1])
        # Manually set item count high to test the queue limit.
        self.marketplace.prod_num_items[self.producer_id] = 1000
        self.assertEqual(self.marketplace.publish(self.producer_id, self.product_1), False)

    def test_new_cart(self):
        """Tests that new cart creation returns sequential IDs."""
        print("\nTesting new_cart")
        self.assertIsNotNone(self.cart_id)
        self.assertEqual(self.cart_id, 0)
        self.assertEqual(self.marketplace.carts[self.cart_id], [])
        self.assertIsNotNone(self.cart_id_2)
        self.assertEqual(self.cart_id_2, 1)
        self.assertEqual(self.marketplace.carts[self.cart_id_2], [])

    def test_add_to_cart(self):
        """Tests adding available products to a cart."""
        print("\nTesting add_to_cart")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.assertEqual(self.marketplace.add_to_cart(self.cart_id, self.product_1), True)

        # The product should now be moved from the producer's items to the cart.
        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 0)
        self.assertEqual(self.marketplace.items[self.producer_id], [])

        self.assertEqual(self.marketplace.carts[self.cart_id], [(self.product_1, self.producer_id)])
        # Adding an unpublished product should fail.
        self.assertEqual(self.marketplace.add_to_cart(self.cart_id, self.product_2), False)

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        print("\nTesting remove_from_cart")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        self.marketplace.remove_from_cart(self.cart_id, self.product_1)

        # The product should be moved back to the producer's item list.
        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 1)
        self.assertEqual(self.marketplace.items[self.producer_id], [self.product_1])
        self.assertEqual(self.marketplace.carts[self.cart_id], [])

    def test_place_order(self):
        """Tests the order placement logic."""
        print("\nTesting place_order")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.publish(self.producer_id, self.product_2)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id_2, self.product_2)
        order_1 = []
        order_2 = []

        order_1 = self.marketplace.place_order(self.cart_id)
        self.assertEqual(order_1, [(self.product_1, self.producer_id)])
        # The first cart should be gone after placing the order.
        self.assertEqual(self.marketplace.carts,
                         {self.cart_id_2: [(self.product_2, self.producer_id)]})

        order_2 = self.marketplace.place_order(self.cart_id_2)
        self.assertEqual(order_2, [(self.product_2, self.producer_id)])
        # All carts should be gone.
        self.assertEqual(self.marketplace.carts, {})

        self.assertIsNotNone(order_1)
        self.assertIsNotNone(order_2)
        self.assertNotEqual(order_1, {})
        self.assertNotEqual(order_2, {})

    def test_print_list(self):
        """Tests that the print_list method outputs the correct format."""
        cons_name = "Consumer 1"
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        order = self.marketplace.place_order(self.cart_id)

        # Capture stdout to verify the print output.
        output = io.StringIO()
        sys.stdout = output
        self.marketplace.print_list(order, cons_name)
        sys.stdout = sys.__stdout__
        self.assertEqual(output.getvalue(),
                         'Consumer 1 bought {\'product_type\': \'Coffee\','
                         '\'name\': \'Indonezia\', \'acidity\': 5.05,'
                         '\'roast_level\': \'MEDIUM\','
                         '\'price\': 1}\n')

class Marketplace:
    """
    A thread-safe marketplace for producers to publish and consumers to buy products.

    This class orchestrates the entire simulation, managing producer inventories,
    available products, and customer shopping carts. It uses various locks to
    ensure that concurrent operations from multiple threads are handled safely.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace.
        """
        LOGGER.info('creating an instance of Marketplace')
        LOGGER.info('Max size of queue in Marketplace: %d', queue_size_per_producer)



        self.queue_size_per_producer = queue_size_per_producer

        self.num_prod = 0
        self.num_carts = 0
        
        # Stores the current number of items for each producer. Index is producer_id.
        self.prod_num_items = []
        
        # Stores the actual items for each producer. Key is producer_id.
        self.items = {}
        
        # Stores the items in each cart. Key is cart_id.
        # The value is a list of (product, producer_id) tuples.
        self.carts = {}

        self.register_lock = Lock()
        self.new_cart_lock = Lock()
        self.cart_lock = Lock()
        
        self.print_lock = Lock()



    def register_producer(self):
        """
        Registers a new producer, returning a unique producer ID.

        This operation is thread-safe.

        Returns:
            int: The unique ID for the new producer.
        """
        LOGGER.info("In method 'register_producer' from class Marketplace")


        with self.register_lock:
            prod_id = self.num_prod
            self.num_prod += 1

        self.prod_num_items.append(0)
        self.items[prod_id] = []
        LOGGER.info("Output of 'register_producer' - producer id: %d", prod_id)
        return prod_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace.

        Args:
            producer_id (int): The ID of the producer.
            product: The product to be published.

        Returns:
            bool: True if successful, False if the producer's queue is full.
        """
        LOGGER.info("In method 'publish' from class Marketplace\
                    \nInputs: producer_id =%s; product=%s",
                    producer_id, product)
        if self.prod_num_items[producer_id] >= self.queue_size_per_producer:
            LOGGER.info("Output of 'publish' - %s", "False")
            return False
        self.items[producer_id].append(product)
        self.prod_num_items[producer_id] += 1
        LOGGER.info("Output of 'publish' - %s", "True")
        return True

    def new_cart(self):
        """
        Creates a new shopping cart and returns its ID.

        This operation is thread-safe.

        Returns:
            int: The unique ID for the new cart.
        """
        LOGGER.info("In method 'new_cart' from class Marketplace")
        with self.new_cart_lock:
            cart_id = self.num_carts
            self.num_carts += 1

        self.carts[cart_id] = []

        LOGGER.info("Output of 'new_cart' - cart_id = %s", cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds an available product to a shopping cart.

        Note: This implementation searches through all producer inventories to find
        an available product, which can be inefficient with many producers/products.

        Args:
            cart_id (int): The ID of the target cart.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        LOGGER.info("In method 'add_to_cart' from class Marketplace\nInputs:\
        cart_id =%s; product=%s", cart_id, product)
        found = False
        with self.cart_lock:
            # Invariant: Search all producer queues for the requested product.
            for i, (_, val) in enumerate(self.items.items()):
                if product in val:


                    val.remove(product)
                    self.prod_num_items[i] -= 1
                    prod_id = i
                    found = True
                    break

        if found:
            self.carts[cart_id].append((product, prod_id))

        LOGGER.info("Output of 'add_to_cart' - %s", found)
        return found

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the marketplace.

        Args:
            cart_id (int): The ID of the cart.
            product: The product to remove.
        """
        LOGGER.info("In method 'remove_from_cart' from class Marketplace\nInputs:\
        cart_id =%s; product=%s", cart_id, product)


        # Invariant: The product must exist in the cart to be removed.
        for item, producer in self.carts[cart_id]:
            if item is product:
                prod_id = producer
                self.carts[cart_id].remove((item, producer))
                break

        self.items[prod_id].append(product)

        with self.cart_lock:
            self.prod_num_items[prod_id] += 1
        LOGGER.info("Finished 'remove_from_cart', no return value")

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This method removes the cart from the active carts in the marketplace.
        Note: This does not adjust the producer's published item count; the
        items are effectively consumed and removed from the simulation.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: The list of (product, producer_id) tuples that were in the cart.
        """
        LOGGER.info("In method 'place_order' from class Marketplace\
        \nInputs:cart_id =%s", cart_id)
        res = self.carts.pop(cart_id)
        LOGGER.info("Output of 'place_order' - res = %s", res)
        return res

    def print_list(self, order, consumer_name):
        """
        Prints the contents of a finalized order for a given consumer.

        This operation is thread-safe to prevent interleaved print statements.

        Args:
            order (list): The list of items from a placed order.
            consumer_name (str): The name of the consumer who made the purchase.
        """
        LOGGER.info("In method 'print_list' from class Marketplace\
        \nInputs:order =%s; consumer_name: %s", order, consumer_name)
        for item in order:
            with self.print_lock:
                print(consumer_name + " bought "+ str(item[0]))
        LOGGER.info("Finished 'print_list', no return value")


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.

    The producer runs in an infinite loop, attempting to publish its assigned
    products according to the specified quantities and timings.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list where each element is a tuple of
                             (product, quantity, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the producer's queue is full.
            **kwargs: Keyword arguments for the parent Thread class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products


        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and enters an infinite loop to publish
        its products.
        """
        prod_id = self.marketplace.register_producer()
        # Invariant: This producer will run indefinitely.
        while True:
            # Invariant: Loop through each type of product this producer can make.
            for (item, quantity, wait_time) in self.products:
                i = 0
                # Invariant: Publish the specified quantity of the current product.
                while i < quantity:
                    available = self.marketplace.publish(prod_id, item)

                    if available:
                        # If publish is successful, wait before producing the next one.
                        time.sleep(wait_time)
                        i += 1
                    else:
                        # If the producer's queue is full, wait and retry.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
