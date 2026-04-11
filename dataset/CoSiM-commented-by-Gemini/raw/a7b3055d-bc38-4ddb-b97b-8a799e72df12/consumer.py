"""
This module contains several concatenated versions of a producer-consumer
marketplace simulation. The file is disorganized and contains multiple, slightly
different definitions for classes like `Marketplace` and `Producer`.

The documentation will proceed linearly through the file, commenting on each
component as it appears, while noting the duplicated and overlapping definitions.

Core components defined (multiple times) in this file:
- Marketplace: A thread-safe class to manage producers, products, and carts.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places an order.
- TestMarketplace: A suite of unit tests.
- Product/Coffee/Tea: Dataclasses for products.
"""


from threading import Thread, currentThread, Lock
import time
import unittest
import logging
from logging.handlers import RotatingFileHandler

# NOTE: The first Consumer class appears to correspond to the SECOND Marketplace class definition below.

class Consumer(Thread):
    """
    Represents a consumer thread that shops in the marketplace.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists for the consumer to process.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): Time to wait before retrying to add a product.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.
        
        It gets a new cart ID, processes all add/remove operations, and then
        places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                op_type = operation['type']
                wanted_product = operation['product']
                wanted_quantity = operation['quantity']
                current_quantity = 0

                # Invariant: Loop until the desired quantity for the operation is met.
                while current_quantity < wanted_quantity:
                    can_do_op = None
                    if op_type == "add":
                        # Persistently try to add the product.
                        can_do_op = self.marketplace.add_to_cart(cart_id, wanted_product)
                    elif op_type == "remove":
                        self.marketplace.remove_from_cart(cart_id, wanted_product)

                    if can_do_op is False:
                        # If adding failed, wait and retry.
                        time.sleep(self.retry_wait_time)
                    else:
                        current_quantity += 1
            
            # Finalize the purchase.
            self.marketplace.place_order(cart_id)


# =============================================================================
# TestMarketplace and the first Marketplace definition
# =============================================================================

logging.basicConfig(
        handlers=[RotatingFileHandler('./marketplace.log', maxBytes=100000, backupCount=10,
                mode='a')],
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s")
logging.Formatter.converter = time.gmtime
logger = logging.getLogger()

class TestMarketplace(unittest.TestCase):
    """
    A suite of unit tests for one of the Marketplace implementations in this file.
    
    Note: These tests appear to correspond to the second `Marketplace` class
    definition found later in the file.
    """
    @classmethod
    def setUpClass(cls):
        print('SetUpClass')

    def setUp(self):
        """Initializes a marketplace and products for each test."""
        # This Marketplace instance corresponds to the second definition below.
        self.marketplace = Marketplace(15)
        self.prod1 = { "product_type": "Coffee", "name": "Indonezia", "acidity": 5.05,
                        "roast_level": "MEDIUM", "price": 1 }
        self.prod2 = { "product_type": "Tea", "name": "Linden", "type": "Herbal", "price": 9 }

    def test_register_producer(self):
        """Tests that producer IDs are generated sequentially."""
        print('
Test Register Producer
')
        for new_id in range(3):
            self.assertEqual(self.marketplace.register_producer(), new_id)

    def test_publish(self):
        """Tests that publishing succeeds and fails correctly based on queue size."""
        print('
Test Publish
')
        pid = self.marketplace.register_producer()
        # This test assumes a queue size of 15, so it should pass.
        for _ in range(self.marketplace.queue_size_per_producer):
            self.assertTrue(self.marketplace.publish(pid, self.prod1))
        # This should fail if the queue size was 15 and the loop ran 15 times.
        # self.assertFalse(self.marketplace.publish(pid, self.prod2))

    def test_new_cart(self):
        """Tests that cart IDs are generated sequentially."""
        print('
Test New Cart
')
        for i in range(3):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_add_to_cart(self):
        """Tests that a product can be added to a cart."""
        print('
Test Add
')
        pid = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(pid, self.prod2)
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.prod2))
        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.prod1))

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        print('
Test Remove
')
        pid = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(pid, self.prod1)
        self.marketplace.add_to_cart(cart_id, self.prod1)
        prod1_occurences_cart = self.marketplace.carts[cart_id].count((self.prod1, pid))
        self.marketplace.remove_from_cart(cart_id, self.prod1)
        new_prod1_occurences_cart = self.marketplace.carts[cart_id].count((self.prod1, pid))
        self.assertLess(new_prod1_occurences_cart, prod1_occurences_cart)

    def test_place_order(self):
        """Tests the final placing of an order."""
        print('
Test Place Order
')
        pid = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        expected_cart = [self.prod1, self.prod2]
        self.marketplace.publish(pid, self.prod1)
        self.marketplace.publish(pid, self.prod2)
        self.marketplace.add_to_cart(cart_id, self.prod1)
        self.marketplace.add_to_cart(cart_id, self.prod2)
        res = self.marketplace.place_order(cart_id)
        # Sort for comparison as order is not guaranteed.
        self.assertEqual(sorted(res, key=lambda x: x['name']), sorted(expected_cart, key=lambda x: x['name']))

# This appears to be the primary `Marketplace` class intended for use by the tests.
class Marketplace:
    """
    Manages inventory and transactions in a thread-safe manner.

    This implementation uses dictionaries to map IDs to producer/cart lists and
    employs a set of locks to manage concurrent access to different operations.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {}
        self.num_producers = 0
        self.carts = {}
        self.num_carts = 0
        # A set of locks for different critical sections.
        self.producer_reg = Lock()
        self.cart_reg = Lock()
        self.print_res = Lock()
        self.add_prod = Lock()
        self.remove_prod = Lock()
        logging.info("Set up Marketplace with queue_size_per_producer = %s", queue_size_per_producer)

    def register_producer(self):
        """Registers a new producer with a unique ID."""
        with self.producer_reg:
            curr_producer_id = self.num_producers
            self.producers[self.num_producers] = []
            self.num_producers += 1
            logging.info('Producer registered with id = %s', curr_producer_id)
        return curr_producer_id

    def publish(self, producer_id, product):
        """Publishes a product for a given producer."""
        logging.info("Producer with id = %s wants to publish product = %s", producer_id, product)
        # This method is not thread-safe as it accesses `self.producers` outside a lock.
        if len(self.producers[producer_id]) == self.queue_size_per_producer:
            logging.info("Producer with id = %s can't publish %s", producer_id, product)
            return False
        self.producers[producer_id].append(product)
        logging.info("Producer with id = %s published product = %s", producer_id, product)
        return True

    def new_cart(self):
        """Creates a new cart and returns its unique ID."""
        with self.cart_reg:
            curr_cart_id = self.num_carts
            self.carts[self.num_carts] = []
            self.num_carts += 1
        logging.info("Consumer has cart registered cart with id = %s ", curr_cart_id)
        return curr_cart_id

    def add_to_cart(self, cart_id, product):
        """Adds a product to a cart if available, removing it from the producer's stock."""
        with self.add_prod:
            logging.info("Consumer with cart_id = %s wants to add product = %s", cart_id, product)
            # Find the product across all producers.
            for p_id in range(self.num_producers):
                if product in self.producers[p_id]:
                    self.producers[p_id].remove(product)
                    self.carts[cart_id].append((product, p_id))
                    return True
            return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the original producer's stock."""
        with self.remove_prod:
            logging.info("Consumer with cart_id = %s wants to remove product = %s", cart_id, product)
            p_id = -1
            # Find the product in the cart to identify its original producer.
            for elem in self.carts[cart_id]:
                if product == elem[0]:
                    self.producers[elem[1]].append(product)
                    p_id = elem[1]
                    break
            if p_id != -1:
                self.carts[cart_id].remove((product, p_id))

    def place_order(self, cart_id):
        """Finalizes an order and returns the list of bought products."""
        bought_products = []
        with self.print_res:
            logging.info("Consumer with cart_id = %s place order", cart_id)
            for elem in self.carts[cart_id]:
                thread_name = currentThread().getName()
                product = elem[0]
                print(f"{thread_name} bought {product}")
                bought_products.append(elem[0])
            logging.info("Consumer with cart_id = %s has products %s", cart_id, str(bought_products))
        return bought_products

# =============================================================================
# Second Producer and Product definitions
# =============================================================================

class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    (This is a second, slightly different definition of the Producer class).
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = -1

    def run(self):
        """The main execution logic for the producer thread."""
        self.producer_id = self.marketplace.register_producer()
        # Invariant: The producer runs in an infinite loop.
        while True:
            for (product, quantity, wait_time) in self.products:
                for _ in range(quantity):
                    can_publish = self.marketplace.publish(self.producer_id, product)
                    if can_publish is True:
                        time.sleep(wait_time)
                    else:
                        # If publishing fails, wait and retry.
                        time.sleep(self.republish_wait_time)

from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple dataclass for a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for a Tea product, adding a 'type' attribute."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for a Coffee product, adding acidity and roast level."""
    acidity: str
    roast_level: str
