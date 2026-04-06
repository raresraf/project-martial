"""
This module simulates a multi-threaded producer-consumer model for an e-commerce
marketplace.

It defines all the core components for the simulation:
- Consumer: A thread that simulates a customer adding items to a cart and buying them.
- Marketplace: The central, thread-safe class that coordinates producers and
  consumers, managing product inventories and customer carts.
- Producer: A thread that continuously creates products and adds them to the marketplace.
- Product (Tea, Coffee): Dataclasses representing the items being sold.
- TestMarketplace: Unit tests to verify the functionality of the Marketplace.
"""


from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer thread that simulates a customer's shopping process.

    Each consumer is given a list of shopping carts (which are lists of actions)
    and executes them against the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts", where each cart is a list of
                          operations (add/remove).
            marketplace (Marketplace): A reference to the central marketplace object.
            retry_wait_time (int): Time in seconds to wait before retrying to
                                   add a product if it's not available.
            **kwargs: Keyword arguments for the Thread parent class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_operation(self, quantity, cart_id, product):
        """
        Helper method to handle adding one or more items of a product to a cart.
        
        If the marketplace cannot add the item immediately (e.g., out of stock),
        this method will pause and retry until it succeeds.
        """
        for _ in range(quantity):
            while not self.marketplace.add_to_cart(cart_id, product):
                time.sleep(self.retry_wait_time)

    def remove_operation(self, quantity, cart_id, product):
        """
        Helper method to handle removing one or more items of a product from a cart.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """The main execution logic for the consumer thread."""
        for cart in self.carts:
            id_cart = self.marketplace.new_cart()
            for operation in cart:
                if operation["type"] == "add":
                    self.add_operation(operation["quantity"], id_cart, operation["product"])
                else:
                    self.remove_operation(operation["quantity"], id_cart, operation["product"])

            self.marketplace.place_order(id_cart)


from threading import Lock, currentThread
import unittest
import logging
import time
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

# --- Global Logger Setup ---
# Sets up a rotating file logger to record all marketplace activities.
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

FORMATTER = logging.Formatter('$asctime : $levelname : $name : $message', style='$')
HANDLER = RotatingFileHandler('marketplace.log', maxBytes=20000, backupCount=5)
HANDLER.setFormatter(FORMATTER)

FORMATTER.converter = time.gmtime

LOGGER.addHandler(HANDLER)


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Prepares a new marketplace instance and sample data for each test."""
        self.queue_size_per_producer = 15
        self.marketplace = Marketplace(self.queue_size_per_producer)
        self.marketplace.producer_id = 3
        self.marketplace.cart_id = 5
        self.products = [Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM'),
                         Tea(name='Linden', price=9, type='Herbal'),
                         Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM'),
                         Tea(name='Wild Cherry', price=5, type='Black'),
                         Tea(name='Cactus fig', price=3, type='Green'),
                         Coffee(name='Ethiopia', price=10, acidity=5.09, roast_level='MEDIUM')]


    def test_register_producer(self):
        """Tests that producer IDs are assigned sequentially."""
        self.assertIsNotNone(self.marketplace.register_producer())
        self.assertEqual(self.marketplace.register_producer(), 5)

    def test_publish(self):
        """Tests that products can be successfully published to a producer's inventory."""
        producer_id = self.marketplace.register_producer()
        dict1 = {producer_id: {self.products[0]: 2}}
        self.marketplace.products[producer_id][self.products[0]] = 1

        self.assertTrue(self.marketplace.publish(producer_id, self.products[0]))
        self.assertDictEqual(self.marketplace.products, dict1)

    def test_new_cart(self):
        """Tests that cart IDs are assigned sequentially."""
        self.assertIsNotNone(self.marketplace.new_cart())
        self.assertEqual(self.marketplace.new_cart(), 7)

    def test_add_to_cart(self):
        """Tests adding available and unavailable products to a cart."""
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.publish(producer_id, self.products[1])
        cart_id = self.marketplace.new_cart()
        carts = {cart_id: {(self.products[1], producer_id): 1}}

        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.products[3]))
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.products[1]))
        self.assertDictEqual(self.marketplace.carts[cart_id], carts[cart_id])

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.publish(producer_id, self.products[1])
        self.marketplace.publish(producer_id, self.products[1])

        self.marketplace.add_to_cart(cart_id, self.products[0])
        self.marketplace.add_to_cart(cart_id, self.products[1])
        self.marketplace.add_to_cart(cart_id, self.products[1])
        self.marketplace.remove_from_cart(cart_id, self.products[0])
        carts = {cart_id: {(self.products[1], producer_id): 2}}

        self.assertDictEqual(self.marketplace.carts[cart_id], carts[cart_id])

    def test_place_order(self):
        """Tests that placing an order correctly frees up producer publication slots."""
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.publish(producer_id, self.products[1])
        self.marketplace.publish(producer_id, self.products[1])

        self.marketplace.add_to_cart(cart_id, self.products[0])
        self.marketplace.add_to_cart(cart_id, self.products[1])
        self.marketplace.add_to_cart(cart_id, self.products[1])
        self.marketplace.remove_from_cart(cart_id, self.products[0])
        self.marketplace.place_order(cart_id)

        self.assertEqual(self.marketplace.products_published[producer_id], 1)

class Marketplace:
    """
    The central, thread-safe hub for coordinating producers and consumers.
    
    This class manages all state for the simulation, including producer inventories,
    shopping carts, and ID generation. It uses locks to ensure that concurrent
    access from multiple threads does not lead to race conditions.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        
        Args:
            queue_size_per_producer (int): The maximum number of products a
                                           producer can have published at once.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.consumer_lock = Lock()
        self.producer_lock = Lock()
        self.products = {}  # {producer_id: {product: quantity}}
        self.carts = {}  # {cart_id: {(product, producer_id): quantity}}
        self.products_published = {}  # {producer_id: count}
        self.producer_id = -1
        self.cart_id = -1

    def register_producer(self):
        """Atomically registers a new producer and returns a unique ID."""
        with self.producer_lock:
            LOGGER.info('[OLD]Last producer id:%d', self.producer_id)
            self.producer_id += 1
            self.products_published[self.producer_id] = 0
            self.products[self.producer_id] = {}
            LOGGER.info('[UPDATE]New producer id:%d', self.producer_id)
            return self.producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product if they have capacity.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        with self.producer_lock:
            LOGGER.info('[INPUT]Producer_id: %s and Product: %s', producer_id, product)
            res = False
            # Pre-condition: Check if producer is below their publication limit.
            if self.products_published[producer_id] <= self.queue_size_per_producer:
                self.products_published[producer_id] += 1
                if product in list(self.products[producer_id].keys()):
                    self.products[producer_id][product] += 1
                else:
                    self.products[producer_id][product] = 1
                res = True

            LOGGER.info('[OUTPUt]Method returns: %r', res)
            return res

    def new_cart(self):
        """Atomically creates a new empty cart and returns its ID."""
        with self.consumer_lock:
            LOGGER.info('[OLD]:Cart_id: %d', self.cart_id)
            self.cart_id += 1
            self.carts[self.cart_id] = {}
            LOGGER.info('[UPDATE]:Cart_id: %d', self.cart_id)
            return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Searches for a product from any producer and adds it to the cart.
        
        This method is atomic with respect to other consumer operations.
        
        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        with self.consumer_lock:
            LOGGER.info('[INPUT]Cart_id: %d and Product: %s', cart_id, product)

            # Invariant: Search all producers for the product.
            for producer in list(self.products.keys()):
                if product in list(self.products[producer].keys()):
                    # Decrement producer's stock.
                    self.products[producer][product] -= 1
                    # Add product to cart, tracking its origin producer.
                    if (product, producer) in list(self.carts[cart_id].keys()):
                        self.carts[cart_id][(product, producer)] += 1
                    else:
                        self.carts[cart_id][(product, producer)] = 1
                    # Remove product from producer's dict if stock is depleted.
                    if self.products[producer][product] == 0:
                        self.products[producer].pop(product, 0)
                    LOGGER.info('[OUTPUT]Method returns: True')
                    return True

        LOGGER.info('[OUTPUT]Method returns: False')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the producer's inventory.
        
        This method is atomic with respect to other consumer operations.
        """
        with self.consumer_lock:
            LOGGER.info('[INPUT]Cart_id: %d and Product: %s', cart_id, product)
            # Find the product in the cart, identified by its (product, producer_id) tuple key.
            for (prod, producer_id) in list(self.carts[cart_id].keys()):
                if prod == product:
                    # Decrement cart quantity.
                    self.carts[cart_id][(product, producer_id)] -= 1
                    if self.carts[cart_id][(product, producer_id)] == 0:
                        self.carts[cart_id].pop((product, producer_id), 0)
                    # Return item to producer's inventory.
                    if product in list(self.products[producer_id].keys()):
                        self.products[producer_id][product] += 1
                    else:
                        self.products[producer_id][product] = 1
                    break

    def place_order(self, cart_id):
        """
        Finalizes an order, printing purchased items and freeing producer slots.
        
        This method is atomic with respect to other consumer operations.
        """
        with self.consumer_lock:
            LOGGER.info('[INPUT]Place order for cart_id: %d', cart_id)
            # Iterate through all items in the cart.
            for (product, producer), quantity in self.carts[cart_id].items():
                for _ in range(quantity):
                    # Decrement the producer's published count, freeing up a slot.
                    self.products_published[producer] -= 1
                    print(f"{currentThread().getName()} bought {product}")
            LOGGER.info('[OUTPUT]The cart was printed: %d', cart_id)



from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer thread that continuously supplies products.
    
    The producer registers with the marketplace once and then enters an infinite
    loop to publish its assigned products.
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
        self.producer_id = self.marketplace.register_producer()
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the producer thread."""

        # Invariant: This thread will run forever, simulating continuous production.
        while True:
            for product, quantity, prod_time in self.products:
                for _ in range(quantity):
                    result = self.marketplace.publish(self.producer_id, product)

                    # Block Logic: Handle successful and failed publications.
                    if result is True:
                        # On success, wait for the specified production time.
                        time.sleep(prod_time)
                    else:
                        # On failure (queue full), wait and retry until successful.
                        while not self.marketplace.publish(self.producer_id, product):
                            time.sleep(self.republish_wait_time)


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
