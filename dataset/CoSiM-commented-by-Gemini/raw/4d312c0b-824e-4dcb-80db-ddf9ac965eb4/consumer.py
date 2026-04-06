"""
This module simulates a multi-threaded producer-consumer model for an e-commerce
marketplace.

It defines the `Consumer`, `Marketplace`, and `Producer` classes that interact
in a concurrent environment. The Marketplace uses coarse-grained locks to manage
access to shared data structures representing inventories and shopping carts.
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
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the consumer thread."""
        products_list = []
        cart_ids = []

        # First, create all necessary cart IDs for this consumer's session.
        for list_cart in self.carts:
            cart_id = self.marketplace.new_cart()
            cart_ids.append(cart_id)

        # Then, process all operations for each cart.
        for list_cart in self.carts:
            cart_id = cart_ids[self.carts.index(list_cart)]
            
            for dict_command in list_cart:
                command = dict_command['type']
                prod = dict_command['product']
                quantity = dict_command['quantity']

                counter_add = 0
                counter_remove = 0

                # Block Logic: Handle 'add' operations.
                if command == 'add':
                    # Pre-condition: Retry adding the product until successful.
                    # This simulates waiting for a product to be restocked.
                    while counter_add < quantity:
                        if not self.marketplace.add_to_cart(cart_id, prod):
                            time.sleep(self.retry_wait_time)
                        else:
                            counter_add += 1
                # Block Logic: Handle 'remove' operations.
                else:
                    while counter_remove < quantity:
                        self.marketplace.remove_from_cart(cart_id, prod)
                        counter_remove += 1

        # After all operations are done, place all orders.
        for cart in cart_ids:
            products_list.extend(self.marketplace.place_order(cart))

        # Use a lock to ensure that the output from different consumers is not interleaved.
        self.marketplace.print_lock.acquire()
        for product in products_list:
            print(f'{self.name} bought', product)
        self.marketplace.print_lock.release()

from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

# --- Global Logger Setup ---
# Configures a rotating file logger to record marketplace activities.
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('marketplace.log', maxBytes=3000, backupCount=5)
logger.addHandler(handler)

class Marketplace:
    """
    The central, thread-safe hub for coordinating producers and consumers.
    
    This implementation uses lists to manage inventories and carts, with indices
    acting as IDs. It employs two main locks, one for producer-related actions
    and one for consumer-related actions, to manage concurrent access.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        
        Args:
            queue_size_per_producer (int): The maximum number of products a
                                           producer can have published at once.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_list = []
        self.consumer_carts = []
        self.lock_prod = Lock()
        self.lock_cons = Lock()
        self.producer_id = 0
        self.cart_id = 0
        # This list tracks which producer a product came from when it's added to a cart.
        self.taken_products = []
        # A lock to synchronize print statements from multiple consumers.
        self.print_lock = Lock()

    def register_producer(self):
        """
        Atomically registers a new producer and returns a unique ID.
        
        The producer ID corresponds to an index in the `producer_list`.
        """
        self.lock_prod.acquire()
        logger.info("Producer wants to register")

        self.producer_id += 1
        self.producer_list.insert(self.producer_id - 1, [])

        logger.info("Producer %d got registered", self.producer_id)
        self.lock_prod.release()

        return self.producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product if they have inventory capacity.
        
        Returns:
            bool: True if successful, False if the producer's queue is full.
        """
        logger.info("Producer %d wants to publish %s", producer_id, product)
        
        # Pre-condition: Check if the producer is below their publication limit.
        if len(self.producer_list[producer_id - 1]) == self.queue_size_per_producer:
            return False

        self.producer_list[producer_id - 1].append(product)
        logger.info("Producer %d published %s", producer_id, product)
        return True

    def new_cart(self):
        """Atomically creates a new empty cart and returns its ID."""
        self.lock_cons.acquire()
        logger.info("Consumer requested a cart")

        self.cart_id += 1
        self.consumer_carts.insert(self.cart_id - 1, [])

        logger.info("Consumer got cart %d", self.cart_id)
        self.lock_cons.release()

        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Searches all producer inventories for a product and adds it to the cart.
        
        This implementation is inefficient as it iterates through all producers
        for every add operation.
        
        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        self.lock_cons.acquire()
        logger.info("Cart %d wants to add %s", cart_id, product)

        copy_lst = list(self.producer_list)

        # Invariant: Search all producers for the requested product.
        for lst in copy_lst:
            if product in lst:
                indx = copy_lst.index(lst)
                # Add to consumer's cart.
                self.consumer_carts[cart_id - 1].append(product)
                # Track which producer this item came from for later removal.
                self.taken_products.append((indx + 1, product))
                # Remove from producer's inventory.
                self.producer_list[indx].remove(product)
                self.lock_cons.release()
                logger.info("Cart %d added %s", cart_id, product)
                return True

        self.lock_cons.release()
        logger.info("Cart %d added %s", cart_id, product)
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the original producer."""
        logger.info("Cart %d wants to remove %s", cart_id, product)
        
        self.consumer_carts[cart_id - 1].remove(product)

        # Find the product in the 'taken' list to identify its original producer.
        for prod in self.taken_products:
            if prod[1] == product:
                # Return the product to the correct producer's inventory list.
                self.producer_list[prod[0] - 1].append(product)
                break
        logger.info("Cart %d removed %s", cart_id, product)

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the list of products in the cart.
        
        Note: This implementation has a potential design flaw. It does not
        replenish the producer's inventory slots, meaning producers will
        eventually be unable to publish new products even after their items are sold.
        """
        logger.info("Cart %d placed order", cart_id)
        return self.consumer_carts[cart_id - 1]

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Prepares a new Marketplace instance for each test."""
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        """Tests that producer registration returns a non-zero ID."""
        self.assertNotEqual(self.marketplace.register_producer(), 0, "Wrong id for producer!")

    def test_publish(self):
        """Tests that a product can be successfully published."""
        actual_product = self.marketplace.publish(self.marketplace.register_producer(), 
                                                    Tea("Linden", 13, "Floral"))
        self.assertTrue(actual_product, "The product should be published!")

    def test_new_cart(self):
        """Tests that cart creation returns a non-zero ID."""
        self.assertNotEqual(self.marketplace.new_cart(), 0, "Wrong id for cart!")

    def test_add_to_cart(self):
        """Tests that an unavailable product cannot be added to the cart."""
        actual_cart_id = self.marketplace.new_cart()
        wanted_product = self.marketplace.add_to_cart(actual_cart_id,
                                                    Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.assertFalse(wanted_product, "This product should not be added to cart now!")

    def test_remove_from_cart(self):
        """Tests that a product is correctly removed from a cart."""
        actual_cart_id = self.marketplace.new_cart()
        self.marketplace.publish(self.marketplace.register_producer(),
                                    Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.add_to_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.remove_from_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.assertListEqual(self.marketplace.consumer_carts[actual_cart_id - 1], [],
                                        "Product was not removed!")

    def test_place_order(self):
        """Tests that placing an order returns the correct list of products."""
        prod_id = self.marketplace.register_producer()
        self.marketplace.publish(prod_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.publish(prod_id, Tea("Brewstar", 17, "Green"))
        actual_cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.add_to_cart(actual_cart_id, Tea("Brewstar", 17, "Green"))
        self.marketplace.remove_from_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        remainder_list = []
        remainder_list.append(Tea("Brewstar", 17, "Green"))
        self.assertCountEqual(self.marketplace.place_order(actual_cart_id),
                                    remainder_list, "Wrong order!")


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer thread that continuously supplies products to the marketplace.
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
        self.time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the producer thread."""
        producer_id = self.marketplace.register_producer()

        # Invariant: This thread will run forever, simulating continuous production.
        while True:
            for prod in self.products:
                i = 0
                # Block Logic: Publish the specified quantity of a product.
                while i < prod[1]:
                    # If publishing fails (queue full), wait and retry.
                    if not self.marketplace.publish(producer_id, prod[0]):
                        time.sleep(self.time)
                    else:
                        i += 1
                        # On success, wait for the specified production time.
                        time.sleep(prod[2])
