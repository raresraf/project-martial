"""
This module implements a producer-consumer simulation of a marketplace.

It defines the core components of the simulation:
- Marketplace: A thread-safe class that manages producers, products, and carts.
  It uses a coarse-grained locking strategy with multiple `threading.Lock`
  objects to control access to shared data structures.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places an order.
- TestMarketplace: A suite of unit tests to verify the marketplace functionality.
"""


from threading import Thread, Lock
from time import sleep
import unittest
import logging
from logging.handlers import RotatingFileHandler


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    The consumer processes a list of shopping carts, where each cart contains
    a series of operations (add/remove products). It simulates the behavior
    of a customer by persistently trying to acquire products.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of cart operations for the consumer to process.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): Time to wait before retrying to add a product.
            **kwargs: Keyword arguments for the Thread class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through its assigned carts, executes the add/remove operations
        for each product, and finally places the order.
        """
        # A consumer thread is assigned a single cart ID for its lifetime.
        curr_id = self.marketplace.new_cart()
        # The consumer processes a list of "shopping lists" or desired carts.
        for curr_cart in self.carts:
            for elem in curr_cart:
                action_type = elem["type"]
                prod_id = elem["product"]
                quantity = elem["quantity"]
                
                for i in range(quantity):
                    if action_type == "add":
                        # Invariant: Persistently try to add the product until successful.
                        while not self.marketplace.add_to_cart(curr_id, prod_id):
                            sleep(self.retry_wait_time)
                    if action_type == "remove":
                        self.marketplace.remove_from_cart(curr_id, prod_id)
                        sleep(self.retry_wait_time)
        
        # After all operations, place the order and print the results.
        order = self.marketplace.place_order(curr_id)
        for i in order:
            # Use a shared lock to prevent interleaved print statements.
            with self.marketplace.print_lock:
                print(f"cons{curr_id} bought {i}")

class Marketplace:
    """
    Manages the inventory and transactions in a thread-safe manner.

    This implementation uses several locks to protect different parts of the
    shared state. It maintains a global list of all available products, which
    is a potential performance bottleneck under high contention.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): Max products per producer.
        """
        # Set up logging for marketplace events.
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(asctime)s;%(message)s")
        self.rotating_file_handler = RotatingFileHandler('marketplace.log', 'w')
        self.rotating_file_handler.setLevel(logging.INFO)
        self.rotating_file_handler.setFormatter(self.formatter)
        self.log.addHandler(self.rotating_file_handler)

        self.queue_size_per_producer = queue_size_per_producer
        
        self.producers = {} # Maps producer_id to a list of their products.
        self.no_prod = 0    # Counter for generating producer IDs.
        
        self.carts = {}     # Maps cart_id to a list of products in the cart.
        self.no_carts = 0   # Counter for generating cart IDs.

        # A global list of all products currently for sale in the marketplace.
        self.market_products = []

        # Architectural Pattern: A set of locks for different operations.
        # This is a coarse-grained locking strategy.
        self.product_lock = Lock()  # For registering new producers.
        self.cart_lock = Lock()     # For creating new carts.
        self.publish_lock = Lock()  # For publishing products.
        self.add_lock = Lock()      # For adding items to a cart.
        self.print_lock = Lock()    # For synchronized console output.

    def register_producer(self):
        """Registers a new producer, returning a unique integer ID."""
        with self.product_lock:
            self.log.info("begin register method")
            self.no_prod += 1
            id_p = self.no_prod
        
        self.producers[id_p] = []
        self.log.info("end register method")
        return id_p

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product.

        Returns:
            bool: True if successful, False if the producer's queue is full.
        """
        with self.publish_lock:
            self.log.info("begin publish method")
            # Pre-condition: Check if the producer has space in their queue.
            if len(self.producers[int(producer_id)]) >= self.queue_size_per_producer:
                self.log.info("end publish method with False")
                return False

            self.producers[int(producer_id)].append(product)
            self.market_products.append(product)
            self.log.info("end publish method with True")
            return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its unique ID."""
        with self.cart_lock:
            self.log.info("begin new_cart method")
            self.no_carts += 1
            new_cart_id = self.no_carts
        
        self.carts[new_cart_id] = []
        self.log.info("end new_cart method")
        return new_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart if it's available in the marketplace.

        This method has a critical section that finds a product in the global
        list, removes it, adds it to the cart, and removes it from the
        original producer's list.
        """
        with self.add_lock:
            self.log.info("begin add_to_cart method")
            if product in self.market_products:
                self.carts[cart_id].append(product)
                self.market_products.remove(product)
                
                # Find which producer had the product and remove it from their list.
                for id_p in self.producers:
                    if product in self.producers[id_p]:
                        self.producers[id_p].remove(product)
                        break

                self.log.info("end add_to_cart method with True")
                return True
        self.log.info("end add_to_cart method with False")
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace."""
        self.log.info("begin remove_from_cart method")
        # Note: This method is not thread-safe. Concurrent removes could cause issues.
        for prod in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove(prod)
                self.market_products.append(prod)
                # The product is not returned to the original producer's queue,
                # which may be a bug or a design choice.
                break
        self.log.info("end remove_from_cart method")

    def place_order(self, cart_id):
        """Finalizes the purchase, returning the list of products in the cart."""
        self.log.info("begin place_order method")
        order = self.carts[cart_id]
        self.log.info("end place_order method")
        return order


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""

    def setUp(self):
        """Initializes a marketplace with a queue size of 2 for each test."""
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        """Tests that producer IDs are generated sequentially."""
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        """Tests that publishing a product adds it to the market."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])

    def test_new_cart(self):
        """Tests the creation of a new cart."""
        self.marketplace.new_cart()
        self.assertIn(1, self.marketplace.carts)

    def test_add_to_cart(self):
        """Tests adding an available product to a cart."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')"))
        self.assertNotIn("Tea(name='Linden', price=9, type='Herbal')", self.marketplace.market_products)

    def test_remove_from_cart(self):
        """Tests that removing a product from a cart returns it to the market."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.marketplace.remove_from_cart(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.assertIn("Tea(name='Linden', price=9, type='Herbal')", self.marketplace.market_products)

    def test_place_order(self):
        """Tests that placing an order returns the correct items."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')")
        order = self.marketplace.place_order(1)
        self.assertEqual(order, ["Tea(name='Linden', price=9, type='Herbal')"])


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of (product, quantity, delay) tuples.
            marketplace (Marketplace): The marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """The main execution logic for the producer thread."""
        producer_id = self.marketplace.register_producer()
        
        # Invariant: The producer runs in an infinite loop to continuously supply products.
        while True:
            for prod in self.products:
                product_id, quantity, wait_time = prod
                
                i = 0
                # Invariant: Publish the specified quantity of the current product.
                while i < int(quantity):
                    if self.marketplace.publish(str(producer_id), product_id):
                        i += 1
                        sleep(wait_time)
                
                # Wait before cycling to the next product.
                sleep(self.republish_wait_time)
