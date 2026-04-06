"""
This module simulates a multi-threaded producer-consumer model for an e-commerce
marketplace.

It defines `Consumer`, `Producer`, and `Marketplace` classes. This implementation
has several notable design choices and potential logic flaws, particularly in how
it manages product availability and state between consumers and producers.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer thread that processes a list of shopping commands.

    Note: This implementation creates a single cart and applies all operations
    from all its assigned "carts" to this one cart, rather than creating a new
    cart for each.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "carts", where each cart is a list of
                          operations (add/remove).
            marketplace (Marketplace): A reference to the central marketplace object.
            retry_wait_time (int): Time to wait before retrying an operation.
            **kwargs: Keyword arguments for the Thread parent class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_to_cart(self, quantity, cart_id, product):
        """
        Helper method to add items to the cart, retrying if unavailable.
        """
        i = 0
        while i < quantity:
            added_ok = self.marketplace.add_to_cart(cart_id, product)
            if added_ok:
                i = i + 1
            else:
                time.sleep(self.retry_wait_time)

    def remove_from_cart(self, quantity, cart_id, product):
        """Helper method to remove items from the cart."""
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """The main execution logic for the consumer thread."""
        
        # A single cart is created for all operations this consumer will perform.
        cart_id = self.marketplace.new_cart()
        for cart_list in self.carts:
            for cart_event in cart_list:
                if cart_event["type"] == "add":
                    self.add_to_cart(cart_event["quantity"], cart_id, cart_event["product"])
                else:
                    self.remove_from_cart(cart_event["quantity"], cart_id, cart_event["product"])
        
        # After all operations, place the order for the single cart.
        for product in self.marketplace.place_order(cart_id):
            print(self.name, "bought", product)


from threading import Lock
import unittest
import time
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

class Marketplace:
    """
    Coordinates producers and consumers in a simulated e-commerce environment.

    Warning: This implementation has several design flaws that can lead to
    race conditions and incorrect behavior. For example, it uses a global
    `product_in_cart` flag that acts as a lock on a product type, meaning only
    one of any given product can be in any cart across the entire system at once.
    The `place_order` logic is also not synchronized with inventory removal.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        
        Args:
            queue_size_per_producer (int): The max number of products a single
                                           producer can have published.
        """
        self.queue_size_per_peroducer = queue_size_per_producer
        self.products = []  # List of lists, where index = producer_id
        self.carts = []     # List of lists, where index = cart_id
        
        # Flawed global lock: this dict attempts to track if a product is in *any* cart.
        self.product_in_cart = {}
        
        self.lock_cart = Lock()
        self.lock_producer = Lock()
        self.logger = logging.getLogger('marketplace')
        handler = RotatingFileHandler('marketplace.log', maxBytes=4096, backupCount=10)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        logging.Formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


        self.logger.setLevel("INFO")

    def register_producer(self):
        """Atomically registers a new producer by adding a new list to the products inventory."""
        self.logger.info("Method register_producer started")
        self.lock_producer.acquire()
        self.products.append([])
        ret = len(self.products) - 1
        self.lock_producer.release()
        self.logger.info("Method register_producer returned " + str(ret))
        return ret

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer if they have capacity.
        
        Note: This incorrectly marks the product type as unavailable globally.
        """
        self.logger.info("Method publish started")
        self.logger.info("producer_id = " + str(producer_id))
        self.logger.info("product = " + str(product))
        self.lock_producer.acquire()
        if len(self.products[producer_id]) < self.queue_size_per_peroducer:
            self.products[producer_id].append(product)
            # This is a major flaw: it prevents any other consumer from adding this
            # product type, regardless of stock.
            self.product_in_cart[product] = False
            self.lock_producer.release()
            self.logger.info("New product published to marketplace")
            return True

        self.lock_producer.release()
        self.logger.info("Method publish returned False")
        return False

    def new_cart(self):
        """Atomically creates a new cart by appending a list to the carts inventory."""
        self.logger.info("Method new_cart started")
        self.lock_cart.acquire()
        self.carts.append([])
        ret = len(self.carts) - 1
        self.lock_cart.release()
        self.logger.info("Method new_cart returned " + str(ret))
        return ret

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart if it's not in another cart.
        
        Note: This logic is flawed. It checks a global flag and does not actually
        remove the item from the producer's inventory upon adding to cart,
        creating a race condition.
        """
        self.logger.info("Method add_to_cart started")
        self.logger.info("cart_id = " + str(cart_id))
        self.logger.info("product = " + str(product))
        if product in self.product_in_cart.keys() and not self.product_in_cart[product]:
            self.carts[cart_id].append(product)
            # This globally locks the product type, preventing others from adding it.
            self.product_in_cart[product] = True
            self.logger.info("New product added to cart " + str(cart_id))
            return True

        self.logger.info("Method add_to_cart returned False")
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and makes it globally available again."""
        self.logger.info("Method remove_from_cart started")
        self.logger.info("cart_id = " + str(cart_id))
        self.logger.info("product = " + str(product))
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            # Makes the product type globally available again.
            self.product_in_cart[product] = False
            self.logger.info("Product removed from cart")

    def place_order(self, cart_id):
        """
        Finalizes an order.
        
        Note: This is not a transactional operation. It attempts to remove items
        from producer inventories after the fact, which is not thread-safe and
        relies on `add_to_cart` (which doesn't remove items) having worked correctly.
        """
        self.logger.info("Method place_order started")
        self.logger.info("cart_id = " + str(cart_id))
        for cart_product in self.carts[cart_id]:
            for prod_products in self.products:
                if cart_product in prod_products:
                    prod_products.remove(cart_product)
        self.logger.info("Method place_order returned " + str(self.carts[cart_id]))
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Prepares a new Marketplace instance and sample data before each test."""
        self.marketplace = Marketplace(15)
        self.products = [Coffee("Espresso", 7, 4.00, "MEDIUM"), 
                        Coffee("Irish", 10, 5.00, "MEDIUM"), 
                        Tea("Black", 10, "Green")]

    def test_register_producer(self):
        """Tests sequential producer ID generation."""
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        """Tests that a product can be published."""
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(0, self.products[0]))
        self.assertTrue(self.marketplace.publish(0, self.products[1]))

    def test_new_cart(self):
        """Tests sequential cart ID generation."""
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        """Tests adding a product to a cart."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()

        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[0]))
        self.assertEqual(len(self.marketplace.carts[0]), 1)
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[1]))
        self.assertEqual(len(self.marketplace.carts[0]), 2)

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()

        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.marketplace.add_to_cart(0, self.products[0])

        self.marketplace.add_to_cart(0, self.products[1])
        self.marketplace.remove_from_cart(0, self.products[2])
        self.assertEqual(len(self.marketplace.carts[0]), 2)
        self.marketplace.remove_from_cart(0, self.products[0])
        self.assertEqual(len(self.marketplace.carts[0]), 1)

    def test_place_order(self):
        """Tests that an order contains the correct products."""
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])

        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[1])
        self.assertEqual(self.marketplace.place_order(0), [self.products[0], self.products[1]])


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer thread that continuously supplies products.
    
    Warning: This implementation has a bug where it re-registers as a new
    producer on every iteration of its main loop, leading to an infinite
    number of producers being created.
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
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the producer thread."""
        
        # Invariant: This thread will run forever.
        while True:
            # BUG: A new producer ID is acquired on every loop iteration, creating
            # an unbounded number of producers instead of reusing one.
            producer_id = self.marketplace.register_producer()
            for product in self.products:
                i = 0
                num_of_products = product[1]
                curr_product = product[0]
                curr_product_wait_time = product[2]
                while i < num_of_products:
                    published_ok = self.marketplace.publish(producer_id, curr_product)
                    if published_ok:
                        i += 1
                        time.sleep(curr_product_wait_time)
                    else:
                        time.sleep(self.republish_wait_time)
