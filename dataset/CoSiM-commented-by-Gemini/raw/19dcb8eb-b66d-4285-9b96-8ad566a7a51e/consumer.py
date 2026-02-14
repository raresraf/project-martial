"""
This file simulates a Producer-Consumer model for an e-commerce marketplace.

It contains several modules concatenated together:
- Product classes (`Product`, `Tea`, `Coffee`): Dataclasses for items.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds items to a cart and "buys" them.
- Marketplace: The central shared resource, coordinating producers and consumers.
- TestMarketplace: A unittest class for the marketplace.

NOTE: This implementation contains several race conditions and logical bugs.
"""
import time
from threading import Thread, Lock
import unittest
from dataclasses import dataclass


# === Product Module ===

@dataclass(frozen=True)
class Product:
    """A base dataclass for a product with a name and price."""
    name: str
    price: int

@dataclass(frozen=True)
class Tea(Product):
    """A Tea product, inheriting from Product and adding a 'type'."""
    type: str

@dataclass(frozen=True)
class Coffee(Product):
    """A Coffee product, inheriting from Product with acidity and roast level."""
    acidity: str
    roast_level: str


# === Producer Module ===

class Producer(Thread):
    """
    A worker thread that produces items and publishes them to the marketplace.

    BUG: This class continuously re-registers itself as a new producer on every
    single publish attempt, which is incorrect. It should register only once.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """Continuously tries to publish products to the marketplace."""
        while True:
            for prod_info in self.products:
                product, quantity, wait_time = prod_info
                product_counter = 0
                while product_counter < quantity:
                    # BUG: This registers a new producer on every attempt.
                    producer_id = self.marketplace.register_producer()
                    if self.marketplace.publish(producer_id, product):
                        product_counter += 1
                        time.sleep(wait_time)
                    else:
                        # Marketplace queue is full, wait before retrying.
                        time.sleep(self.republish_wait_time)


# === Consumer Module ===

class Consumer(Thread):
    """
    A worker thread that simulates a consumer's shopping activities.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer.

        Args:
            carts (list): A list of "carts", where each cart is a list of
                          shopping activities (dictionaries).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying to add a
                                     product if it's out of stock.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # This dictionary is not used effectively across the class.
        self.activities = {}

    def get_info(self, consumer_id, cart_activities):
        """
        Processes a list of shopping activities for a given cart.

        Args:
            consumer_id (int): The ID of the cart to operate on.
            cart_activities (list): A list of activity dictionaries.
        """
        for activity in cart_activities:
            activity_type = activity.get("type")
            product = activity.get("product")
            quantity = activity.get("quantity")

            if activity_type == "add":
                # This is a busy-wait loop. It will continuously try to add
                # the product until it succeeds.
                product_counter = 0
                while product_counter < quantity:
                    if self.marketplace.add_to_cart(consumer_id, product):
                        product_counter += 1
                    else:
                        time.sleep(self.retry_wait_time)
            elif activity_type == "remove":
                product_counter = 0
                while product_counter < quantity:
                    self.marketplace.remove_from_cart(consumer_id, product)
                    product_counter += 1

    def run(self):
        """
        The main execution loop for the consumer. Processes each cart assigned.
        """
        for cart_activities in self.carts:
            consumer_id = self.marketplace.new_cart()
            self.get_info(consumer_id, cart_activities)
            
            bought_products = self.marketplace.place_order(consumer_id)
            for product in bought_products:
                print(f"{self.name} bought {product}")


# === Marketplace Module ===

class Marketplace:
    """
    The central marketplace, a shared resource for producers and consumers.

    NOTE: This class is not fully thread-safe. Several methods lack the
    necessary locking to prevent race conditions.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.carts_counter = -1
        self.producers_counter = -1
        self.carts = []
        self.products = [] # Tracks number of products per producer.
        self.in_stock_products = []
        self.in_stock_products_producers = [] # Tracks (producer_id, product).

        self.carts_lock = Lock()
        self.producers_lock = Lock()
        self.in_stock_lock = Lock()

    def register_producer(self):
        """Thread-safely registers a new producer ID."""
        with self.producers_lock:
            self.producers_counter += 1
            self.products.append(0)
            return self.producers_counter

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product.

        RACE CONDITION: This method is not thread-safe. Two producers could
        read `self.products[producer_id]` simultaneously, and both might
        incorrectly believe they can publish, leading to exceeding the queue size.
        """
        if self.products[producer_id] < self.queue_size_per_producer:
            self.in_stock_products.append(product)
            self.products[producer_id] += 1
            self.in_stock_products_producers.append((producer_id, product))
            return True
        return False

    def new_cart(self):
        """Thread-safely creates a new empty cart and returns its ID."""
        with self.carts_lock:
            self.carts_counter += 1
            self.carts.append([])
            return self.carts_counter

    def add_to_cart(self, cart_id, product):
        """Thread-safely moves a product from stock to a consumer's cart."""
        with self.in_stock_lock:
            if product in self.in_stock_products:
                self.carts[cart_id].append(product)
                self.in_stock_products.remove(product)
                return True
            return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to stock.
        
        RACE CONDITION: This method is not thread-safe. Access to
        `self.carts[cart_id]` and `self.in_stock_products` is not protected by a lock.
        """
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            self.in_stock_products.append(product)

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        LOGIC ERROR & RACE CONDITION: This method is not thread-safe and has
        incorrect logic. The line `element = self.in_stock_products_producers[prod]`
        will raise a TypeError because `prod` is a Product object, not an integer index.
        The surrounding loop and lock are also insufficient.
        """
        # This loop is logically flawed and will not work as intended.
        for prod in self.carts[cart_id]:
            # This check is inefficient and the subsequent indexing is incorrect.
            if prod in self.in_stock_products_producers:
                with self.producers_lock:
                    # This line will cause a TypeError.
                    element = self.in_stock_products_producers[prod]
                    self.products[element] -= 1
        return self.carts[cart_id]


# === Test Module ===

class TestMarketplace(unittest.TestCase):
    """A basic test suite for the marketplace."""
    def setUp(self):
        """Sets up the test environment."""
        self.marketplace = Marketplace(24)
        self.tea1 = Tea("Mint", 15, "Green")
        self.tea2 = Tea("Earl grey", 30, "Black")
        self.coffee = Coffee("Lavazza", 14, "2.23", "MEDIUM")
        self.producer = Producer([[self.tea1, 8, 0.11],
                                  [self.tea2, 5, 0.7],
                                  [self.coffee, 1, 0.13]],
                                 self.marketplace,
                                 0.35)
        self.consumer = Consumer([[{"type": "add", "product": self.coffee, "quantity": 1},
                                   {"type": "add", "product": self.tea1, "quantity": 4},
                                   {"type": "add", "product": self.tea2, "quantity": 2},
                                   {"type": "remove", "product": self.tea2, "quantity": 1}
                                   ]],
                                 self.marketplace,
                                 0.31)
        self.cart_id = self.marketplace.new_cart()

    def test_register_function(self):
        """Tests the producer registration."""
        # Due to the bug in Producer, this test is not fully representative.
        self.assertEqual(str(self.marketplace.register_producer()), "0")
