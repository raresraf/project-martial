"""
This module simulates a marketplace with producers and consumers using a multi-threaded approach.

It defines three main classes:
- Producer: A thread that generates and publishes products to the marketplace.
- Consumer: A thread that browses products, adds/removes them from a personal cart, and
  eventually places an order.
- Marketplace: The central class that synchronizes producers and consumers, manages product
  inventory, and handles cart operations and order placements in a thread-safe manner.

The module also includes unit tests for the Marketplace's functionality.
"""

from threading import Thread, Lock
from time import sleep
from Queue import Queue
import unittest
import logging
from logging.handlers import RotatingFileHandler


class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread, performing a series of operations
    defined in its cart, such as adding or removing products, and then
    placing a final order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of actions (add/remove) for the consumer to perform.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying an operation.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # A new cart is created in the marketplace for this consumer instance.
        self.name = self.marketplace.new_cart()

    def run(self):
        """
        The main execution loop for the consumer.

        Processes each operation in the cart, retrying if a product is not
        immediately available. After processing the cart, it places the order
        and prints the purchased products.
        """
        # Invariant: The loop processes each product operation defined in the cart.
        for cart in self.carts:
            for prod in cart:
                type_op = prod["type"]
                prod_id = prod["product"]
                quantity = prod["quantity"]

                # Pre-condition: 'add' operation tries to acquire a product.
                if type_op == "add":
                    for i in range(quantity):
                        # The loop continues until the product is successfully added to the cart.
                        while not self.marketplace.add_to_cart(self.name, str(prod_id)):
                            sleep(self.retry_wait_time)

                # Pre-condition: 'remove' operation releases a product from the cart.
                if type_op == "remove":
                    for i in range(quantity):
                        self.marketplace.remove_from_cart(self.name, prod_id)
                        sleep(self.retry_wait_time)

        # After all cart operations, the final order is placed.
        list_of_products = self.marketplace.place_order(int(self.name))

        # The printer lock ensures that output from different consumers is not interleaved.
        with self.marketplace.lock_printer:
            for product in list_of_products:
                print(f"cons{self.name} bought {product}")


class Marketplace:
    """
    Manages the inventory and transactions between producers and consumers.

    This class provides a thread-safe environment for producers to publish
    products and for consumers to manage their carts and place orders. It uses
    locks to protect shared data structures from race conditions.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        # Set up logging for marketplace activities.
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.format = logging.Formatter("%(asctime)s %(message)s")
        self.rotating_handler = RotatingFileHandler('marketpplace.log', 'w')
        self.rotating_handler.setFormatter(self.format)
        self.log.addHandler(self.rotating_handler)

        self.queue_size_per_producer = queue_size_per_producer
        # `dict_prod`: Stores products published by each producer.
        self.dict_prod = {}
        # `dict_con`: Stores the contents of each consumer's cart.
        self.dict_con = {}
        # Locks for ensuring thread-safe operations.
        self.lock_prod = Lock()
        self.lock_register = Lock()
        self.lock_con = Lock()
        self.lock_publish = Lock()
        self.lock_printer = Lock()
        # Counters for generating unique producer and cart IDs.
        self.generateId = 0
        self.cartId = 0
        # `products_list`: A master list of all available products in the marketplace.
        self.products_list = []

    def register_producer(self):
        """
        Registers a new producer, assigning a unique ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.lock_register:
            self.generateId += 1
        
        # Initializes an empty list for the new producer's products.
        self.dict_prod[self.generateId] = []
        return self.generateId

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a new product to the marketplace.

        The operation is limited by `queue_size_per_producer`.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise.
        """
        with self.lock_publish:
            if self.queue_size_per_producer > len(self.dict_prod[producer_id]):
                self.dict_prod[producer_id].append(product)
                self.products_list.append(product)
                return True
            else:
                return False

    def new_cart(self):
        """
        Creates a new cart for a consumer, assigning a unique ID.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.lock_con:
            self.cartId += 1

        # Initializes an empty list for the new cart's contents.
        self.dict_con[self.cartId] = []
        return self.cartId

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified consumer's cart.

        This involves removing the product from the general availability list
        and the producer's inventory, and adding it to the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The product to be added.

        Returns:
            bool: True if the product was successfully added, False if it was not found.
        """
        with self.lock_prod:
            if product in self.products_list:
                self.products_list.remove(product)
                # Finds the producer of the product to remove it from their stock.
                for key in self.dict_prod:
                    for value in self.dict_prod[key]:
                        if value == product:
                            self.dict_con[int(cart_id)].append(product)
                            self.dict_prod[key].remove(value)
                            return True
            return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart, making it available again.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (str): The product to be removed.
        """
        if str(product) in self.dict_con[int(cart_id)]:
            self.dict_con[int(cart_id)].remove(str(product))
            # Returns the product to the general availability list.
            self.products_list.append(str(product))

    def place_order(self, cart_id):
        """
        Finalizes the transaction for a given cart.

        Args:
            cart_id (int): The ID of the cart for which the order is placed.

        Returns:
            list: A list of products that were in the cart.
        """
        return self.dict_con[int(cart_id)]


class TestMarketPlace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Set up a new Marketplace instance for each test."""
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        """Test if producers are registered with sequential unique IDs."""
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        """Test if products can be published to the marketplace."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        self.assertEqual(self.marketplace.products_list,
                         ["Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')"])

    def test_new_cart(self):
        """Test if new carts are created with sequential unique IDs."""
        self.assertEqual(1, self.marketplace.new_cart())

    def test_add_to_cart(self):
        """Test if a product can be added to a cart."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        self.marketplace.add_to_cart(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        print(f'List of products in cart: {self.marketplace.dict_con}')

    def test_remove_from_cart(self):
        """Test if a product can be removed from a cart."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        self.marketplace.add_to_cart(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        print(f'List of products in cart: {self.marketplace.dict_con}')
        print(f'List of all available products: {self.marketplace.products_list}')
        self.marketplace.remove_from_cart(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        print(f'List of products in cart: {self.marketplace.dict_con}')
        print(f'List of all available products: {self.marketplace.products_list}')

    def test_place_order(self):
        """Test if an order can be placed correctly."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        self.marketplace.add_to_cart(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        print(f'List of products in cart: {self.marketplace.dict_con}')
        print(f'List of all available products: {self.marketplace.products_list}')
        print(f'Placed order: {self.marketplace.place_order(1)}')


class Producer(Thread):
    """
    Represents a producer that supplies products to the marketplace.

    Each producer runs in its own thread, publishing a predefined list of
    products to the marketplace at specified intervals.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products for the producer to publish.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Registers itself with the marketplace to get a unique ID.
        self.id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer.

        Continuously attempts to publish its products to the marketplace.
        """
        # Invariant: This loop runs indefinitely, trying to publish products.
        while True:
            for product in self.products:
                product_id = product[0]
                quantity = product[1]
                sleep_time = product[2]
                for i in range(quantity):
                    flag = self.marketplace.publish(self.id, str(product_id))
                    # If publishing is successful, wait for the defined `sleep_time`.
                    if flag:
                        sleep(sleep_time)
                    # If the producer's queue is full, wait before retrying.
                    sleep(self.republish_wait_time)
