"""
This module defines classes for a simulated marketplace system, including:
- Consumer: Represents a buyer interacting with the marketplace.
- Producer: Represents a seller supplying products to the marketplace.
- Marketplace: Manages product inventory, carts, and order processing, ensuring thread-safe operations.
- Product, Tea, Coffee: Data classes for defining product types and their attributes.

The module also includes a set of unit tests for the Marketplace class to ensure its core functionalities work as expected.
"""


from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace. Inherits from `threading.Thread`.
    Consumers interact with the marketplace to add and remove products from their cart,
    and finally place an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of products the consumer intends to buy.
            marketplace (Marketplace): The marketplace instance the consumer interacts with.
            retry_wait_time (float): Time in seconds to wait before retrying an action (e.g., adding to cart).
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method,
                      including 'name' for the consumer's identifier.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)
        self.name = kwargs["name"]

    def run(self):
        """
        Executes the consumer's shopping logic.
        - Creates a new cart in the marketplace.
        - Iterates through the list of desired items, adding or removing them from the cart.
        - Places the final order and prints the purchased products.
        """
        id_cart = self.marketplace.new_cart()

        # Block Logic: Process each cart provided to the consumer.
        # Invariant: Each 'cart' element represents a list of purchase commands.
        for cart in self.carts:
            # Block Logic: Process each command within the current cart.
            # Pre-condition: 'com' is a dictionary containing 'type', 'quantity', and 'product'.
            for com in cart:
                type_com = com['type']
                quantity = com['quantity']
                product = com['product']
                if type_com == "remove":
                    # Block Logic: Remove specified quantity of a product from the cart.
                    for i in range(quantity):
                        self.marketplace.remove_from_cart(id_cart, product)
                elif type_com == "add":
                    i = 0
                    # Block Logic: Add specified quantity of a product to the cart, retrying if necessary.
                    # Invariant: The loop continues until the desired quantity is added.
                    while 1:
                        if i >= quantity:
                            break
                        # Block Logic: Attempt to add the product to the cart.
                        # Invariant: If add_to_cart fails, the consumer waits and retries.
                        while not self.marketplace.add_to_cart(id_cart, product):
                            sleep(self.retry_wait_time)
                        i += 1
        # Block Logic: Place the final order for all items in the cart.
        for prod in self.marketplace.place_order(id_cart):
            to_print = "{} bought {}".format(self.name, prod)
            print(to_print)

from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import time
import unittest

class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    Ensures that core functionalities like producer registration, product publishing,
    cart creation, and product management work as expected.
    """
    
    def test_register_producer(self):
        """
        Tests the `register_producer` method.
        Verifies that a new producer is successfully registered and returns an ID of 0.
        """
        market = Marketplace(10)
        self.assertEquals(market.register_producer(), 0, "Expected to return id 0")

    def test_publish(self):
        """
        Tests the `publish` method.
        Verifies that a product can be successfully published by a registered producer.
        """
        market = Marketplace(10)
        producter_id = market.register_producer()
        self.assertTrue(market.publish(producter_id, "Tea"), "Expected to return True")

    def test_new_cart(self):
        """
        Tests the `new_cart` method.
        Verifies that a new cart is successfully created and returns an ID of 0.
        """
        market = Marketplace(10)
        self.assertEquals(market.new_cart(), 0, "Expected to return id 0")

    def test_remove_from_cart(self):
        """
        Tests the `remove_from_cart` method.
        Verifies that a product can be successfully removed from a cart.
        """
        market = Marketplace(10)
        cart_id = market.new_cart()
        market.add_to_cart(cart_id, "Tea")
        self.assertTrue(market.remove_from_cart(cart_id, "Tea"), "Expected to return True")

    def test_place_order(self):
        """
        Tests the `place_order` method.
        Verifies that placing an order returns the correct product.
        """
        market = Marketplace(10)
        cart_id = market.new_cart()
        market.add_to_cart(cart_id, "Tea")
        self.assertEquals(market.place_order(cart_id)[0], "Tea", "Expected to return Tea")


class Marketplace:
    """
    Manages the overall marketplace operations, including producers, consumers,
    product queues, and logging. Ensures thread-safe access to shared resources.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with a specified queue size per producer.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have in its queue.
        """
        
        self.queue_size_per_producer = queue_size_per_producer
        self.mutex = Lock()  # Protects shared resources (producers, carts, counters)
        self.producers = []  # List of producer queues, each a list of products
        self.carts = []      # List of consumer carts, each a list of [product, producer_id]
        self.producers_no = -1 # Counter for assigning unique producer IDs
        self.consumers_no = -1 # Counter for assigning unique consumer (cart) IDs

        # Block Logic: Configure logging for the marketplace.
        # Invariant: Log messages will be formatted and rotated.
        logging.Formatter.converter = time.gmtime
        log_formatter = logging.Formatter("%(asctime)s:%(levelname)s: \\
                                         %(filename)s::%(funcName)s:%(lineno)d %(message)s")
        logger = logging.getLogger()
        logger.propagate = False
        handler_file = RotatingFileHandler('marketplace.log', maxBytes=4096, backupCount=1)

        handler_file.setFormatter(log_formatter)
        logger.addHandler(handler_file)
        logger.setLevel(logging.INFO)

        self.logger = logger

    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns a unique ID.
        Ensures thread-safe registration using a mutex.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        
        self.mutex.acquire()
        self.producers_no += 1
        self.producers.append([])
        producer_id = self.producers_no
        self.mutex.release()
        self.logger.info("New producer: {}".format(producer_id))

        return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace from a specific producer.
        The product is added to the producer's queue if there is space.
        Ensures thread-safe access to the producer's queue.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The name of the product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        publish_state = False

        self.mutex.acquire()
        # Block Logic: Check if the producer's queue has space.
        # Pre-condition: `producer_id` refers to a valid, registered producer.
        if len(self.producers[producer_id]) < self.queue_size_per_producer:
            self.producers[producer_id].append(product)
            publish_state = True
        self.mutex.release()

        self.logger.info("state: {} \\
            add product {} of producer: {}".format(publish_state, product, producer_id))

        return publish_state

    def new_cart(self):
        """
        Creates a new shopping cart for a consumer and assigns a unique ID.
        Ensures thread-safe cart creation using a mutex.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        self.mutex.acquire()
        cart = []
        self.carts.append(cart)
        self.consumers_no += 1
        cart_id = self.consumers_no
        self.mutex.release()
        self.logger.info("New char id: {}".format(cart_id))

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific cart.
        It searches through all producer queues for the product, removes it from the first found,
        and adds it to the consumer's cart. Ensures thread-safe operation.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (str): The name of the product to add.

        Returns:
            bool: True if the product was successfully added to the cart, False otherwise.
        """
        self.mutex.acquire()
        i = 0
        # Block Logic: Iterate through each producer's product list.
        # Invariant: `prod` represents a single producer's product queue.
        for prod in self.producers:
            # Block Logic: Iterate through products in the current producer's queue.
            # Pre-condition: `produs` is a product offered by the current producer.
            for produs in prod:
                if product == produs:
                    # Inline: Store the product and its original producer's ID for later removal or order.
                    self.carts[cart_id].append([product, i])
                    prod.remove(product)
                    self.mutex.release()
                    self.logger.info("add product {} to cart id: {}".format(product, cart_id))
                    return True
            i += 1
        self.mutex.release()
        self.logger.error("add product {} to cart id: {}".format(product, cart_id))

        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specific cart and returns it to its original producer's queue.
        Ensures thread-safe operation.

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (str): The name of the product to remove.

        Returns:
            bool: Always True, indicating the operation was attempted.
        """
        self.mutex.acquire()

        # Block Logic: Iterate through items in the specified cart to find the product.
        # Invariant: `produs` is a [product_name, producer_id] pair.
        for produs in self.carts[cart_id]:
            if produs[0] == product:
                self.producers[produs[1]].append(produs[0]) # Return to producer's queue
                self.carts[cart_id].remove(produs)
                break
        self.mutex.release()
        self.logger.info("remove product {} to cart id: {}".format(product, cart_id))
        return True

    def place_order(self, cart_id):
        """
        Places an order for all items currently in the specified cart.
        The items are removed from the cart and returned as a list.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: A list of products that were part of the placed order.
        """
        
        products = [x[0] for x in self.carts[cart_id]]
        self.logger.info("place order {} to cart id: {}".format(products, cart_id))

        return products


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace. Inherits from `threading.Thread`.
    Producers register with the marketplace and continuously publish products.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products the producer will publish.
                             Each product is a tuple: (product_id, quantity, wait_time).
            marketplace (Marketplace): The marketplace instance the producer interacts with.
            republish_wait_time (float): Time in seconds to wait before retrying a publish operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method.
        """
        
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        Executes the producer's logic.
        - Registers itself with the marketplace.
        - Continuously publishes products based on its `products` list,
          with specified quantities and wait times.
        """
        producer_id = self.marketplace.register_producer()

        # Block Logic: Main loop for continuous product publishing.
        # Invariant: The producer attempts to publish products indefinitely.
        while 1:
            # Block Logic: Iterate through the producer's defined products.
            # Pre-condition: `elem` is a tuple (id_prod, cantitate, timp_asteptare).
            for elem in self.products:
                (id_prod, cantitate, timp_asteptare) = elem
                sleep_time = cantitate * timp_asteptare
                sleep(sleep_time)
                i = 0
                # Block Logic: Publish the specified quantity of the current product.
                # Invariant: The loop continues until `cantitate` products are published.
                while 1:
                    if i >= cantitate:
                        break
                    # Block Logic: Attempt to publish the product, retrying if the marketplace is full.
                    # Invariant: If publish fails, the producer waits and retries.
                    while not self.marketplace.publish(producer_id, id_prod):
                        sleep(self.republish_wait_time)
                    i += 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base data class for a product.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Data class for a Tea product, inheriting from Product.

    Attributes:
        type (str): The type of tea (e.g., Green, Black, Herbal).
    """
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Data class for a Coffee product, inheriting from Product.

    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    
    acidity: str
    roast_level: str
