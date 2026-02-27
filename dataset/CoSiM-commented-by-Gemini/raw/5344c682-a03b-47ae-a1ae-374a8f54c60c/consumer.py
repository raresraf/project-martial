from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer thread that performs a series of shopping operations
    in a simulated marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping sessions, where each session is a
                          list of command dictionaries.
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): Time in seconds to wait before retrying a
                                     failed 'add' operation.
            **kwargs: Keyword arguments for the parent Thread class, including 'name'.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution logic for the consumer thread.

        It creates a single cart and executes all assigned operations (add/remove)
        within that cart, then places the final order.
        """
        id_cart = self.marketplace.new_cart()

        for cart in self.carts:
            for com in cart:
                type_com = com['type']
                quantity = com['quantity']
                product = com['product']
                
                if type_com == "remove":
                    for i in range(quantity):
                        self.marketplace.remove_from_cart(id_cart, product)
                elif type_com == "add":
                    i = 0
                    while i < quantity:
                        # Attempt to add a product, sleeping and retrying on failure.
                        if self.marketplace.add_to_cart(id_cart, product):
                            i += 1
                        else:
                            sleep(self.retry_wait_time)
        
        # After all operations, place the order.
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
    A suite of unit tests for the Marketplace class.
    Note: Uses the deprecated `assertEquals` method.
    """
    def test_register_producer(self):
        """Tests that the first registered producer gets ID 0."""
        market = Marketplace(10)
        self.assertEquals(market.register_producer(), 0, "Expected to return id 0")

    def test_publish(self):
        """Tests that a product can be published successfully."""
        market = Marketplace(10)
        producter_id = market.register_producer()
        self.assertTrue(market.publish(producter_id, "Tea"), "Expected to return True")

    def test_new_cart(self):
        """Tests that the first new cart gets ID 0."""
        market = Marketplace(10)
        self.assertEquals(market.new_cart(), 0, "Expected to return id 0")

    def test_remove_from_cart(self):
        """Tests removing an item from a cart."""
        market = Marketplace(10)
        cart_id = market.new_cart()
        # This test is flawed as it doesn't account for a producer publishing the item first.
        market.add_to_cart(cart_id, "Tea")
        self.assertTrue(market.remove_from_cart(cart_id, "Tea"), "Expected to return True")

    def test_place_order(self):
        """Tests placing an order."""
        market = Marketplace(10)
        cart_id = market.new_cart()
        # This test is also flawed without a producer.
        market.add_to_cart(cart_id, "Tea")
        self.assertEquals(market.place_order(cart_id)[0], "Tea", "Expected to return Tea")


class Marketplace:
    """
    A thread-safe marketplace simulation using a single global mutex for all
    operations, which ensures safety but limits concurrency. It also includes
    logging of all major operations.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace, its data structures, and logging.

        Args:
            queue_size_per_producer (int): The max number of items any single
                                           producer can have in the market.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.mutex = Lock() # A single, coarse-grained lock for all operations.
        self.producers = [] # List of lists; producers[id] = [product1, product2,...]
        self.carts = []     # List of lists; carts[id] = [[product, producer_id],...]
        self.producers_no = -1 # Counter for producer IDs.
        self.consumers_no = -1 # Counter for cart IDs.

        # Setup for logging all marketplace activities to 'marketplace.log'.
        logging.Formatter.converter = time.gmtime
        log_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(filename)s::%(funcName)s:%(lineno)d %(message)s")
        logger = logging.getLogger()
        logger.propagate = False
        handler_file = RotatingFileHandler('marketplace.log', maxBytes=4096, backupCount=1)
        handler_file.setFormatter(log_formatter)
        logger.addHandler(handler_file)
        logger.setLevel(logging.INFO)
        self.logger = logger

    def register_producer(self) -> int:
        """Atomically registers a new producer and returns their ID."""
        with self.mutex:
            self.producers_no += 1
            self.producers.append([])
            producer_id = self.producers_no
        self.logger.info("New producer: {}".format(producer_id))
        return producer_id

    def publish(self, producer_id, product) -> bool:
        """Atomically publishes a product for a given producer."""
        with self.mutex:
            if len(self.producers[producer_id]) < self.queue_size_per_producer:
                self.producers[producer_id].append(product)
                publish_state = True
            else:
                publish_state = False
        self.logger.info("state: {} add product {} of producer: {}".format(publish_state, product, producer_id))
        return publish_state

    def new_cart(self) -> int:
        """Atomically creates a new empty cart and returns its ID."""
        with self.mutex:
            self.carts.append([])
            self.consumers_no += 1
            cart_id = self.consumers_no
        self.logger.info("New char id: {}".format(cart_id))
        return cart_id

    def add_to_cart(self, cart_id, product) -> bool:
        """
        Atomically adds a product to a cart.
        
        This involves an inefficient O(N*M) search through all producers (N) and
        all their products (M) to find the item.
        """
        with self.mutex:
            for i, prod_list in enumerate(self.producers):
                if product in prod_list:
                    # Store product and its original producer ID in the cart.
                    self.carts[cart_id].append([product, i])
                    prod_list.remove(product)
                    self.logger.info("add product {} to cart id: {}".format(product, cart_id))
                    return True
        self.logger.error("add product {} to cart id: {}".format(product, cart_id))
        return False

    def remove_from_cart(self, cart_id, product):
        """Atomically removes a product from a cart and returns it to the producer."""
        with self.mutex:
            for produs in self.carts[cart_id]:
                if produs[0] == product:
                    # Return the product to its original producer's list.
                    self.producers[produs[1]].append(produs[0])
                    self.carts[cart_id].remove(produs)
                    break
        self.logger.info("remove product {} to cart id: {}".format(product, cart_id))
        return True

    def place_order(self, cart_id) -> list:
        """Finalizes an order, returning the products."""
        # This operation is not protected by the mutex, which could be a race condition.
        products = [x[0] for x in self.carts[cart_id]]
        self.logger.info("place order {} to cart id: {}".format(products, cart_id))
        return products


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    A producer thread that continuously publishes a set of products.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer."""
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        Main execution loop. Registers the producer and then enters an infinite
        loop to publish products.
        """
        producer_id = self.marketplace.register_producer()
        while 1:
            for elem in self.products:
                (id_prod, cantitate, timp_asteptare) = elem
                # This sleep seems to simulate a batch production time.
                sleep(cantitate * timp_asteptare)
                i = 0
                while i < cantitate:
                    # Attempt to publish, sleeping on failure.
                    if self.marketplace.publish(producer_id, id_prod):
                        i += 1
                    else:
                        sleep(self.republish_wait_time)


from dataclasses import dataclass

# The following are simple data classes to represent product types.

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A 'Tea' product with an additional 'type' attribute."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A 'Coffee' product with acidity and roast level attributes."""
    acidity: str
    roast_level: str
