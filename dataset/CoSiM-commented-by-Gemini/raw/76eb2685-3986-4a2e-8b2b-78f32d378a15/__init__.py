"""
# @raw/76eb2685-3986-4a2e-8b2b-78f32d378a15/__init__.py
# @brief Implements a multithreaded producer-consumer simulation for a marketplace.
#
# This module contains the core logic for a marketplace where producer threads create
# and publish products, and consumer threads purchase them. The Marketplace class
# acts as a central hub, managing inventory, carts, and transactions with
# synchronization primitives to ensure thread safety.
"""
from threading import Thread
from time import sleep
from typing import List

from tema.marketplace import Marketplace


class Consumer(Thread):
    """
    Represents a consumer in the marketplace simulation. Each consumer runs in its
    own thread and processes a list of shopping carts.
    """

    def __init__(self, carts: List, marketplace: Marketplace, retry_wait_time: float, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (List): A list of carts, where each cart is a list of products to be
                          added or removed.
            marketplace (Marketplace): The central marketplace instance.
            retry_wait_time (float): Time to wait before retrying to add a product to the cart.
            **kwargs: Keyword arguments for the Thread constructor, including the consumer's name.
        """
        Thread.__init__(self, kwargs=kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self.logger = marketplace.logger

    def run(self):
        """
        The main execution loop for the consumer thread.
        Iterates through each cart, processes the items, and places the order.
        """
        log_msg = "Started consumer " + str(self.kwargs['name'])
        self.marketplace.log(log_msg, str(self.kwargs['name']))

        # Process each cart assigned to this consumer
        for cart in self.carts:
            # Create a new cart in the marketplace for each shopping session
            cart_id = self.marketplace.new_cart()
            self.marketplace.assign_owner(cart_id, str(self.kwargs['name']))
            cart_iter = iter(cart)
            log_msg = "NEW CART" + str(cart_id)
            self.marketplace.log(log_msg, str(self.kwargs['name']))

            # Get the first item from the cart
            req_item = next((item for item in cart_iter), None)

            # Process items in the cart until it's empty
            while req_item is not None:
                # If the action is 'add', try to add the product to the cart
                if req_item['type'] == 'add':
                    res = self.marketplace.add_to_cart(
                        cart_id, req_item['product'])
                    
                    # If the product was added successfully, decrement the quantity
                    if res:
                        req_item['quantity'] -= 1
                        
                        # If the desired quantity of the product has been reached, move to the next item
                        if req_item['quantity'] == 0:
                            req_item = next((item for item in cart_iter), None)
                    else:
                        # If adding the product fails (e.g., out of stock), wait before retrying
                        sleep(self.retry_wait_time)
                
                # If the action is 'remove', remove the product from the cart
                elif req_item['type'] == 'remove':
                    self.marketplace.remove_from_cart(
                        cart_id, req_item['product'])

                    req_item['quantity'] -= 1

                    if req_item['quantity'] == 0:
                        req_item = next((item for item in cart_iter), None)
            
            # Once all items are processed, place the order
            self.marketplace.place_order(cart_id)

        # When all carts are processed, sign out from the marketplace
        self.marketplace.sign_out(str(self.kwargs['name']))


import logging
from logging.handlers import RotatingFileHandler
from threading import Lock, Semaphore
import unittest

from tema.product import Coffee, Product


class Marketplace:
    """
    The central marketplace that manages producers, consumers, products, and carts.
    It uses locks and semaphores to handle concurrent access from multiple threads.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in their queue.
        """
        self.q_limit = queue_size_per_producer
        self.producers = []
        self.carts = []
        self.consumers = []
        # A lock to protect shared data structures like the list of carts.
        self.lock = Lock()

        # Setup for logging all marketplace events.
        logger = logging.getLogger("log_asc")
        logger.setLevel(logging.INFO)
        rfh = RotatingFileHandler('my_log.log', mode='w')
        rfh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        rfh.setFormatter(formatter)
        logger.addHandler(rfh)
        self.logger = logger
        self.mname = "MK"
        self.all_completed = False

    def log(self, msg, src):
        """Logs a message with the source."""
        self.logger.info(src + ":" + msg)

    def register_producer(self):
        """
        Registers a new producer in the marketplace, giving them a unique ID and a queue
        for their products, protected by semaphores.

        Returns:
            int: The ID of the newly registered producer.
        """
        new_producer = {
            'id': len(self.producers),
            'queue': [],
            # Semaphore to track empty slots in the producer's queue.
            'empty_sem': Semaphore(value=self.q_limit),
            # Semaphore to track filled slots in the producer's queue.
            'full_sem': Semaphore(0)
        }
        self.producers.append(new_producer)

        log_msg = "REG PROD [" + str(new_producer['id']) + ']'
        self.log(log_msg, self.mname)
        return new_producer['id']

    def publish(self, producer_id: int, product: Product):
        """
        Publishes a product from a producer to the marketplace. This is a non-blocking
        operation that will fail if the producer's queue is full.

        Args:
            producer_id (int): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise.
        """

        prod_queue = self.producers[producer_id]['queue']
        prod_esem = self.producers[producer_id]['empty_sem']
        prod_fsem = self.producers[producer_id]['full_sem']

        # Non-blocking acquire on the empty semaphore to check for available space.
        acquired = prod_esem.acquire(blocking=False)
        if not acquired:
            log_msg = "REJ PUB REQ S:PROD[" + 
                str(producer_id) + "] " + str(product)
            self.log(log_msg, self.mname)
            return False

        # Add the product to the queue and signal the full semaphore.
        prod_queue.append([product, True])
        log_msg = "ACC PUB REQ S:PROD[" + 
            str(producer_id) + "] " + 
            str(product) + " SLOTS [" + 
            str(self.q_limit - len(prod_queue)) + 
            "]"
        self.log(log_msg, self.mname)
        prod_fsem.release()
        return True

    def new_cart(self):
        """
        Creates a new shopping cart. This operation is protected by a lock to
        prevent race conditions when multiple consumers create carts simultaneously.

        Returns:
            int: The ID of the new cart.
        """
        self.lock.acquire()
        new_cart = {
            'id': len(self.carts),
            'items': [],
            'completed': False,
            'owner': ""
        }
        self.carts.append(new_cart)
        self.lock.release()

        log_msg = "REG CART [" + str(new_cart['id']) + "]"
        self.log(log_msg, self.mname)
        return new_cart['id']

    def assign_owner(self, cart_id: int, owner: str):
        """Assigns an owner to a cart and registers the consumer if not already present."""
        for cart in self.carts:
            if cart['id'] == cart_id:
                cart['owner'] = owner

        if owner not in self.consumers:
            self.consumers.append(owner)

    def product_search(self, name: str):
        """
        Searches for an available product by name across all producers.

        Args:
            name (str): The name of the product to search for.

        Returns:
            A tuple containing the product and its producer if found, otherwise None.
        """
        item_prod = None
        for producer in self.producers:
            for prod in producer['queue']:

                if prod[0].name == name and prod[1]:
                    item_prod = (prod, producer)
                    return item_prod

        return None

    def add_to_cart(self, cart_id: int, product: Product):
        """
        Adds a product to a shopping cart. It searches for an available product and,
        if found, marks it as unavailable and adds it to the cart.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was added successfully, False otherwise.
        """
        log_msg = "ADD REQ [" + self.carts[cart_id]['owner'] + 
            "][C" + str(cart_id) + "] " + str(product)

        # Find the cart by its ID.
        c_iter = iter(self.carts)
        cart = next((c for c in c_iter if c['id'] == cart_id), None)

        # Search for an available product.
        item_prod = self.product_search(product.name)
        
        # If an available product is found, add it to the cart.
        if item_prod is not None:
            req_item = item_prod[0]
            if req_item[1]:
                # Mark the product as "reserved" in the producer's queue.
                req_item[1] = False
                
                cart['items'].append(item_prod)
                log_msg = "ACC " + log_msg
                self.log(log_msg, self.mname)
                return True

        log_msg = "REJ " + log_msg
        self.log(log_msg, self.mname)
        return False

    def remove_from_cart(self, cart_id: int, product: Product):
        """
        Removes a product from a shopping cart, making it available again.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to remove.
        """
        req_prod_name = product.name
        prod_to_remove = None
        for prod in self.carts[cart_id]['items']:
            if prod[0][0].name == req_prod_name:
                # Mark the product as available again.
                prod[0][1] = True
                prod_to_remove = prod

        before_remove = len(self.carts[cart_id]['items'])
        self.carts[cart_id]['items'].remove(prod_to_remove)
        
        after_remove = len(self.carts[cart_id]['items'])
        log_msg = "DEL REQ " + str(prod_to_remove) + 
            str(before_remove) + " " + str(after_remove)
        self.log(log_msg, self.mname)

    def place_order(self, cart_id: int):
        """
        Finalizes an order. The products in the cart are permanently removed
        from the producers' inventories.
        """
        self.carts[cart_id]['completed'] = True

        # For each item in the order, decrement the producer's full semaphore
        # and remove the item from their queue.
        log_msg = "
"
        for item in self.carts[cart_id]['items']:
            producer = item[1]
            prod_esem = producer['empty_sem']
            prod_fsem = producer['full_sem']

            prod_fsem.acquire()
            if item[0] in producer['queue']:
                producer['queue'].remove(item[0])
            else:
                err_log = "ERR COULD NOT FIND " + str(item[0])
                self.log(err_log, self.mname)
            prod_esem.release()
        
        # Log the completed order.
        for item in self.carts[cart_id]['items']:
            log_msg += self.carts[cart_id]['owner'] + 
                ' bought ' + str(item[0][0]) + '
'

        print(log_msg[1:-1])
        self.log(log_msg, self.mname)

    def sign_out(self, cons: str):
        """Removes a consumer from the list of active consumers."""
        self.consumers.remove(cons)
        log_msg = "LOGOUT " + cons + " REMAINING " + str(len(self.consumers))
        self.log(log_msg, self.mname)


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    """

    def setUp(self):
        """Sets up a new Marketplace instance for each test."""
        self.marketplace = Marketplace(3)

    def test_1_register_producer(self):
        """Tests the registration of new producers."""
        market = self.marketplace
        ret = market.register_producer()
        self.assertEqual(ret, 0)
        ret = market.register_producer()
        self.assertEqual(ret, 1)

    def test_2_new_cart(self):
        """Tests the creation of new carts."""
        market = self.marketplace
        ret = market.new_cart()
        self.assertEqual(ret, 0)
        ret = market.new_cart()
        self.assertEqual(ret, 1)

    def test_3_publish(self):
        """Tests publishing products and the queue limit."""
        market = self.marketplace
        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        ret = market.publish(0, Coffee("TestCoff2", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        ret = market.publish(0, Coffee("TestCoff3", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        # This should fail as the queue is full.
        ret = market.publish(0, Coffee("TestCoff4", 10, "0.01", "Medium"))
        self.assertEqual(ret, False)

    def test_4_add_to_cart(self):
        """Tests adding available and unavailable products to the cart."""
        market = self.marketplace
        ret = market.register_producer()
        self.assertEqual(ret, 0)
        ret = market.new_cart()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        # This should succeed as the product is available.
        ret = market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        # This should fail as the product is not in the marketplace.
        ret = market.add_to_cart(0, Coffee("TestCoff2", 10, "0.01", "Medium"))
        self.assertEqual(ret, False)

    def test_5_remove_from_cart(self):
        """Tests removing a product from the cart."""
        market = self.marketplace
        ret = market.register_producer()
        self.assertEqual(ret, 0)
        ret = market.new_cart()
        self.assertEqual(ret, 0)

        market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        market.remove_from_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertNotIn(Coffee("TestCoff1", 10, "0.01", "Medium"),
                         [item[0][0] for item in market.carts[0]['items']])

    def test_6_place_order(self):
        """Tests placing an order and verifies that the product is removed from the producer's queue."""
        market = self.marketplace
        ret = market.register_producer()
        self.assertEqual(ret, 0)
        ret = market.new_cart()
        self.assertEqual(ret, 0)

        market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        # The product should be in the producer's queue, marked as reserved.
        self.assertTrue(any(p[0].name == "TestCoff1" and not p[1] for p in market.producers[0]['queue']))
        market.place_order(0)
        # The product should be gone from the producer's queue after the order is placed.
        self.assertFalse(any(p[0].name == "TestCoff1" for p in market.producers[0]['queue']))

    def test_7_assign_owner(self):
        """Tests assigning an owner to a cart."""
        market = self.marketplace
        ret = market.new_cart()
        self.assertEqual(ret, 0)

        market.assign_owner(0, "TestOwner")
        self.assertEqual(market.carts[0]['owner'], "TestOwner")

    def test_8_sign_out(self):
        """Tests signing out a consumer."""
        market = self.marketplace
        ret = market.new_cart()
        self.assertEqual(ret, 0)

        market.assign_owner(0, "TestOwner")
        self.assertEqual(market.carts[0]['owner'], "TestOwner")

        self.assertEqual(len(market.consumers), 1)
        market.sign_out("TestOwner")
        self.assertEqual(len(market.consumers), 0)


from threading import Thread
from time import sleep
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product


class Producer(Thread):
    """
    Represents a producer in the marketplace simulation. Each producer runs in its
    own thread, creating products and publishing them to the marketplace.
    """

    def __init__(self, products: List[Product], marketplace: Marketplace,
                 republish_wait_time: float, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (List[Product]): A list of products that the producer can create.
            marketplace (Marketplace): The central marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish a product.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, kwargs=kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.curr_index = 0
        self.curr_product = list(self.products[0])
        self.prod_id = -1

    def produce(self) -> Product:
        """
        Simulates producing a product, which involves a delay.
        
        Returns:
            Product: The product that was created.
        """
        self.curr_product[1] -= 1
        
        # Simulate the time it takes to produce the item.
        sleep(float(self.curr_product[2]))
        
        # If the quantity of the current product runs out, move to the next product.
        if self.curr_product[1] == 0:
            self.curr_index += 1

        return self.curr_product[0]

    def run(self):
        """
        The main execution loop for the producer thread.
        Continuously produces and publishes products to the marketplace.
        """
        log_msg = "Started producer " + str(self.kwargs['name'])
        self.marketplace.log(log_msg, str(self.kwargs['name']))

        self.prod_id = self.marketplace.register_producer()
        loop_flag = True
        while loop_flag:

            # Create a new product.
            produced_item = self.produce()
            if len(self.products) > self.curr_index:
                if self.curr_product[1] == 0:
                    self.curr_product = list(self.products[self.curr_index])
            else:
                # Loop back to the beginning of the product list if the end is reached.
                self.curr_product = list(self.products[0])
                self.curr_index = 0

            # Try to publish the product until it succeeds.
            was_published = False
            while not was_published:
                was_published = self.marketplace.publish(
                    self.prod_id, produced_item)
                if not was_published:
                    # If publishing fails, wait before retrying.
                    sleep(self.republish_wait_time)

                # Stop producing if there are no more active consumers.
                if len(self.marketplace.consumers) == 0:
                    loop_flag = False
                    break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A generic product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A tea product, which is a type of Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A coffee product, which is a type of Product."""
    acidity: str
    roast_level: str
