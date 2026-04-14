"""
This module implements a multi-threaded producer-consumer simulation of an
e-commerce marketplace.

It defines the core components of the simulation:
- Marketplace: The central hub where producers publish products and consumers buy them.
- Producer: A thread that generates products and adds them to the marketplace.
- Consumer: A thread that simulates a user shopping for products in carts.
- Product: Dataclasses representing items for sale.
- TestMarketplace: Unit tests to verify the marketplace logic.

The simulation uses threading, locks, and semaphores to manage concurrency.
"""

from threading import Thread
from time import sleep
from typing import List

from tema.marketplace import Marketplace


class Consumer(Thread):
    """
    Represents a consumer thread that shops for products in the marketplace.

    Each consumer processes a list of shopping carts, where each cart contains
    a list of products to add or remove.
    """

    def __init__(self, carts: List, marketplace: Marketplace, retry_wait_time: float, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts: A list of shopping lists for the consumer to process.
            marketplace: The shared Marketplace instance.
            retry_wait_time: Time in seconds to wait before retrying to add a
                             product if it's not available.
            **kwargs: Arguments for the parent Thread class (e.g., name).
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

        Iterates through each assigned shopping cart, registers a new cart in the
        marketplace, processes all 'add' and 'remove' operations for that cart,
        and finally places the order.
        """
        log_msg = "Started consumer " + str(self.kwargs['name'])
        self.marketplace.log(log_msg, str(self.kwargs['name']))

        # Process each shopping list (cart) assigned to this consumer.
        for cart in self.carts:
            # Create a new cart in the marketplace for the current shopping session.
            cart_id = self.marketplace.new_cart()
            self.marketplace.assign_owner(cart_id, str(self.kwargs['name']))
            cart_iter = iter(cart)
            log_msg = "NEW CART" + str(cart_id)
            self.marketplace.log(log_msg, str(self.kwargs['name']))

            
            req_item = next((item for item in cart_iter), None)

            # Process all actions (add/remove) in the shopping list.
            while req_item is not None:

                if req_item['type'] == 'add':
                    # Attempt to add the product to the cart.
                    res = self.marketplace.add_to_cart(
                        cart_id, req_item['product'])

                    if res:
                        # If successful, decrement the quantity to be added.
                        req_item['quantity'] -= 1

                        if req_item['quantity'] == 0:
                            req_item = next((item for item in cart_iter), None)
                    else:
                        # If the product is not available, wait and retry.
                        sleep(self.retry_wait_time)

                elif req_item['type'] == 'remove':
                    # Remove the product from the cart.
                    self.marketplace.remove_from_cart(
                        cart_id, req_item['product'])

                    req_item['quantity'] -= 1

                    if req_item['quantity'] == 0 :
                        req_item=next((item for item in cart_iter), None)

            # Finalize the order for the current cart.
            self.marketplace.place_order(cart_id)

        self.marketplace.sign_out(str(self.kwargs['name']))

import logging
from logging.handlers import RotatingFileHandler

from threading import Lock, Semaphore
import unittest
from tema.product import Coffee, Product


class Marketplace:
    """
    Manages the interaction between producers and consumers.

    This class acts as a shared resource, providing thread-safe methods for
    producers to publish products and for consumers to browse and purchase them.
    It uses semaphores to control the size of producer queues and a lock for
    modifying shared cart data.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer: The maximum number of products each
                                     producer can have in their queue at one time.
        """
        self.q_limit = queue_size_per_producer
        self.producers = []
        self.carts = []
        self.consumers = []
        self.lock = Lock()

        # Set up a rotating file logger for marketplace events.
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
        """Logs a message with a source identifier."""
        self.logger.info(src + ":" + msg)

    def register_producer(self):
        """
        Registers a new producer, creating a dedicated queue and semaphores for it.

        Returns:
            The ID of the newly registered producer.
        """
        new_producer = {
            'id': len(self.producers),
            'queue': [],
            'empty_sem': Semaphore(value=self.q_limit), # Controls available slots.
            'full_sem': Semaphore(0)                   # Controls available items.
        }
        self.producers.append(new_producer)

        log_msg = "REG PROD [" + str(new_producer['id']) + ']'
        self.log(log_msg, self.mname)
        return new_producer['id']



    def publish(self, producer_id: int, product: Product):
        """
        Publishes a product from a producer to the marketplace.

        This is a non-blocking operation. If the producer's queue is full,
        it will fail and return False.

        Args:
            producer_id: The ID of the publishing producer.
            product: The product to be published.

        Returns:
            True if the product was published successfully, False otherwise.
        """

        prod_queue = self.producers[producer_id]['queue']
        prod_esem = self.producers[producer_id]['empty_sem']
        prod_fsem = self.producers[producer_id]['full_sem']

        # Try to acquire a slot in the queue without blocking.
        acquired = prod_esem.acquire(blocking=False)
        if not acquired:
            log_msg = "REJ PUB REQ S:PROD[" + 
                str(producer_id) + "] " + str(product)
            self.log(log_msg, self.mname)
            return False

        # Add product to the queue and signal that an item is available.
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
        Creates a new, empty shopping cart.

        This operation is thread-safe.

        Returns:
            The ID of the new cart.
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
        """Assigns an owner to a cart and registers the consumer."""
        for cart in self.carts:
            if cart['id'] == cart_id:
                cart['owner'] = owner

        if owner not in self.consumers:
            self.consumers.append(owner)

    def product_search(self, name: str):
        """
        Searches all producer queues for an available product by name.

        Args:
            name: The name of the product to search for.

        Returns:
            A tuple of (product_entry, producer) if found, otherwise None.
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
        Adds a product to a specified shopping cart.

        Finds the product in the marketplace, marks it as unavailable for others,
        and adds it to the cart.

        Args:
            cart_id: The ID of the cart to add the item to.
            product: The product to add.

        Returns:
            True if the product was added successfully, False otherwise.
        """

        log_msg = "ADD REQ [" + self.carts[cart_id]['owner'] + 
            "][C" + str(cart_id) + "] " + str(product)

        
        c_iter = iter(self.carts)
        cart = next((c for c in c_iter if c['id'] == cart_id), None)

        # Search for an available product.
        item_prod = self.product_search(product.name)
        
        
        if item_prod is not None:
            # Mark the product as "taken" so no one else can add it.
            req_item = item_prod[0]
            if req_item[1]:
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
            cart_id: The ID of the cart.
            product: The product to remove.
        """

        # Find the product in the cart.
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
        Finalizes an order for a given cart.

        This removes the purchased items from their respective producer queues.
        
        Args:
            cart_id: The ID of the cart to be ordered.
        """
        
        self.carts[cart_id]['completed'] = True

        # For each item, acquire the full_sem and remove it from the producer's queue.
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
            # Release the empty_sem to signal a new slot is available.
            prod_esem.release()

        # Log the completed order.
        for item in self.carts[cart_id]['items']:
            log_msg += self.carts[cart_id]['owner'] + 
                ' bought ' + str(item[0][0]) + '
'

        print(log_msg[1:-1])
        self.log(log_msg, self.mname)

    def sign_out(self, cons: str):
        """Removes a consumer from the active list."""
        self.consumers.remove(cons)
        log_msg = "LOGOUT " + cons + " REMAINING " + str(len(self.consumers))
        self.log(log_msg, self.mname)


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    
    These tests verify the core functionalities of the marketplace, including
    producer/cart registration, publishing, and cart management.
    """

    def setUp(self):
        """Set up a new Marketplace instance for each test."""
        self.marketplace = Marketplace(3)

    def test_1_register_producer(self):
        """Tests that producers are registered with sequential IDs."""
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)
        ret = market.register_producer()
        self.assertEqual(ret, 1)

    def test_2_new_cart(self):
        """Tests that new carts are created with sequential IDs."""
        market = self.marketplace

        ret = market.new_cart()
        self.assertEqual(ret, 0)
        ret = market.new_cart()
        self.assertEqual(ret, 1)

    def test_3_publish(self):
        """Tests the producer queue limit by attempting to publish more items than allowed."""
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        ret = market.publish(0, Coffee("TestCoff2", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        ret = market.publish(0, Coffee("TestCoff3", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        # This one should fail as the queue size is 3.
        ret = market.publish(0, Coffee("TestCoff4", 10, "0.01", "Medium"))
        self.assertEqual(ret, False)

    def test_4_add_to_cart(self):
        """Tests adding available and unavailable items to a cart."""
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        # Should succeed as the item is available.
        ret = market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        # Should fail as the item is not in the marketplace.
        ret = market.add_to_cart(0, Coffee("TestCoff2", 10, "0.01", "Medium"))
        self.assertEqual(ret, False)

    def test_5_remove_from_cart(self):
        """Tests removing an item from a cart."""
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        ret = market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        market.remove_from_cart(
            0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(Coffee("TestCoff1", 10, "0.01", "Medium")
                         in market.carts[0]['items'], False)

    def test_6_place_order(self):
        """Tests that placing an order correctly removes the item from the producer's queue."""
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        ret = market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        self.assertEqual([Coffee("TestCoff1", 10, "0.01", "Medium"),
                         False] in market.producers[0]['queue'], True)
        market.place_order(0)
        self.assertEqual([Coffee("TestCoff1", 10, "0.01", "Medium"),
                         False] in market.producers[0]['queue'], False)

    def test_7_assign_owner(self):
        """Tests assigning an owner to a cart."""
        market = self.marketplace

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        market.assign_owner(0, "TestOwner")
        self.assertEqual(market.carts[0]['owner'], "TestOwner")

    def test_8_sign_out(self):
        """Tests that signing out removes a consumer from the active list."""
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
    Represents a producer thread that generates and publishes products.
    """

    def __init__(self, products: List[Product], marketplace: Marketplace,
                 republish_wait_time: float, **kwargs):
        """
        Initializes a Producer thread.
        
        Args:
            products: A list of products (with quantity and creation time) for
                      the producer to generate.
            marketplace: The shared Marketplace instance.
            republish_wait_time: Time in seconds to wait before retrying to
                                 publish a product if the queue is full.
            **kwargs: Arguments for the parent Thread class.
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
        Simulates the production of a single item.

        It decrements the quantity of the current product type, simulates
        production time, and returns the product.
        
        Returns:
            The produced Product instance.
        """
        
        self.curr_product[1] -= 1

        # Simulate the time it takes to produce the item.
        sleep(float(self.curr_product[2]))

        
        # If the quantity of the current product type is exhausted, move to the next.
        if self.curr_product[1] == 0:
            self.curr_index += 1

        return self.curr_product[0]

    def run(self):
        """
        The main execution loop for the producer thread.

        Continuously produces items and tries to publish them to the marketplace.
        If the marketplace queue for this producer is full, it waits and retries.
        """
        log_msg = "Started producer " + str(self.kwargs['name'])
        self.marketplace.log(log_msg, str(self.kwargs['name']))

        self.prod_id = self.marketplace.register_producer()
        loop_flag = True
        while loop_flag:

            
            produced_item = self.produce()
            if len(self.products) > self.curr_index:
                if self.curr_product[1] == 0:
                    self.curr_product = list(self.products[self.curr_index])
            else:
                self.curr_product = list(self.products[0])
                self.curr_index = 0

            
            # Continuously try to publish the item until successful.
            was_published = False
            while not was_published:
                was_published = self.marketplace.publish(
                    self.prod_id, produced_item)
                if not was_published:
                    sleep(self.republish_wait_time)

                # Stop producing if there are no more active consumers.
                if len(self.marketplace.consumers) == 0:
                    loop_flag = False
                    break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
