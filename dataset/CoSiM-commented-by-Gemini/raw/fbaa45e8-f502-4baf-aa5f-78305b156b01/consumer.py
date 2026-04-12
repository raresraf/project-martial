"""
This module defines the Consumer thread for a marketplace simulation.

The Consumer class simulates a customer that processes a list of shopping
actions from a list of carts, using a semaphore for synchronization.
"""

from threading import Thread, Semaphore
import time

class Consumer(Thread):
    """
    Represents a consumer that interacts with a marketplace in a separate thread.

    This consumer uses a semaphore to control access to shared resources when
    creating a new cart. It processes a list of actions for various carts,
    handling retries for failed operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of carts, where each cart is a list of actions.
        :param marketplace: The marketplace object to interact with.
        :param retry_wait_time: The time in seconds to wait before retrying a failed action.
        :param kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

        # A semaphore to ensure thread-safe operations, such as creating a new cart.
        self.consumer_sem = Semaphore(1)

    def halt_consumer(self, halt_period):
        """
        Pauses the consumer thread for a specified period.
        
        :param halt_period: The duration to sleep, in seconds.
        """
        time.sleep(halt_period)

    def run(self):
        """
        The main execution method for the consumer thread.

        It iterates through each assigned cart, acquires a new cart ID under a
        semaphore lock, and processes all actions. Failed operations are retried
        after a delay. After all actions for a cart are complete, an order is placed.
        """
        # Process each cart in the assigned list.
        for cart in self.carts:
            
            # Use a semaphore to ensure that creating a new cart is an atomic operation.
            self.consumer_sem.acquire()
            cart_id = self.marketplace.new_cart()
            self.consumer_sem.release()

            # Process each operation (add/remove) in the cart.
            for operation in cart:

                actual_quantity = int(operation['quantity'])
                # Invariant: The inner loop runs until the desired quantity for the operation is met.
                for _ in range(actual_quantity):

                    selected_operation = operation['type']
                    requested_item = operation['product']

                    # Block Logic: This loop will retry the operation until it succeeds.
                    while True:

                        if selected_operation == 'add':
                            self.consumer_sem.acquire()
                            add_op = self.marketplace.add_to_cart(cart_id, requested_item)
                            self.consumer_sem.release()

                            # If the add operation was successful, exit the retry loop.
                            if add_op:
                                break

                            # If the operation failed, wait before retrying.
                            self.halt_consumer(self.retry_wait_time)

                        elif selected_operation == 'remove':
                            self.consumer_sem.acquire()
                            rem_op = self.marketplace.remove_from_cart(cart_id, requested_item)
                            self.consumer_sem.release()

                            # If the remove operation was successful (indicated by None), exit the retry loop.
                            if rem_op is None:
                                break

                            # If the operation failed, wait before retrying.
                            self.halt_consumer(self.retry_wait_time)

            # After processing all operations, place the order for the current cart.
            self.marketplace.place_order(cart_id)


from threading import Lock
import threading


import time
import unittest
import logging
from logging.handlers import RotatingFileHandler


logging.basicConfig(
        handlers=[RotatingFileHandler('./marketplace.log', maxBytes=100000, backupCount=10)],
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = time.gmtime

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), self.marketplace.producer_id - 1)

    def test_publish_true(self):
        
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(0, "prod1"))

    def test_publish_false(self):
        
        self.marketplace.register_producer()

        self.marketplace.items_per_producer[0] = 6
        self.assertFalse(self.marketplace.publish(0, "prod1"))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), self.marketplace.num_carts - 1)

    def test_add_to_cart_true(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "prod1")

        cart_id = self.marketplace.new_cart()

        self.assertTrue(self.marketplace.add_to_cart(cart_id, "prod1"))

    def test_add_to_cart_false(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "prod1")

        cart_id = self.marketplace.new_cart()

        self.assertFalse(self.marketplace.add_to_cart(cart_id, "prod2"))

    def test_remove_from_cart(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "prod1")

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod1")

        self.marketplace.remove_from_cart(cart_id, "prod1")
        self.assertEqual(self.marketplace.carts_info[cart_id], [])

    def test_place_order(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "prod1")

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod1")
        product_list_test = self.marketplace.place_order(cart_id)
        self.assertEqual(product_list_test, ['prod1'])

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        
        self.queue_size_per_producer = queue_size_per_producer

        
        self.items_per_producer = []

        
        self.available_in_market = []

        
        self.manufacturer = {}

        
        self.num_carts = 0

        
        self.producer_id = 0

        
        self.carts_info = {}

        
        self.consumer_lock_object = Lock()

        
        self.cart_lock_object = Lock()

    def register_producer(self):
        

        
        producer_id = self.producer_id
        self.producer_id += 1

        
        self.items_per_producer.append([])

        logging.info(f'Method "register_producer" - output: producer_id = {producer_id}')
        return producer_id

    def publish(self, producer_id, product):
        
        logging.info(f'Method "publish" - input: producer_id = {producer_id}, product = {product}')

        
        current_queue_size = len(self.items_per_producer[producer_id])

        
        if current_queue_size < self.queue_size_per_producer:

            
            self.items_per_producer[producer_id].append(product)

            
            self.manufacturer[product] = producer_id

            
            self.available_in_market.append(product)

            logging.info(f'Method "publish" - output: True')
            return True

        
        logging.info(f'Method "publish" - output: False')
        return False

    def new_cart(self):
        
        
        cart_id = self.num_carts
        self.num_carts += 1

        
        if cart_id not in self.carts_info.keys():
            self.carts_info[cart_id] = []

        logging.info(f'Method "new_cart" - output: cart_id = {cart_id}')
        return cart_id

    def add_to_cart(self, cart_id, product):
        

        logging.info(f'Method "add_to_cart" - input: cart_id = {cart_id}, product = {product}')

        
        if product not in self.available_in_market:
            logging.info(f'Method "add_to_cart" - output: False')
            return False

        self.cart_lock_object.acquire()

        
        actual_producer = self.manufacturer[product]

        if product in self.items_per_producer[actual_producer]:
            self.items_per_producer[actual_producer].remove(product)

        
        if product in self.available_in_market:
            self.available_in_market.remove(product)
        self.cart_lock_object.release()

        
        self.carts_info[cart_id].append(product)

        logging.info(f'Method "add_to_cart" - output: True')
        return True

    def remove_from_cart(self, cart_id, product):
        
        logging.info(f'Method "remove_from_cart" - input: cart_id = {cart_id}, product = {product}')
        
        self.available_in_market.append(product)

        
        actual_producer = self.manufacturer[product]
        self.items_per_producer[actual_producer].append(product)

        
        if product in self.carts_info[cart_id]:
            self.carts_info[cart_id].remove(product)

        return

    def place_order(self, cart_id):
        
        logging.info(f'Method "place_order" - input: cart_id = {cart_id}')

        products = []
        
        for cart_key, cart_value in self.carts_info.items():
            if cart_key == cart_id:
                products = cart_value
                del self.carts_info[cart_key]
                break

        self.consumer_lock_object.acquire()
        
        for product in products:
            print(f'{threading.currentThread().name} bought {product}')
        self.consumer_lock_object.release()

        logging.info(f'Method "place_order" - products = {products}')
        return products


from threading import Thread, Semaphore
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        
        self.producer_sem = Semaphore(1)

    def halt_producer(self, halt_period):
        time.sleep(halt_period)

    def run(self):

        
        
        self.producer_sem.acquire()
        current_prod_id = self.marketplace.register_producer()
        self.producer_sem.release()

        
        while True:

            
            for product in self.products:

                
                product_obj = product[0]
                quantity = product[1]
                wait_time = product[2]
                idx = 0

                
                while idx < int(quantity):

                    
                    if self.marketplace.publish(current_prod_id, product_obj) is False:

                        self.halt_producer(self.republish_wait_time)
                    
                    else:
                        self.halt_producer(wait_time)
                        idx += 1
