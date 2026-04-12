"""
This module defines the Consumer thread for a marketplace simulation.

The Consumer class simulates a customer that processes a list of shopping
actions from a list of carts.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer that interacts with a marketplace in a separate thread.

    Each consumer is initialized with a set of carts and performs the actions
    (add/remove products) specified in each one. It uses a lock during
    initialization to acquire a unique consumer ID.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        It acquires a unique consumer_id from the marketplace under a lock
        to ensure thread-safe initialization.

        :param carts: A list of carts, where each cart is a list of actions.
        :param marketplace: The marketplace object to interact with.
        :param retry_wait_time: The time in seconds to wait before retrying a failed 'add' action.
        :param kwargs: Keyword arguments for the Thread constructor.
        """
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

        
        Thread.__init__(self, **kwargs)

        
        # Lock to ensure thread-safe acquisition of a new cart ID.
        self.marketplace.consumer_lock.acquire()
        self.consumer_id = self.marketplace.new_cart()
        self.marketplace.consumer_lock.release()

    def run(self):
        """
        The main execution method for the consumer thread.

        It iterates through all assigned carts and, for each cart, executes all
        specified operations. A 'while' loop is used to process operations sequentially.
        Failed 'add' operations are retried after a delay.
        """
        
        # Iterate through all carts assigned to this consumer.
        for i in range(self.carts.__len__()):
            curr_operation_index = 0
            curr_set_index = i
            
            # Invariant: Process all operations in the current cart.
            while self.carts[curr_set_index].__len__() > curr_operation_index:
                
                operation_type = self.carts[curr_set_index][curr_operation_index]["type"]
                
                opeartion_product = self.carts[curr_set_index][curr_operation_index]["product"]
                
                operation_quantity = self.carts[curr_set_index][curr_operation_index]["quantity"]

                # Pre-condition: Check if the operation is 'add'.
                if operation_type == 'add':
                    j = 0
                    
                    # Invariant: Continue until the desired quantity has been added.
                    while j < operation_quantity:
                        
                        # Block Logic: If adding to cart fails, wait and retry.
                        if not self.marketplace.add_to_cart(self.consumer_id, opeartion_product):
                            sleep(self.retry_wait_time)
                            continue
                        j = j + 1
                else:
                    # Block Logic: Handle the 'remove' operation.
                    for _ in range(0, operation_quantity):
                        self.marketplace.remove_from_cart(self.consumer_id, opeartion_product)
                curr_operation_index = curr_operation_index + 1
        
        # After all operations, place the order and print the purchased products.
        for product in self.marketplace.place_order(self.consumer_id):
            print(Thread.getName(self), 'bought', product)


from threading import Lock
from copy import deepcopy

import logging
import logging.handlers
import time
import unittest

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.current_producer_id = 0
        self.current_cart_id = 0

        
        self.producer_database = {}
        
        
        self.consumer_database = {}


        formatter = logging.Formatter("[%(asctime)s]: %(message)s")
        formatter.converter = time.gmtime
        logging.Formatter.converter = formatter.converter
        logging.basicConfig(filename="marketplace.log", level=logging.DEBUG,
            format="[%(asctime)s]: %(message)s")
        self.logger = logging.getLogger()
        self.handler = logging.handlers.RotatingFileHandler("marketplace.log",
            maxBytes=0, backupCount=10, encoding=None, delay=False)
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        self.handler.doRollover()

    def register_producer(self):
        
        result = self.current_producer_id
        self.current_producer_id += 1
        self.logger.info("New producer id is: %d", result)
        return result

    def publish(self, producer_id, product):
        
        
        
        if producer_id not in self.producer_database:
            self.producer_database[producer_id] = []
            self.producer_database[producer_id].append(product)
            self.logger.info("Product %s is added to producer with id: %d",
                product.__str__(), producer_id)
        
        else:
            
            if self.producer_database[producer_id].__len__() >= self.queue_size_per_producer:
                return False
            
            self.producer_database[producer_id].append(product)
            self.logger.info("Product %s is added to producer with id: %d",
                product.__str__(), producer_id)
        return True

    def new_cart(self):
        
        result = self.current_cart_id
        self.current_cart_id += 1
        self.logger.info("New cart id is: %d", result)
        return result

    def add_to_cart(self, cart_id, product):
        
        
        if cart_id not in self.consumer_database:
            self.consumer_database[cart_id] = []

        
        for producer in range(self.current_producer_id):
            if product in self.producer_database[producer]:
                
                self.consumer_database[cart_id].append((producer, product))
                
                self.producer_database[producer].remove(product)
                self.logger.info("Add to cart %s the product %d from producer %d",
                    product.__str__(), cart_id, producer)
                self.logger.info("Removing from producer %d the product %s",
                    producer, product.__str__())
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        
        for cart_product in self.consumer_database[cart_id]:
            if product in cart_product:
                
                self.consumer_database[cart_id].remove(cart_product)
                
                self.producer_database[cart_product[0]].append(cart_product[1])
                self.logger.info("Removing from cart %d the product %s",
                    cart_id, product.__str__())
                self.logger.info("Adding to producer %d the product %s",
                    cart_product[0], cart_product[1].__str__())
                return

    def place_order(self, cart_id):
        
        
        
        
        
        result = []
        for cart_product in self.consumer_database[cart_id]:
            result.append(cart_product[1])
        deep_result = deepcopy(result)
        self.logger.info("The order %s was added to cart %d",
            result.__str__(), cart_id)
        return deep_result

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        


        self.marketplace = Marketplace(10)
        self.producer_id = self.marketplace.register_producer()
        self.cart_id = self.marketplace.new_cart()
        self.product1 = str({
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        })
        self.product2 = str({
            "product_type": "Tea",
            "name": "Linden",
            "type": "Herbal",
            "price": 9
        })

    def test_register_producer(self):
        
        my_range = 1000
        
        for i in range(1, my_range):
            self.assertEqual(self.marketplace.register_producer(), i, "Wrong id generated")

    def test_publish(self):
        


        self.marketplace.publish(self.producer_id, self.product1)
        self.marketplace.publish(self.producer_id, self.product2)
        self.assertEqual(self.marketplace.producer_database[self.producer_id][0],
            self.product1, "Publish wrong on product1")
        self.assertEqual(self.marketplace.producer_database[self.producer_id][1],
            self.product2, "Publish wrong on product2")

    def test_new_cart(self):
        
        my_range = 1000
        
        for i in range(1, my_range):
            self.assertEqual(self.marketplace.new_cart(), i, "Wrong id generated")

    def test_add_to_cart(self):
        


        self.marketplace.publish(self.producer_id, self.product1)
        self.marketplace.publish(self.producer_id, self.product2)

        self.marketplace.add_to_cart(self.cart_id, self.product2)

        self.assertEqual(self.marketplace.producer_database[self.producer_id][0], self.product1,
            "Something went wrong on adding to cart in producer_database")

        self.assertEqual(self.marketplace.consumer_database[self.cart_id][0][0], self.cart_id,
            "Something went wrong on adding to cart ID")

        self.assertEqual(self.marketplace.consumer_database[self.cart_id][0][1], self.product2,
            "Something went wrong on adding to cart in consumer_database")

    def test_remove_from_cart(self):
        


        self.marketplace.publish(self.producer_id, self.product1)
        self.marketplace.publish(self.producer_id, self.product2)

        self.marketplace.add_to_cart(self.cart_id, self.product2)
        self.marketplace.add_to_cart(self.cart_id, self.product1)

        self.marketplace.remove_from_cart(self.cart_id, self.product1)

        self.assertNotEqual(self.marketplace.consumer_database[self.cart_id][0],
            (self.producer_id, self.product1),
            "The product1 wasn't removed successfully from the cart")

        self.assertEqual(self.marketplace.consumer_database[self.cart_id][0],
            (self.producer_id, self.product2),
            "The product2 was touched unexpectedly")

        self.assertEqual(self.marketplace.producer_database[self.producer_id][0], self.product1,
            "The product removed from cart didn't return to the producer")

    def test_place_order(self):
        
        self.marketplace.publish(self.producer_id, self.product1)
        self.marketplace.publish(self.producer_id, self.product2)



        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product2)

        self.assertEqual(self.marketplace.place_order(self.cart_id)[0],
            self.marketplace.consumer_database[self.cart_id][0][1], "Wrong product1 returned")

        self.assertEqual(self.marketplace.place_order(self.cart_id)[1],
            self.marketplace.consumer_database[self.cart_id][1][1], "Wrong product2 returned")


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        Thread.__init__(self, **kwargs)
        
        self.marketplace.producer_lock.acquire()
        self.producer_id = self.marketplace.register_producer()
        self.marketplace.producer_lock.release()

    def run(self):
        
        while True:
            
            for product in self.products:
                quantity = product[1]
                
                while quantity > 0:
                    
                    if self.marketplace.publish(self.producer_id, product[0]):
                        quantity = quantity - 1
                        sleep(product[2])
                    
                    else:
                        sleep(self.republish_wait_time)
