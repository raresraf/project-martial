


"""
This module defines the Consumer thread for a marketplace simulation.

The Consumer class simulates a customer interacting with the marketplace,
processing a list of shopping actions like adding and removing products
from a cart, and finally placing an order.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer that interacts with a marketplace in a separate thread.

    Each consumer is initialized with a set of actions to perform on a series of carts.
    It simulates adding and removing products and then places an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of carts, where each cart is a list of actions to perform.
        :param marketplace: The marketplace object with which the consumer interacts.
        :param retry_wait_time: The time in seconds to wait before retrying a failed action.
        :param kwargs: Keyword arguments for the Thread constructor, including the consumer's name.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution method for the consumer thread.

        Acquires a new cart, iterates through the predefined shopping actions,
        and places the final order. It handles failed attempts to add products
        by waiting and retrying.
        """
        # Acquire a unique cart ID from the marketplace for this consumer session.
        cart_id = self.marketplace.new_cart()

        # Process each cart provided during initialization.
        for cart in self.carts:
            # Execute each action (add/remove) specified in the cart.
            for action in cart:
                action_type = action["type"]
                product_name = action["product"]
                quantity = int(action["quantity"])

                # Pre-condition: Determine if the action is to add or remove a product.
                if action_type == "add":
                    # Invariant: Continue until the desired quantity of the product has been added.
                    while quantity > 0:
                        # Attempt to add the product to the cart.
                        if self.marketplace.add_to_cart(cart_id, product_name) == True:
                            quantity -= 1
                        else:
                            # If adding fails (e.g., product unavailable), wait and retry.
                            time.sleep(self.retry_wait_time)
                elif action_type == "remove":
                    # Invariant: Continue until the desired quantity has been removed.
                    while quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, product_name)
                        quantity -= 1

        # After all actions are performed, place the final order.
        orders = self.marketplace.place_order(cart_id)
        # Print the final list of products purchased by the consumer.
        for order in orders:
            print(str(self.name) + " bought " + str(order))
>>>> file: marketplace.py



from threading import Lock
import logging
import unittest

from tema.producer import Producer
from tema.product import Product
from tema.product import Coffee


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {}
        self.producers_locks = {}
        self.create_producer_lock = Lock()
        self.carts = {}
        self.create_cart_lock = Lock()

        logging.basicConfig(filename='marketplace.log', level=logging.INFO)

    def register_producer(self):
        
        

        self.create_producer_lock.acquire()

        new_id = len(self.producers)
        self.producers[new_id] = []
        self.producers_locks[new_id] = Lock()

        self.create_producer_lock.release()



        logging.info("Registered new producer with id " + str(new_id))
        return new_id

    def publish(self, producer_id, product):
        
        logging.info("Start adding new product " + str(product) + " from producer " + str(producer_id))

        product_lock = self.producers_locks[producer_id]
        add_flag = True

        product_lock.acquire()
        products = self.producers[producer_id]

        if len(products) >= self.queue_size_per_producer:
            add_flag = False
        else:
            products.append(product)

        product_lock.release()
        logging.info("Added new product " + str(product) + " from producer " + str(producer_id))

        return add_flag

    def new_cart(self):
        
        

        self.create_cart_lock.acquire()

        new_id = len(self.carts)
        self.carts[new_id] = []

        self.create_cart_lock.release()



        logging.info("Created new cart with id " + str(new_id))
        return new_id

    def add_to_cart(self, cart_id, product):
        
        logging.info("In add cart adding " + str(product) + " to cart " + str(cart_id))
        producers_no = len(self.producers)

        cart = self.carts[cart_id]
        add_flag = False
        
        self.create_producer_lock.acquire()

        for producer_id, producer_products in self.producers.items():
            product_lock = self.producers_locks[producer_id]
            product_lock.acquire()
            if product in producer_products:
                cart.append((product, producer_id))
                producer_products.remove(product)
                add_flag = True
                product_lock.release()                
                break
            product_lock.release()

        self.create_producer_lock.release()
        if add_flag:
            logging.info("Added new product " + str(product) + " to cart " + str(cart_id))
        else:
            logging.info("Couldn't add " + str(product) + " to cart " + str(cart_id))
        return add_flag

    def remove_from_cart(self, cart_id, product):
        
        logging.info("Removing product " + str(product) + " from cart " + str(cart_id))

        cart = self.carts[cart_id]
        producer_id = 0
        for (cart_product, cart_producer_id) in cart:
            if product == cart_product:
                cart.remove((product, cart_producer_id))
                producer_id = cart_producer_id
                break

        self.producers_locks[producer_id].acquire()
        self.producers[producer_id].append(product)
        self.producers_locks[producer_id].release()

        logging.info("Removed product " + str(product) + " with prod id " + str(producer_id) + " from cart " + str(cart_id))


    def place_order(self, cart_id):
        
        products = []

        for product in self.carts[cart_id]:
            products.append(product[0])

        logging.info("Placed order for " + str(cart_id))
        return products

class TestMarketPlace(unittest.TestCase):
    def setUp(self):
        self.marketplace = Marketplace(15)

    def test_register_producer(self):
        producer = self.marketplace.register_producer()
        self.assertEqual(producer, 0)
        producer = self.marketplace.register_producer()
        self.assertEqual(producer, 1)
        producer = self.marketplace.register_producer()
        self.assertEqual(producer, 2)

    def test_publish(self):
        producer = self.marketplace.register_producer()
        coffee = Coffee("Cafea", 5, 2, "MEDIUM")
        for i in range (0, 15):
            self.assertEqual(self.marketplace.publish(producer, coffee), True)
        self.assertNotEqual(self.marketplace.publish(producer, coffee), True)

    def test_new_cart(self):
        cart = self.marketplace.new_cart()
        self.assertEqual(cart, 0)
        cart = self.marketplace.new_cart()
        self.assertEqual(cart, 1)
        cart = self.marketplace.new_cart()
        self.assertEqual(cart, 2)

    def test_add_to_cart(self):
        producer = self.marketplace.register_producer()
        cart = self.marketplace.new_cart()
        coffee = Coffee("Cafea", 5, 2, "MEDIUM")
        for i in range (0, 3):
            self.marketplace.publish(producer, coffee)
        for i in range (0, 3):
            self.assertEqual(self.marketplace.add_to_cart(cart, coffee), True)
        self.assertEqual(self.marketplace.add_to_cart(cart, coffee), False)

    def test_remove_from_cart(self):
        producer = self.marketplace.register_producer()
        cart = self.marketplace.new_cart()
        coffee = Coffee("Cafea", 5, 2, "MEDIUM")
        for i in range (0, 5):
            self.marketplace.publish(producer, coffee)
        for i in range (0, 3):
            self.marketplace.add_to_cart(cart, coffee)
        self.marketplace.remove_from_cart(cart, coffee)
        self.assertEqual(len(self.marketplace.carts[cart]), 2)
        self.marketplace.remove_from_cart(cart, coffee)
        self.assertEqual(len(self.marketplace.carts[cart]), 1)

    def test_place_order(self):
        producer = self.marketplace.register_producer()
        cart = self.marketplace.new_cart()
        coffee = Coffee("Cafea", 5, 2, "MEDIUM")
        for i in range (0, 5):
            self.marketplace.publish(producer, coffee)
        self.marketplace.add_to_cart(cart, coffee)
        self.assertEqual(self.marketplace.place_order(cart), [coffee])
        self.marketplace.add_to_cart(cart, coffee)
        self.assertEqual(self.marketplace.place_order(cart), [coffee, coffee])>>>> file: producer.py


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        max_product_number = self.marketplace.queue_size_per_producer
        producer_id = self.marketplace.register_producer()

        while True:
            for product in self.products:
                product_name = product[0]
                quantity = int(product[1])
                process_time = float(product[2])

                while quantity > 0:
                    time.sleep(process_time)

                    if self.marketplace.publish(producer_id, product_name) == True:
                        quantity -= 1
                    else:
                        time.sleep(self.republish_wait_time)>>>> file: product.py


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    
    acidity: str
    roast_level: str
