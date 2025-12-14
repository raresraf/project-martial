


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs["name"]

    def run(self):
        
        for cart in self.carts:

            cart_id = self.marketplace.new_cart()

            for data in cart:
                i = 0
                item = data["product"]
                operation = data["type"]

                while i < data["quantity"]:

                    if operation == "add":
                        available = self.marketplace.add_to_cart(cart_id, item)
                        if available:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)

                    if operation == "remove":
                        self.marketplace.remove_from_cart(cart_id, item)
                        i += 1

            order = self.marketplace.place_order(cart_id)

            self.marketplace.print_list(order, self.consumer_name)

from logging.handlers import RotatingFileHandler
import logging
import time
import os
if not os.path.exists("Logs"):
    os.makedirs("Logs")

def get_log(name):
    
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    
    logging.Formatter.converter = time.gmtime
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    
    handler = RotatingFileHandler('Logs/marketplace.log', maxBytes=2000, backupCount=20)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)

    return logger

from threading import Lock
import unittest
import io
import sys
sys.path.append("tema")
from logger import get_log

LOGGER = get_log('marketplace.log')

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(15)
        self.product_1 = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        self.product_2 = {
            "product_type": "Tea",
            "name": "Bubble Tea",
            "price": 10
        }
        self.producer_id = self.marketplace.register_producer()
        self.cart_id = self.marketplace.new_cart()
        self.cart_id_2 = self.marketplace.new_cart()

    def test_register_producer(self):
        
        print("\nTesting register_producer")
        self.assertIsNotNone(self.producer_id)
        self.assertEqual(self.producer_id, 0)
        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 0)
        self.assertEqual(self.marketplace.items[self.producer_id], [])

    def test_publish(self):
        
        print("\nTesting publish")
        self.assertEqual(self.marketplace.publish(self.producer_id, self.product_1), True)
        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 1)
        self.assertEqual(self.marketplace.items[self.producer_id], [self.product_1])
        self.marketplace.prod_num_items[self.producer_id] = 1000
        self.assertEqual(self.marketplace.publish(self.producer_id, self.product_1), False)

    def test_new_cart(self):
        
        print("\nTesting new_cart")
        self.assertIsNotNone(self.cart_id)
        self.assertEqual(self.cart_id, 0)
        self.assertEqual(self.marketplace.carts[self.cart_id], [])
        self.assertIsNotNone(self.cart_id_2)
        self.assertEqual(self.cart_id_2, 1)
        self.assertEqual(self.marketplace.carts[self.cart_id_2], [])

    def test_add_to_cart(self):
        
        print("\nTesting add_to_cart")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.assertEqual(self.marketplace.add_to_cart(self.cart_id, self.product_1), True)

        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 0)
        self.assertEqual(self.marketplace.items[self.producer_id], [])

        self.assertEqual(self.marketplace.carts[self.cart_id], [(self.product_1, self.producer_id)])
        self.assertEqual(self.marketplace.add_to_cart(self.cart_id, self.product_2), False)

    def test_remove_from_cart(self):
        
        print("\nTesting remove_from_cart")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        self.marketplace.remove_from_cart(self.cart_id, self.product_1)

        self.assertEqual(self.marketplace.prod_num_items[self.producer_id], 1)
        self.assertEqual(self.marketplace.items[self.producer_id], [self.product_1])
        self.assertEqual(self.marketplace.carts[self.cart_id], [])

    def test_place_order(self):
        
        print("\nTesting place_order")
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.publish(self.producer_id, self.product_2)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id_2, self.product_2)
        order_1 = []
        order_2 = []

        order_1 = self.marketplace.place_order(self.cart_id)
        self.assertEqual(order_1, [(self.product_1, self.producer_id)])
        self.assertEqual(self.marketplace.carts,
                         {self.cart_id_2: [(self.product_2, self.producer_id)]})

        order_2 = self.marketplace.place_order(self.cart_id_2)
        self.assertEqual(order_2, [(self.product_2, self.producer_id)])
        self.assertEqual(self.marketplace.carts, {})

        self.assertIsNotNone(order_1)
        self.assertIsNotNone(order_2)
        self.assertNotEqual(order_1, {})
        self.assertNotEqual(order_2, {})

    def test_print_list(self):
        
        cons_name = "Consumer 1"
        self.marketplace.publish(self.producer_id, self.product_1)
        self.marketplace.add_to_cart(self.cart_id, self.product_1)
        order = self.marketplace.place_order(self.cart_id)

        output = io.StringIO()
        sys.stdout = output
        self.marketplace.print_list(order, cons_name)
        sys.stdout = sys.__stdout__
        self.assertEqual(output.getvalue(),
                         'Consumer 1 bought {\'product_type\': \'Coffee\','
                         '\'name\': \'Indonezia\', \'acidity\': 5.05,'
                         '\'roast_level\': \'MEDIUM\','
                         '\'price\': 1}\n')

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        LOGGER.info('creating an instance of Marketplace')
        LOGGER.info('Max size of queue in Marketplace: %d', queue_size_per_producer)



        self.queue_size_per_producer = queue_size_per_producer

        self.num_prod = 0
        self.num_carts = 0
        
        self.prod_num_items = []
        
        self.items = {}
        
        self.carts = {}

        self.register_lock = Lock()
        self.new_cart_lock = Lock()
        self.cart_lock = Lock()
        
        self.print_lock = Lock()



    def register_producer(self):
        
        LOGGER.info("In method 'register_producer' from class Marketplace")


        with self.register_lock:
            prod_id = self.num_prod
            self.num_prod += 1

        self.prod_num_items.append(0)
        self.items[prod_id] = []
        LOGGER.info("Output of 'register_producer' - producer id: %d", prod_id)
        return prod_id

    def publish(self, producer_id, product):
        
        LOGGER.info("In method 'publish' from class Marketplace\
                    \nInputs: producer_id =%s; product=%s",
                    producer_id, product)
        if self.prod_num_items[producer_id] >= self.queue_size_per_producer:
            LOGGER.info("Output of 'publish' - %s", "False")
            return False
        self.items[producer_id].append(product)
        self.prod_num_items[producer_id] += 1
        LOGGER.info("Output of 'publish' - %s", "True")
        return True

    def new_cart(self):
        
        LOGGER.info("In method 'new_cart' from class Marketplace")
        with self.new_cart_lock:
            cart_id = self.num_carts
            self.num_carts += 1

        self.carts[cart_id] = []

        LOGGER.info("Output of 'new_cart' - cart_id = %s", cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        LOGGER.info("In method 'add_to_cart' from class Marketplace\nInputs:\
        cart_id =%s; product=%s", cart_id, product)
        found = False
        with self.cart_lock:
            for i, (_, val) in enumerate(self.items.items()):
                if product in val:


                    val.remove(product)
                    self.prod_num_items[i] -= 1
                    prod_id = i
                    found = True
                    break

        if found:
            self.carts[cart_id].append((product, prod_id))

        LOGGER.info("Output of 'add_to_cart' - %s", found)
        return found

    def remove_from_cart(self, cart_id, product):
        
        LOGGER.info("In method 'remove_from_cart' from class Marketplace\nInputs:\
        cart_id =%s; product=%s", cart_id, product)


        for item, producer in self.carts[cart_id]:
            if item is product:
                prod_id = producer
                self.carts[cart_id].remove((item, producer))
                break

        self.items[prod_id].append(product)

        with self.cart_lock:
            self.prod_num_items[prod_id] += 1
        LOGGER.info("Finished 'remove_from_cart', no return value")

    def place_order(self, cart_id):
        
        LOGGER.info("In method 'place_order' from class Marketplace\
        \nInputs:cart_id =%s", cart_id)
        res = self.carts.pop(cart_id)
        LOGGER.info("Output of 'place_order' - res = %s", res)
        return res

    def print_list(self, order, consumer_name):
        
        LOGGER.info("In method 'print_list' from class Marketplace\
        \nInputs:order =%s; consumer_name: %s", order, consumer_name)
        for item in order:
            with self.print_lock:
                print(consumer_name + " bought "+ str(item[0]))
        LOGGER.info("Finished 'print_list', no return value")


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products


        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        
        prod_id = self.marketplace.register_producer()
        while True:
            for (item, quantity, wait_time) in self.products:
                i = 0
                while i < quantity:
                    available = self.marketplace.publish(prod_id, item)

                    if available:
                        time.sleep(wait_time)
                        i += 1
                    else:
                        time.sleep(self.republish_wait_time)


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
