

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        cart_id = self.marketplace.new_cart()


        for cart in self.carts:
            for operation in cart:
                op_name = operation.get("type")
                product = operation.get("product")
                quantity = operation.get("quantity")

                if op_name == "add":
                    times = 0
                    while times < quantity:
                        wait_time = self.marketplace.add_to_cart(cart_id, product)
                        if wait_time:
                            times += 1
                        else:
                            time.sleep(self.retry_wait_time)
                elif op_name == "remove":
                    for times in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

        self.marketplace.place_order(cart_id)

import unittest
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock, currentThread

from tema.product import Product


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.producers_lock = Lock()
        self.last_producer_id = 0
        self.producers_products = {}

        self.carts_lock = Lock()
        self.last_cart_id = 0
        self.carts = {}

        self.print_lock = Lock()

        logging.basicConfig(
            handlers=[RotatingFileHandler("marketplace.log", maxBytes=100000, backupCount=3)],
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt="%d-%b-%y %H:%M:%S")



    def register_producer(self):
        
        logging.info("Entered function register_producer().")

        self.last_producer_id += 1
        self.producers_products[self.last_producer_id] = []

        logging.info("Function register_producer() returned %d.", self.last_producer_id)
        return self.last_producer_id

    def publish(self, producer_id, product):
        
        logging.info("Entered function publish. Param: producer_id = %d, product = %s",
                     producer_id, product)
        ret = False

        if len(self.producers_products[producer_id]) < self.queue_size_per_producer:
            self.producers_lock.acquire()
            self.producers_products[producer_id].append(product)
            self.producers_lock.release()
            ret = True

        logging.info("Function publish returned %s.", ret)
        return ret

    def new_cart(self):
        
        self.last_cart_id += 1
        self.carts[self.last_cart_id] = []

        return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        
        logging.info("Entered function add_to_cart. Param: cart_id = %d, product = %s",
                     cart_id, product)

        success = False

        
        products_dict = self.producers_products.copy()

        for key, val in products_dict.items():
            if product in val:
                self.producers_lock.acquire()
                self.producers_products[key].remove(product)
                self.producers_lock.release()

                self.carts_lock.acquire()
                
                self.carts[cart_id].append((product, key))
                self.carts_lock.release()
                success = True
                break

        logging.info("Function add_to_cart retuned %s.", success)
        return success

    def remove_from_cart(self, cart_id, product):
        
        logging.info("Entered function remove_from_cart. Param: cart_id = %s, product = %s",
                     cart_id, product)

        producer_id = -1

        
        for product_tuple in self.carts[cart_id]:
            if product_tuple[0] == product:
                self.carts_lock.acquire()
                self.carts[cart_id].remove(product_tuple)
                self.carts_lock.release()
                producer_id = product_tuple[1]
                break
        self.producers_products[producer_id].append(product)
        logging.info("Function remove_from_cart ended.")

    def place_order(self, cart_id):
        
        logging.info("Entered function place_order. Param: cart_id = %d", cart_id)

        order = []
        for product_tuple in self.carts[cart_id]:
            order.append(product_tuple[0])

        for prod in order:
            self.print_lock.acquire()
            print(currentThread().getName(), "bought", prod)
            self.print_lock.release()

        logging.info("Function place_order returned %s.", order)
        return order


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(15)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 1)

    def test_publish_no_space_left(self):
        
        producer_id = 1
        product = Product("Coffee", 10)
        self.marketplace.producers_products[producer_id] = [""] * 15
        self.assertFalse(self.marketplace.publish(producer_id, product))

    def test_publish(self):
        
        producer_id = 1
        product = Product("Coffee", 15)
        self.marketplace.producers_products[producer_id] = []
        self.assertTrue(self.marketplace.publish(producer_id, product))
        self.assertTrue(product in self.marketplace.producers_products[producer_id])

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 1)

    def test_add_to_cart_unexisting_product(self):
        
        cart_id = 1
        product = Product("Tea", 15)
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product))

    def test_add_to_cart(self):
        
        cart_id = 1
        producer_id = 1
        product = Product("Tea", 15)
        self.marketplace.producers_products[producer_id] = [product]
        self.marketplace.carts[cart_id] = []

        self.assertTrue(self.marketplace.add_to_cart(1, product))
        self.assertEqual(len(self.marketplace.producers_products[producer_id]), 0)
        self.assertTrue((product, producer_id) in self.marketplace.carts[cart_id])

    def remove_from_cart(self):
        
        cart_id = 1
        producer_id = 1
        product = Product("Tea", 15)
        self.marketplace.carts[cart_id] = [(product, producer_id)]

        self.marketplace.remove_from_cart(cart_id, product)

        self.assertEqual(len(self.marketplace.carts[cart_id]), 0)
        self.assertTrue(product in self.marketplace.producers_products[producer_id])

    def test_place_order(self):
        
        cart_id = 1
        producer_id = 1
        product1 = Product("Coffee", 10)
        product2 = Product("Tea", 15)

        self.marketplace.carts[cart_id] = [(product1, producer_id), (product2, producer_id)]
        self.assertEqual(self.marketplace.place_order(cart_id), [product1, product2])

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products


        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        while True:
            producer_id = self.marketplace.register_producer()
            for prod in self.products:
                for _ in range(prod[1]):
                    wait = self.marketplace.publish(producer_id, prod[0])
                    if wait:
                        time.sleep(prod[2])
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
