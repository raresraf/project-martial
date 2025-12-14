


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()


            for operation in cart:
                
                if operation["type"] == "add":
                    count = 0
                    
                    while count < operation["quantity"]:
                        
                        if self.marketplace.add_to_cart(cart_id, operation["product"]):


                            count += 1
                        else:
                            
                            sleep(self.retry_wait_time)
                
                elif operation["type"] == "remove":
                    count = 0
                    
                    while count < operation["quantity"]:
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        
                        count += 1
             
            products_bought = self.marketplace.place_order(cart_id)
            for product in products_bought:
                print(self.kwargs["name"], "bought", product, flush=True)
                >>>> file: marketplace.py


from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
from distutils.log import INFO
import time
import unittest
import sys
sys.path.insert(1, './tema')
from product import Tea, Coffee

class Marketplace:
    
    
    myLogger = logging.getLogger('marketplace.log')
    myLogger.setLevel(INFO)
    file_handler = RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=5)
    file_handler.setLevel(INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    formatter.converter = time.gmtime
    file_handler.setFormatter(formatter)
    myLogger.addHandler(file_handler)

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        
        
        self.available_products = []
        self.cart_id = -1
        
        
        
        self.carts = []
        
        
        self.queue_size = []

        self.lock = Lock()

    def register_producer(self):
        
        self.lock.acquire()
        self.myLogger.info("Entered method register_producer")
        
        self.producer_id += 1
        
        self.available_products.append([])
        self.available_products[self.producer_id] = []
        self.queue_size.append(0)
        
        self.myLogger.info("Exited method register_producer with producer_id=%s", self.producer_id)
        self.lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        
        self.myLogger.info("Entered method publish with producer_id=%s, product=%s",
                           producer_id, product)
        id_producer = int(producer_id)
        
        if self.queue_size[id_producer] < self.queue_size_per_producer:
            
            self.available_products[id_producer].append(product[0])
            self.queue_size[id_producer] += 1
            self.myLogger.info("Exited method publish with True")
            return True
        self.myLogger.info("Exited method publish with False")
        return False

    def new_cart(self):
        
        self.lock.acquire()
        self.myLogger.info("Entered method new_cart")
        self.cart_id += 1
        
        self.carts.append([])
        self.carts[self.cart_id] = []
        self.myLogger.info("Exited method new_cart with cart_id=%s", self.cart_id)
        self.lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        self.lock.acquire()
        self.myLogger.info("Entered method add_to_cart with cart_id=%s, product=%s",
                           cart_id, product)
        ids = 0
        
        while ids <= self.producer_id:
            
            if product in self.available_products[ids]:
                
                self.carts[cart_id].append([product, ids])
                
                self.available_products[ids].remove(product)
                self.myLogger.info("Exited method add_to_cart with True")
                self.lock.release()
                return True
            
            ids += 1
        self.myLogger.info("Exited method add_to_cart with False")
        self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.myLogger.info("Entered method remove_from_cart with cart_id=%s, product=%s",
                           cart_id, product)
        
        for produs in self.carts[cart_id]:
            if produs[0] == product:
                
                self.carts[cart_id].remove([product, produs[1]])
                self.available_products[produs[1]].append(product)
                self.myLogger.info("Exited method remove_from_cart")
                return

    def place_order(self, cart_id):
        
        self.myLogger.info("Entered place_order with cart_id=%s", cart_id)
        cart_products = []
        
        for products in self.carts[cart_id]:
            cart_products.append(products[0])
            
            self.lock.acquire()
            self.queue_size[products[1]] -= 1
            self.lock.release()
        self.myLogger.info("Exited place_order.")
        return cart_products

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        
        self.marketplace = Marketplace(2)
        
        self.tea = Tea(name='Linden', price=9, type='Herbal')
        self.coffee = Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.register_producer(), 3)

    def test_publish(self):
        
        
        self.assertTrue(self.marketplace.publish(0, self.tea))
        self.assertListEqual(self.marketplace.available_products[0], [self.tea])
        self.assertEqual(self.marketplace.queue_size[0], 1)
        
        self.assertTrue(self.marketplace.publish(0, self.coffee))
        self.assertListEqual(self.marketplace.available_products[0], [self.tea, self.coffee])
        self.assertEqual(self.marketplace.queue_size[0], 2)
        
        self.assertFalse(self.marketplace.publish(0, self.tea))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.new_cart(), 3)

    def test_add_to_cart(self):
        
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        self.assertFalse(self.marketplace.add_to_cart(0, self.coffee))

    def test_remove_from_cart(self):
        
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        self.marketplace.remove_from_cart(0, self.tea)
        
        self.assertListEqual(self.marketplace.carts[0], [])
        
        self.assertListEqual(self.marketplace.available_products[0], [self.tea])

    def test_place_order(self):
        
        
        self.marketplace.publish(0, self.tea)
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.tea))
        self.assertListEqual(self.marketplace.carts[0], [[self.tea, 0]])
        
        self.assertListEqual(self.marketplace.available_products[0], [])
        
        self.assertEqual(self.marketplace.queue_size[0], 1)
        
        cart_list = self.marketplace.place_order(0)
        self.assertListEqual(cart_list, [self.tea])
        
        
        self.assertEqual(self.marketplace.queue_size[0], 0)


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):


        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                count = 0
                
                while count < product[1]:
                    
                    if self.marketplace.publish(producer_id, product):
                        
                        sleep(product[2])
                        
                        count += 1
                    else:
                        
                        sleep(self.republish_wait_time)


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
