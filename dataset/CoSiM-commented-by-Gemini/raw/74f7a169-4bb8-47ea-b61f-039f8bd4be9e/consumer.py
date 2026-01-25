


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)


        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def add_product(self, cart_id, product, quantity):
        
        for _ in range(quantity):
            while not self.marketplace.add_to_cart(cart_id, product):
                sleep(self.retry_wait_time)

    def remove_product(self, cart_id, product, quantity):
        
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            
            for request in cart:
                command = request["type"]
                product = request["product"]
                quantity = request["quantity"]

                
                if command == "add":
                    self.add_product(cart_id, product, quantity)
                elif command == "remove":
                    self.remove_product(cart_id, product, quantity)

            
            order = self.marketplace.place_order(cart_id)

            
            self.marketplace.print_order(order, self.name)

import logging
import time
import unittest
from logging.handlers import RotatingFileHandler
from threading import Lock, current_thread


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0  
        self.cart_id = 0  
        self.producers = {}  
        self.carts = {}  
        self.products = {}  
        self.mutex = Lock()  

        
        self.logger = logging.getLogger("Logger")
        self.handler = RotatingFileHandler("marketplace.log", maxBytes=25000, backupCount=10)
        self.handler.setLevel(logging.INFO)
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s: %(message)s"))
        logging.Formatter.converter = time.gmtime
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def register_producer(self):
        
        with self.mutex:
            self.producers[self.producer_id] = []
            self.producer_id += 1
            self.logger.info("Thread %s has producer_id=%d", current_thread().name,
                             self.producer_id - 1)
            return str(self.producer_id - 1)

    def publish(self, producer_id, product):
        
        self.logger.info("Thread %s has producer_id=%s, product=%s", current_thread().name,
                         producer_id, product)
        producer_index = int(producer_id)

        
        if len(self.producers[producer_index]) == self.queue_size_per_producer:
            return False

        
        self.producers[producer_index].append(product)
        self.products[product] = producer_index

        return True

    def new_cart(self):
        
        with self.mutex:
            self.carts[self.cart_id] = []
            self.cart_id += 1
            self.logger.info("Thread %s has cart_id=%s", current_thread().name, self.cart_id - 1)
            return self.cart_id - 1

    def add_to_cart(self, cart_id, product):
        
        self.logger.info("Thread %s has cart_id=%d, product=%s", current_thread().name,
                         cart_id, product)

        for list_of_products in self.producers.values():
            
            if product in list_of_products:
                list_of_products.remove(product)
                self.carts[cart_id].append(product)

                return True

        return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("Thread %s has cart_id=%d, product=%s", current_thread().name,
                         cart_id, product)

        
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            self.producers[self.products[product]].append(product)

            return True

        return False

    def place_order(self, cart_id):
        
        self.logger.info("Thread %s has cart_id=%d", current_thread().name, cart_id)

        
        cart_content = self.carts[cart_id]
        self.carts.pop(cart_id)

        self.logger.info("Thread %s has cart_content=%s", current_thread().name, cart_content)
        return cart_content

    def print_order(self, order, name):
        
        with self.mutex:
            self.logger.info("Thread %s has order=%s, name=%s", current_thread().name,
                             order, name)
            for product in order:
                print(f"{name} bought {product}")


class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        
        for producer_id in range(100):
            self.assertEqual(self.marketplace.register_producer(), str(producer_id), "wrong id")

    def test_publish(self):
        
        prod_id = self.marketplace.register_producer()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(product1 in self.marketplace.producers[int(prod_id)],
                        "product is not on the marketplace")
        self.assertEqual(self.marketplace.products[product1], int(prod_id),
                         "don't recognize the product")

        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")
        self.assertTrue(product2 in self.marketplace.producers[int(prod_id)],
                        "product is not on the marketplace")
        self.assertEqual(self.marketplace.products[product2], int(prod_id),
                         "don't recognize the product")

        self.assertFalse(self.marketplace.publish(prod_id, product3), "failed not to publish")

    def test_new_cart(self):
        
        for cart_id in range(100):
            self.assertEqual(self.marketplace.new_cart(), cart_id, "wrong id")

    def test_add_to_cart(self):
        
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")
        self.assertFalse(self.marketplace.publish(prod_id, product3), "failed not to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(product1 in self.marketplace.carts[cart_id], "is not in the cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")
        self.assertTrue(product2 in self.marketplace.carts[cart_id], "is not in the cart")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product3),
                         "product should not be in the market")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product1),
                         "product should be already in the cart")

    def test_remove_from_cart(self):
        
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"
        product3 = "chocolate"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")

        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product1), "not in the cart")
        self.assertTrue(product1 not in self.marketplace.carts[cart_id], "is in cart")
        self.assertTrue(product1 in self.marketplace.producers[int(prod_id)],
                        "not in producer's list")
        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product2), "not in the cart")
        self.assertTrue(product2 not in self.marketplace.carts[cart_id], "is in cart")
        self.assertTrue(product2 in self.marketplace.producers[int(prod_id)],
                        "not in producer's list")
        self.assertFalse(self.marketplace.remove_from_cart(cart_id, product3), "not in the cart")

    def test_place_order(self):
        
        prod_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        product1 = "coffee"
        product2 = "tea"

        self.assertTrue(self.marketplace.publish(prod_id, product1), "failed to publish")
        self.assertTrue(self.marketplace.publish(prod_id, product2), "failed to publish")

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product1), "failed to add to cart")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product2), "failed to add to cart")

        self.assertEqual(self.marketplace.place_order(cart_id), ["coffee", "tea"],
                         "not the same order")


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        while True:
            
            for product in self.products:
                
                product_id = product[0]
                quantity = product[1]
                production_time = product[2]
                
                for _ in range(quantity):
                    
                    while not self.marketplace.publish(self.producer_id, product_id):
                        sleep(self.republish_wait_time)

                    sleep(production_time)


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
