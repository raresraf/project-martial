


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.id_cart = self.marketplace.new_cart()
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        for cart in self.carts:
            for spread in cart:
                tip, prod, qty = spread.values()
                i = 0
                while i < qty:
                    if tip == "add" and self.marketplace.add_to_cart(self.id_cart, prod):
                        i += 1
                    elif tip == "remove" and self.marketplace.remove_from_cart(self.id_cart, prod):
                        i += 1
                    else:
                        time.sleep(self.retry_wait_time)
        items_bought = self.marketplace.place_order(self.id_cart)
        if items_bought is not None:
            for product in items_bought:
                print(str(self.name) + " bought " + str(product))


from logging.handlers import RotatingFileHandler
from threading import Lock
import logging
import time

class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_ids = 0
        self.products = {}
        self.cart_ids = 0
        self.clients = {}

        
        self.register_lock = Lock()
        self.new_cart_lock = Lock()
        self.add_cart_lock = Lock()

        
        self.logger = logging.getLogger('marketplace_logger')
        self.logger.setLevel(logging.INFO)

        handler = RotatingFileHandler('marketplace.log', maxBytes=100000, backupCount=3, encoding='utf-8')


        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def register_producer(self):
        
        self.logger.info("Registering producer.")

        with self.register_lock:
            self.producer_ids += 1
            self.products[self.producer_ids] = []
            self.logger.info("Registered producer no %d.".format(self.producer_ids))
            
            return self.producer_ids
   
    def publish(self, producer_id, product):
        

        self.logger.info("Publishing product %s from producer no. %d".format(product, producer_id))

        id_p = int(producer_id)
        max_size = int(self.queue_size_per_producer)

        if len(self.products[id_p]) < max_size:
            self.products[id_p].append(product)
            self.logger.info("Published product %s to producer no %d successfully.".format(product, producer_id))
            return True
        else:
            self.logger.info("Failed to publish the product %s to producer no %d.".format(product, producer_id))
            return False

    def new_cart(self):
        
        self.logger.info("Adding new cart.")

        with self.new_cart_lock:
            self.cart_ids += 1
            self.clients[self.cart_ids] = []
            self.logger.info("Added cart no %d.".format(self.cart_ids))

            return self.cart_ids

    def add_to_cart(self, cart_id, product):
        
        self.logger.info("Adding product %s to cart no %d.".format(product, cart_id))

        for id_p in self.products.keys():
            if product in self.products[id_p]:
                with self.add_cart_lock:
                    self.products[id_p].remove(product)
                self.clients[cart_id].append([id_p, product])
                self.logger.info("Added product %s to cart no %d successfully.".format(product, cart_id))
                return True

        self.logger.info("Failed to add product %s to cart no %d.".format(product, cart_id))
        return False

    def remove_from_cart(self, cart_id, product):
        

        self.logger.info("Removing product %s from cart no %d.".format(product, cart_id))

        for spread in self.clients[cart_id]:
            id_p, prod = spread
            if product == prod:
                self.products[id_p].append(product)
                self.clients[cart_id].remove(spread)
                self.logger.info("Sucessfully removed product %s from cart no %d.".format(product, cart_id))
                return True

        self.logger.info("Failed to remove product %s from cart no %d.".format(product, cart_id))
        return False

    def place_order(self, cart_id):
        

        self.logger.info("Placing order for cart no %d.".format(cart_id))

        items_bought = []
        for spread in self.clients[cart_id]:
            _, prod = spread
            items_bought.append(prod)
        if len(items_bought) > 0:
            self.logger.info("There were %s items for cart no %d.".format(cart_id, items_bought))
            return items_bought


from math import prod
from pickletools import markobject
import unittest
from marketplace import Marketplace
from product import Tea

class MarketplaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.marketplace = Marketplace(1)
        self.test_product = Tea("tea", 10, "tea")
        return super().setUp()

    
    def test_register_producer(self):
        producer_id = self.marketplace.register_producer()

        self.assertEqual(producer_id, 1)

    
    def test_successful_publish(self):
        producer_id = self.marketplace.register_producer()
        result = self.marketplace.publish(producer_id, self.test_product)

        self.assertTrue(result)

    
    def test_failed_publish(self):
        producer_id = self.add_one_product()
        result = self.marketplace.publish(producer_id, self.test_product)



        self.assertFalse(result)

    
    def test_new_cart(self):
        cart_id = self.marketplace.new_cart()

        self.assertEqual(cart_id, 1)

    
    def test_successful_add_to_cart(self):
        self.add_one_product()
        
        cart_id = self.marketplace.new_cart()
        result = self.marketplace.add_to_cart(cart_id, self.test_product)

        self.assertTrue(result)

    
    def test_failed_add_to_cart(self):
        cart_id = self.marketplace.new_cart()
        result = self.marketplace.add_to_cart(cart_id, self.test_product)

        self.assertFalse(result)

    
    def test_successful_remove_from_cart(self):
        self.add_one_product()

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.test_product)
        result = self.marketplace.remove_from_cart(cart_id, self.test_product)

        self.assertTrue(result)

    
    def test_successful_remove_from_cart(self):
        cart_id = self.marketplace.new_cart()
        result = self.marketplace.remove_from_cart(cart_id, self.test_product)

        self.assertFalse(result)

    
    def test_place_order(self):
        self.add_one_product()

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.test_product)
        result = self.marketplace.place_order(cart_id)
        
        self.assertEqual(result[0], self.test_product)

    
    def add_one_product(self):
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, self.test_product)

        return producer_id
            
if __name__ == '__main__':
    unittest.main()>>>> file: producer.py


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.id_producer = self.marketplace.register_producer()
        self.republish_wait_time = republish_wait_time

    def run(self):
        while 1:
            for spread in self.products:
                id_p, cant, delay = spread
                i = 0
                while i < int(cant):
                    if self.marketplace.publish(str(self.id_producer), id_p):
                        time.sleep(delay)
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
