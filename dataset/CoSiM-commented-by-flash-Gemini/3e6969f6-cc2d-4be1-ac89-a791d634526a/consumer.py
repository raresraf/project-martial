


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        
        for cart in self.carts:
            consumer_id = self.marketplace.new_cart()
            for product in cart:
                size = product["quantity"]
                if product["type"] == "add":
                    while size > 0:
                        if self.marketplace.add_to_cart(consumer_id, product["product"]) is True:
                            size -= 1
                        else:
                            time.sleep(self.retry_wait_time)

                else:
                    while size > 0:
                        self.marketplace.remove_from_cart(
                            consumer_id, product["product"])
                        size -= 1

            final_products = self.marketplace.place_order(consumer_id)
            for product in final_products:
                print(self.kwargs['name'], "bought", product, flush=True)


import time
from threading import RLock
import unittest
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import sys
sys.path.insert(1, './tema/')
from product import *


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_size = 0
        self.consumer_size = 0
        self.lock = RLock()
        self.carts = []
        self.shop_items = []
        self.products_from_producer = []
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            'marketplace.log', maxBytes=2000, backupCount=10)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)8s: %(message)s')
        handler.setFormatter(formatter)
        logging.Formatter.converter = time.gmtime
        self.logger.addHandler(handler)



    def register_producer(self):
        
        
        self.logger.info("Entering register_producer function")
        self.lock.acquire()
        var = self.producer_size
        self.producer_size += 1
        self.products_from_producer.append(0)
        self.lock.release()
        self.logger.info(
            "Leaving register_producer function with result %d", var)
        return var

    def publish(self, producer_id, product):
        
        
        self.logger.info(
            "Entering publish function with producer_id=%d and product=%s", producer_id, product)
        if self.products_from_producer[producer_id] == self.queue_size_per_producer:
            self.logger.info("Leaving publish function with result %r", False)
            return False
        else:
            prod = {}
            prod["id"] = producer_id
            prod["product"] = product

            self.products_from_producer[producer_id] += 1
            self.shop_items.append(prod)
            self.logger.info("Leaving publish function with result %r", True)
            return True



    def new_cart(self):
        
        
        
        self.logger.info("Entering new_cart function")
        self.lock.acquire()
        var = self.consumer_size
        self.consumer_size += 1
        self.carts.append([])
        self.lock.release()
        self.logger.info("Leaving new_cart function with result %d", var)
        return var

    def add_to_cart(self, cart_id, product):
        

        
        self.logger.info(
            "Entering add_to_cart function with cart_id =%d and product=%s", cart_id, product)

        done = 0
        self.lock.acquire()
        for prod in self.shop_items:
            if prod["product"] == product:
                self.carts[cart_id].append(prod)
                done = 1
                self.shop_items.remove(prod)
                break
        self.lock.release()
        self.logger.info(
            "Leaving add_to_cart function with result %r", bool(done))

        return bool(done)

    def remove_from_cart(self, cart_id, product):
        
        
        self.logger.info(
            "Entering remove_from_cart function with cart_id =%d and product=%s", cart_id, product)
        for prod in self.carts[cart_id]:
            if prod["product"] == product:
                self.shop_items.append(prod)
                self.carts[cart_id].remove(prod)
                break
        self.logger.info("Leaving add_to_cart function")

    def place_order(self, cart_id):


        

        
        self.logger.info(
            "Entering place_order function with cart_id =%d", cart_id)
        final_list = []
        
        for prod in self.carts[cart_id]:
            self.products_from_producer[prod["id"]] -= 1
            final_list.append(prod["product"])
        self.logger.info(
            "Leaving place_order function with list: %s", final_list)
        return final_list


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(3)

    

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.register_producer(), 3)
        self.assertEqual(self.marketplace.register_producer(), 4)
        self.assertEqual(self.marketplace.register_producer(), 5)

    

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.new_cart(), 3)
        self.assertEqual(self.marketplace.new_cart(), 4)
        self.assertEqual(self.marketplace.new_cart(), 5)

    

    def test_publish(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))

    

    def test_add_to_cart(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertFalse(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))

    

    def test_remove_from_cart(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertEqual(len(self.marketplace.carts[0]), 2)
        self.marketplace.remove_from_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM'))
        self.assertEqual(len(self.marketplace.carts[0]), 1)

    

    def test_place_order(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.add_to_cart(
            0, Tea(name='English Breakfast', price=2, type='Black')))
        self.assertFalse(self.marketplace.add_to_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertEqual(len(self.marketplace.carts[0]), 2)
        self.marketplace.remove_from_cart(
            0, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM'))
        self.assertEqual(len(self.marketplace.carts[0]), 1)
        self.assertTrue(self.marketplace.add_to_cart(
            0, Tea(name='White Peach', price=5, type='White')))
        list = [Tea(name='English Breakfast', price=2, type='Black'),
                Tea(name='White Peach', price=5, type='White')]
        self.assertEqual(self.marketplace.place_order(0), list)
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    

    def run(self):
        producer_id = self.marketplace.register_producer()
        while self.kwargs['daemon'] is True:
            for product in self.products:
                count_product = product[1]
                while count_product > 0:


                    if self.marketplace.publish(producer_id, product[0]) is True:
                        count_product -= 1
                        time.sleep(product[2])
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
