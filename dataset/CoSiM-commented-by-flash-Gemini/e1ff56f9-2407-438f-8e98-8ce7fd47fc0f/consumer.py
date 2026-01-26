

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for entry in cart:
                
                (entry_type, product, quantity) =\
                    (entry["type"], entry["product"], entry["quantity"])
                aux = 0
                while aux < quantity:
                    
                    
                    if entry_type == "add":
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    else:
                        self.marketplace.remove_from_cart(cart_id, product)
                    aux = aux + 1

            
            self.marketplace.place_order(cart_id)

from threading import Lock, currentThread
from logging.handlers import RotatingFileHandler
import unittest
import logging
import tema.product

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {} 
        self.products = []
        self.consumers = {} 
                            
        self.lock = Lock()
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=5000, backupCount=10)


        self.logger.addHandler(handler)

    def register_producer(self):
        

        
        
        with self.lock:
            producer_id = len(self.producers) + 1
            self.producers[producer_id] = []


            self.logger.info("return from register_producer %s", str(producer_id))
            return producer_id

    def publish(self, producer_id, product):
        

        
        
        
        self.logger.info("input to publish %s %s", str(producer_id), str(product))

        if len(self.producers[producer_id]) > self.queue_size_per_producer:
            self.logger.info("return from publish False")
            return False

        self.producers[producer_id].append(product)
        self.products.append(product)
        self.logger.info("return from publish True")
        return True



    def new_cart(self):
        

        
        
        with self.lock:
            cart_id = len(self.consumers) + 1
            self.consumers[cart_id] = []


            self.logger.info("return from new_cart %s", str(cart_id))
            return cart_id

    def add_to_cart(self, cart_id, product):
        

        
        
        
        self.logger.info("input to add_to_cart %s %s", str(cart_id), str(product))

        with self.lock:
            aux = 0
            if cart_id in self.consumers:
                for producer_id in self.producers:
                    if product in self.producers[producer_id]:
                        aux = producer_id

                if aux == 0:
                    self.logger.info("return from add_to_cart False")
                    return False

            self.consumers[cart_id].append((aux, product))
            self.products.remove(product)
            self.producers[aux].remove(product)
            self.logger.info("return from add_to_cart True")
            return True

    def remove_from_cart(self, cart_id, product):
        

        
        
        self.logger.info("input to remove_from_cart %s %s", str(cart_id), str(product))

        with self.lock:
            if cart_id in self.consumers:
                for search in self.consumers[cart_id]:
                    if search[1] == product:
                        self.consumers[cart_id].remove(search)
                        self.products.append(product)
                        if len(self.producers[search[0]]) < self.queue_size_per_producer:
                            self.producers[search[0]].append(product)
                        return

    def place_order(self, cart_id):
        

        
        
        self.logger.info("input to place_order %s", str(cart_id))

        if cart_id in self.consumers:
            order_list = []
            for product in self.consumers[cart_id]:
                print(currentThread().getName() + " bought " + str(product[1]))

            self.logger.info("return from place_order %s", str(order_list))
            return order_list

        return []



class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(5)
        self.product = tema.product.Tea('Linden', 10, 'Herbal')
        self.product2 = tema.product.Coffee('Arabica', 10, '5.05', 'MEDIUM')

    def test_register_producer(self):
        
        self.marketplace.register_producer()
        self.assertEqual(len(self.marketplace.producers), 1, 'wrong number of producers')

    def test_publish(self):
        
        self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(1, self.product), True, 'failed to publish')

    def test_new_cart(self):
        
        self.marketplace.new_cart()
        self.assertEqual((self.marketplace.consumers), 1, 'wrong number of carts')

    def test_add_to_cart(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(1, self.product), True, 'failed to add')

    def test_remove_from_cart(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product)
        self.marketplace.remove_from_cart(1, self.product)
        self.assertEqual(self.product in self.marketplace.consumers[1], False, 'failed to remove')

    def test_place_order(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.publish(1, self.product2)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product)
        self.marketplace.add_to_cart(1, self.product2)
        aux = self.marketplace.place_order(1)
        correct_list = []
        correct_list.append(self.product)
        correct_list.append(self.product2)
        self.assertEqual(aux, correct_list, 'failed to place order')

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for (product, quantity, wait_time) in self.products:
                
                aux = 0
                while aux < quantity:
                    
                    
                    aux = aux + 1
                    while not self.marketplace.publish(self.prod_id, product):
                        time.sleep(self.republish_wait_time)
                    time.sleep(wait_time)


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
