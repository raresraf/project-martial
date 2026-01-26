


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.id_cart = self.marketplace.new_cart()

    def run(self):
        
        for cart in self.carts:
            for curr in cart:
                
                for i in range(curr["quantity"]):
                    
                    if curr["type"] == "add":
                        ret = False
                        while not ret:
                            ret = self.marketplace.add_to_cart(self.id_cart, curr["product"])
                            if not ret:
                                sleep(self.retry_wait_time)
                    
                    else:
                        self.marketplace.remove_from_cart(self.id_cart, curr["product"])
                        sleep(self.retry_wait_time)
        
        final = self.marketplace.place_order(self.id_cart)
        
        for order in final:
            with self.marketplace.print_lock:
                print(f"cons{self.id_cart} bought {order}")

import unittest
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(asctime)s;%(message)s")
        self.rotating_file_handler = RotatingFileHandler('marketplace.log', 'w')
        self.rotating_file_handler.setLevel(logging.INFO)
        self.rotating_file_handler.setFormatter(self.formatter)
        self.log.addHandler(self.rotating_file_handler)

        
        self.prods = {}
        
        self.cons = {}
        
        self.no_prods = 0
        
        self.no_cons = 0
        
        self.producer_lock = Lock()
        
        self.publish_lock = Lock()
        
        self.cart_lock = Lock()
        
        self.add_cart_lock = Lock()
        
        self.print_lock = Lock()
        
        self.available_prods = []
        
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        
        self.log.info("Register product")
        with self.producer_lock:
            
            self.no_prods = self.no_prods + 1
        


        self.prods[self.no_prods] = []
        self.log.info("Register product final")
        return self.no_prods

    def publish(self, producer_id, product):
        
        self.log.info("Publish")
        with self.publish_lock:
            
            if len(self.prods[producer_id]) < self.queue_size_per_producer:
                
                self.prods[producer_id].append(product)
                
                self.available_prods.append(product)
                self.log.info("Publish final")
                return True
            return False

    def new_cart(self):
        
        self.log.info("New Cart")
        with self.cart_lock:
            
            self.no_cons = self.no_cons + 1
        


        self.cons[self.no_cons] = []
        self.log.info("New Cart final")
        return self.no_cons

    def add_to_cart(self, cart_id, product):
        
        self.log.info("Add to Cart")
        with self.cart_lock:
            
            if product in self.available_prods:
                
                self.available_prods.remove(product)
                
                for (i, products) in self.prods.items():
                    if product in products:
                        
                        self.cons[cart_id].append(product)
                        
                        self.prods[i].remove(product)
                        self.log.info("Add to Cart final")
                        return True
            return False

    def remove_from_cart(self, cart_id, product):
        
        self.log.info("Remove from cart")
        
        if product in self.cons[cart_id]:
            
            self.cons[cart_id].remove(product)
            


            self.available_prods.append(product)
        self.log.info("Remove from cart final")
    def place_order(self, cart_id):
        
        self.log.info("Place order")
        return self.cons[cart_id]

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(3)
    def test_register(self):
        
        self.assertEqual(self.marketplace.register_producer(), 1)
    def test_publish(self):
        


        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        self.assertEqual(self.marketplace.available_prods,
                         ["Tea(name='Wild Cherry', price=5, type='Black')"])
    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 1)
    def test_add_to_cart(self):
        


        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        self.marketplace.add_to_cart(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        print(self.marketplace.cons)
    def test_remove_from_cart(self):
        


        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        self.marketplace.add_to_cart(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        print(self.marketplace.cons)
        self.marketplace.remove_from_cart(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        print(self.marketplace.cons)
    def test_place_order(self):
        


        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        self.marketplace.add_to_cart(1, "Tea(name='Wild Cherry', price=5, type='Black')")
        print(self.marketplace.cons)
        print(self.marketplace.place_order(1))


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        
        while True:
            for prod in self.products:
                
                for quant in range(prod[1]):
                    
                    ret = self.marketplace.publish(self.id_producer, prod[0])
                    
                    if ret:
                        sleep(prod[2])
                    sleep(self.republish_wait_time)
