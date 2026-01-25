


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            
            id_cart = self.marketplace.new_cart()
            for action in cart:
                if action['type'] == 'add':
                    
                    quantity = 0
                    while quantity < action['quantity']:
                        if self.marketplace.add_to_cart(id_cart, action['product']):
                            quantity += 1
                        else:
                            sleep(self.retry_wait_time)
                elif action['type'] == 'remove':
                    for _ in range(action['quantity']):
                        self.marketplace.remove_from_cart(id_cart, action['product'])
            
            self.marketplace.place_order(id_cart)


import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import BoundedSemaphore
from random import randint
import sys
import threading
import time
from uuid import UUID, uuid1
from unittest import TestCase

from tema.product import Coffee, Product, Tea


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.producer_queues = {}
        
        
        
        

        self.carts = {}
        
        
        
        
        self.carts_mutex = BoundedSemaphore(1)
        self.print_mutex = BoundedSemaphore(1)

        logger = logging.getLogger()
        
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=10**6, backupCount=10)
        
        formatter = logging.Formatter('%(asctime)s UTC %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def register_producer(self) -> UUID:
        
        logging.info('register_producer() was called')
        id_prod = uuid1()
        self.producer_queues[id_prod] = [BoundedSemaphore(1), 0, []]
        logging.info('register_producer() returned (%s)', id_prod)


        return id_prod

    def publish(self, producer_id: UUID, product: Product) -> bool:
        
        logging.info('publish(%s, %s) was called', producer_id, product)
        if self.producer_queues[producer_id][1] < self.queue_size_per_producer:
            self.producer_queues[producer_id][2].append([product, True])
            with self.producer_queues[producer_id][0]:
                self.producer_queues[producer_id][1] += 1
            logging.info('publish(%s, %s) returned True', producer_id, product)
            return True
        logging.info('publish(%s, %s) returned False', producer_id, product)


        return False

    def new_cart(self) -> int:
        
        logging.info('new_cart() was called')
        with self.carts_mutex:
            id_cart = randint(0, sys.maxsize)
            while id_cart in list(self.carts.keys()):
                id_cart = randint(0, sys.maxsize)
            self.carts[id_cart] = []
            logging.info('new_cart() returned %d', id_cart)


            return id_cart

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        
        logging.info('add_to_cart(%d, %s) was called', cart_id, product)
        for id_prod in self.producer_queues:
            with self.producer_queues[id_prod][0]:
                for prod in self.producer_queues[id_prod][2]:
                    if prod[0] == product and prod[1]:
                        prod[1] = False
                        self.producer_queues[id_prod][1] -= 1
                        self.carts[cart_id].append((id_prod, product))
                        logging.info('publish(%s, %s) returned True', cart_id, product)
                        return True
        logging.info('add_to_cart(%s, %s) returned False', cart_id, product)


        return False

    def remove_from_cart(self, cart_id: int, product: Product):
        
        logging.info('remove_from_cart(%d, %s) was called', cart_id, product)
        for item in self.carts[cart_id]:
            if item[1] == product:
                with self.producer_queues[item[0]][0]:
                    
                    for prod in self.producer_queues[item[0]][2]:
                        if prod[0] == product:
                            prod[1] = True
                            self.producer_queues[item[0]][1] += 1
                            self.carts[cart_id].remove((item[0], product))
                            logging.info('remove_from_cart(%d, %s) returned', cart_id, product)
                            return

    def place_order(self, cart_id: int):
        
        logging.info('place_order(%d) was called', cart_id)
        result = []
        for item in self.carts[cart_id]:
            result.append(item[1])
            with self.producer_queues[item[0]][0]:
                for prod in self.producer_queues[item[0]][2]:
                    if prod[0] == item[1]:
                        self.producer_queues[item[0]][2].remove(prod)
                        self.producer_queues[item[0]][1] -= 1
                        break
        
        self.carts.pop(cart_id)
        with self.print_mutex:
            for item in result:
                print(threading.current_thread().name, "bought", item)
        logging.info('place_order(%d) returned %s', cart_id, result)
        return result

class TestMarketplace(TestCase):
    "Clasa test marketplace"
    def setUp(self):
        "Initializare variabile locale"
        self.marketplace = Marketplace(1)
        self.coffee = Coffee('Indonezia', 4, '5.05', 'MEDIUM')
        self.tea = Tea('Linden', 9, 'Herbal')

    def test_register_producer(self):
        "Test test_register_producer"
        id_prod = self.marketplace.register_producer()
        self.assertIsInstance(id_prod, UUID, "Return type not UUID")
        self.assertEqual(self.marketplace.producer_queues[id_prod][1], 0, "Initial size not 0")
        self.assertEqual(len(self.marketplace.producer_queues[id_prod][2]), 0, "Queue not empty")
        self.assertNotEqual(id_prod, self.marketplace.register_producer(), "IDs equal")

    def test_publish(self):
        "Test test_publish"
        id_prod = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(id_prod, self.coffee), "First publish should pass")
        self.assertFalse(self.marketplace.publish(id_prod, self.tea), "Second publish should fail")

    def test_new_cart(self):
        "Test test_new_cart"
        cart_id = self.marketplace.new_cart()
        self.assertGreaterEqual(cart_id, 0, "Cart ID should not be negative")
        self.assertNotEqual(cart_id, self.marketplace.new_cart(), "IDs equal")

    def test_add_to_cart(self):
        "Test test_add_to_cart"
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.coffee), "Coffee is in store")
        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.tea), "Tea not in store")

    def test_remove_from_cart(self):
        "Test test_remove_from_cart"
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.coffee)
        self.marketplace.remove_from_cart(cart_id, self.tea)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1, "Item not in cart")
        self.marketplace.remove_from_cart(cart_id, self.coffee)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 0, "Item in cart, size 0")

    def test_place_order(self):
        "Test test_place_order"
        id_prod = self.marketplace.register_producer()
        self.marketplace.publish(id_prod, self.coffee)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.tea)
        self.marketplace.add_to_cart(cart_id, self.coffee)
        self.marketplace.remove_from_cart(cart_id, self.tea)
        self.assertEqual(self.marketplace.place_order(cart_id), [self.coffee], "Results differ")


from threading import Thread
from time import sleep

class Producer(Thread):
    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.product_actions = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        while True:
            
            for (product, quantity, delay) in self.product_actions:
                total = 0
                
                while total < quantity:
                    if self.marketplace.publish(self.id_producer, product):
                        total += 1
                        sleep(delay)
                    else:
                        sleep(self.republish_wait_time)
                    