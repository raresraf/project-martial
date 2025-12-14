

from time import sleep
from threading import Thread


class Consumer(Thread):
    
    carts = []
    marketplace = None
    retry_wait_time = -1
    name = None

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts


        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']
        

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for cmd in cart:
                cmd_type = cmd['type']
                product = cmd['product']
                quantity = cmd['quantity']

                if cmd_type == 'add':
                    i = 0
                    while i < quantity:
                        product_added = self.marketplace.add_to_cart(cart_id, product)
                        
                        if product_added:
                            i += 1
                        
                        else:
                            sleep(self.retry_wait_time)
                elif cmd_type == 'remove':
                    
                    for i in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

            
            products = self.marketplace.place_order(cart_id)
            for i in products:
                print(self.name + ' bought ' + str(i))

import time
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import unittest
from random import randint
from tema.product import Product, Tea, Coffee


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        
        old_id = -1
        new_id = -1
        for _ in range(randint(3, 100)):
            old_id = self.marketplace.producers_ids
            new_id = self.marketplace.register_producer()
        self.assertEqual(old_id + 1, new_id)

    def test_new_cart(self):
        
        old_cart_id = -1
        new_cart_id = -1
        for _ in range(randint(3, 100)):
            old_cart_id = self.marketplace.carts_ids
            new_cart_id = self.marketplace.new_cart()
        self.assertEqual(old_cart_id + 1, new_cart_id)

    def test_publish_true(self):
        
        max_len = self.marketplace.queue_size_per_producer
        id_prod = self.marketplace.register_producer()
        for _ in range(randint(0, max_len - 2)):
            published = self.marketplace.publish(id_prod, Tea('test_tea', 10, 'test_type'))
            self.assertTrue(published)

    def test_publish_false(self):
        
        published = False
        max_len = self.marketplace.queue_size_per_producer
        id_prod = self.marketplace.register_producer()
        for _ in range(randint(max_len + 1, 2 * max_len)):
            published = self.marketplace.publish(id_prod, Tea('test_tea', 10, 'test_type'))
        self.assertFalse(published)

    def test_add_to_cart_true(self):
        
        cart = self.marketplace.new_cart()

        id1 = self.marketplace.register_producer()
        id2 = self.marketplace.register_producer()

        product = Tea('test_tea', 10, 'test_type')

        published = self.marketplace.publish(id1, product)
        self.assertTrue(published)

        published = self.marketplace.publish(id2, product)
        self.assertTrue(published)

        found = self.marketplace.add_to_cart(cart, product)
        self.assertTrue(found)

    def test_add_to_cart_false(self):
        
        cart = self.marketplace.new_cart()

        id1 = self.marketplace.register_producer()
        id2 = self.marketplace.register_producer()

        product1 = Tea('test_tea', 10, 'test_type')
        product2 = Coffee('test_coffee', 20, 'test', 'test')

        published = self.marketplace.publish(id1, product1)
        self.assertTrue(published)

        published = self.marketplace.publish(id2, product1)
        self.assertTrue(published)

        found = self.marketplace.add_to_cart(cart, product2)
        self.assertFalse(found)

    def test_remove_from_cart(self):
        
        cart = self.marketplace.new_cart()
        id1 = self.marketplace.register_producer()
        product = Tea('test_tea', 10, 'test_type')

        published = self.marketplace.publish(id1, product)
        self.assertTrue(published)

        found = self.marketplace.add_to_cart(cart, product)
        self.assertTrue(found)

        dim_before = len(self.marketplace.carts[cart])
        self.marketplace.remove_from_cart(cart, product)
        dim_after = len(self.marketplace.carts[cart])

        self.assertTrue(dim_before > dim_after)

    def test_place_order(self):
        
        c_1 = self.marketplace.new_cart()
        c_2 = self.marketplace.new_cart()
        id_1 = self.marketplace.register_producer()
        id_2 = self.marketplace.register_producer()
        p_1 = Tea('test_tea', 10, 'test_type')
        p_2 = Coffee('test_coffee', 20, 'test', 'test')

        published = self.marketplace.publish(id_1, p_2)
        self.assertTrue(published)
        published = self.marketplace.publish(id_2, p_1)
        self.assertTrue(published)
        published = self.marketplace.publish(id_2, p_2)
        self.assertTrue(published)
        published = self.marketplace.publish(id_1, p_1)
        self.assertTrue(published)

        found = self.marketplace.add_to_cart(c_2, p_2)
        self.assertTrue(found)
        found = self.marketplace.add_to_cart(c_1, p_1)
        self.assertTrue(found)
        found = self.marketplace.add_to_cart(c_1, p_2)
        self.assertTrue(found)
        found = self.marketplace.add_to_cart(c_1, p_1)
        self.assertTrue(found)

        prod = self.marketplace.place_order(c_1)
        self.assertEqual(len(prod), 3)
        self.assertEqual(prod[0], p_1)
        self.assertEqual(prod[1], p_2)
        self.assertEqual(prod[2], p_1)


class Marketplace:
    
    queue_size_per_producer = -1
    producers_ids = -1
    carts_ids = -1
    
    producers_queues = {}
    carts = {}
    
    lock = Lock()

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        handlers=[RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=10)],
        level=logging.INFO,
        format="[%(asctime)s] - [%(levelname)s] : %(funcName)s:%(lineno)d -> %(message)s",
        datefmt='%Y-%m-%d  %H:%M:%S'
    )

    def __init__(self, queue_size_per_producer):


        self.queue_size_per_producer = queue_size_per_producer
        

    def register_producer(self):
        
        logging.info('ENTER')

        with self.lock:
            self.producers_ids += 1
            new_id = self.producers_ids
        self.producers_queues[new_id] = []

        logging.info('EXIT')
        return new_id

    def publish(self, producer_id, product):
        
        logging.info('ENTER\n %s %s', str(producer_id), str(product))

        if len(self.producers_queues[producer_id]) < self.queue_size_per_producer:
            
            
            
            item = [product, -1]
            self.producers_queues[producer_id].append(item)
            logging.info('EXIT')
            return True

        logging.info('EXIT')
        return False

    def new_cart(self):
        
        logging.info('ENTER')

        with self.lock:
            self.carts_ids += 1
            new_cart = self.carts_ids
        self.carts[new_cart] = []

        logging.info('EXIT')
        return new_cart

    def add_to_cart(self, cart_id, product):
        
        logging.info('ENTER\n %s %s', str(cart_id), str(product))

        
        for key, value in self.producers_queues.items():
            for product_tuple in value:
                with self.lock:
                    
                    if product_tuple[0] == product and product_tuple[1] == -1:
                        
                        product_tuple[1] = cart_id
                        
                        self.carts[cart_id].append((product, key))
                        logging.info('EXIT')
                        return True

        logging.info('EXIT')
        return False

    def remove_from_cart(self, cart_id, product):
        
        logging.info('ENTER\n %s %s', str(cart_id), str(product))

        for product_tuple in self.carts.get(cart_id):
            if product_tuple[0] == product:
                producer = product_tuple[1]
                
                for item in self.producers_queues[producer]:
                    with self.lock:
                        
                        if item[0] == product and item[1] == cart_id:
                            
                            item[1] = -1
                            break
                self.carts[cart_id].remove(product_tuple)
                break

        logging.info('EXIT')

    def place_order(self, cart_id):
        
        logging.info('ENTER\n %s', str(cart_id))

        products = []
        for product_tuple in self.carts[cart_id]:
            product = product_tuple[0]
            producer_id = product_tuple[1]
            products.append(product)
            
            for item in self.producers_queues[producer_id]:
                with self.lock:
                    if item[0] == product and item[1] == cart_id:
                        self.producers_queues[producer_id].remove(item)
                        break

        logging.info('EXIT')
        return products

from time import sleep
from threading import Thread


class Producer(Thread):
    
    p_id = -1
    products = []
    marketplace = None
    republish_wait_time = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.p_id = marketplace.register_producer()
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        

    def run(self):
        while True:
            for product in self.products:
                product_type = product[0]
                quantity = product[1]
                time = product[2]

                i = 0
                while i < quantity:
                    
                    published = self.marketplace.publish(self.p_id, product_type)
                    if published:
                        sleep(time)
                        i += 1
                    
                    else:
                        sleep(self.republish_wait_time)
