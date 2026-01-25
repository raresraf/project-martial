


from threading import Thread
import time
import sys

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name")

    def run(self):
        for cart in self.carts:
            id_cart = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]
                my_type = operation["type"]
                product = operation["product"]
                contor = 0
                while contor < quantity:
                    
                    if my_type == "add":
                        if self.marketplace.add_to_cart(id_cart, product):
                            contor = contor + 1
                        else:
                            time.sleep(self.retry_wait_time)
                    
                    else:
                        self.marketplace.remove_from_cart(id_cart, product)
                        contor = contor + 1
            placed_order = self.marketplace.place_order(id_cart)
            for each_p in placed_order:
                
                sys.stdout.flush()
                print(f"{self.name} bought {each_p}")
                >>>> file: marketplace.py



import unittest
from threading import Lock
from time import gmtime



import logging
from logging.handlers import RotatingFileHandler

from tema.producer import Producer
from tema.consumer import Consumer
from tema.product import Coffee, Tea



logging.basicConfig(handlers=[
    RotatingFileHandler('marketplace.log', maxBytes=100000, backupCount=10)
],
                    level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = gmtime
logger = logging.getLogger()


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        logger.info(
            'The marketplace %s with maximum queue of %s is initializing...',
            self, queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer
        self.product_to_producer = {} 
        self.producer_to_products = {} 
        self.carts = {} 
        self.cart_counter = 0
        self.producer_counter = 0

        self.lock_register = Lock()
        self.lock_maximum_elements = Lock()
        self.lock_cart_size = Lock()
        self.lock_remove_from = Lock()

        logger.info('Initialization ended successfully!')

    def register_producer(self):
        

        logger.info('Starting producer registration by %s...', self)
        with self.lock_register:
            
            self.producer_to_products[self.producer_counter] = []
            self.producer_counter += 1
            logger.info('Producer with id: %d was created!',
                        self.producer_counter - 1)
            return self.producer_counter - 1

    def publish(self, producer_id, product):
        

        logger.info("Providing product %s by producer %s to marketplace %s...",
                    product, producer_id, self)
        with self.lock_maximum_elements:
            
            if len(self.producer_to_products[producer_id]) \
                >= self.queue_size_per_producer:
                logger.info('Providing product failed!')
                return False
            self.producer_to_products[producer_id].append(product)
            self.product_to_producer[product] = producer_id
        logger.info('Providing product ended successfully!')
        return True

    def new_cart(self):
        

        with self.lock_cart_size:
            
            logger.info('Creating new cart by marketplace %s...', self)
            self.carts[self.cart_counter] = []
            self.cart_counter += 1
            logger.info('A new cart with id %d was created!',
                        self.cart_counter - 1)
        return self.cart_counter - 1

    def add_to_cart(self, cart_id, product):
        

        logger.info('Adding product %s in the cart %s using marketplace %s...',
                    product, cart_id, self)
        
        all_producers = self.producer_to_products.keys()
        for producer in all_producers:
            number_of_products = \
                self.producer_to_products[producer].count(product)
            
            if number_of_products > 0:
                
                self.carts[cart_id].append(product)
                self.producer_to_products[producer].remove(product)
                logger.info('Adding a new product ended successfully!')
                return True
        logger.info('Adding a new product failed!')
        return False

    def remove_from_cart(self, cart_id, product):
        

        logger.info(
            'Removing product %s from the cart %s using marketplace %s...',
            product, cart_id, self)
        
        producer = self.product_to_producer[product]
        with self.lock_remove_from:
            self.carts[cart_id].remove(product)
            self.producer_to_products[producer].append(product)
        logger.info('Removing a new product ended successfully!')

    def place_order(self, cart_id):
        

        logger.info('Placing a new order from cart %s using marketplace %s',
                    cart_id, self)
        
        final_order = self.carts.pop(cart_id, None)
        logger.info('The order %s was provided!', final_order)
        return final_order


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        
        self.size_marketplace = 2

        self.marketplace = Marketplace(self.size_marketplace)
        self.consumer0 = Consumer(carts=[],
                                  marketplace=self.marketplace,
                                  retry_wait_time=100,
                                  kwargs=dict({"name": "consumer0"}))
        self.consumer1 = Consumer(carts=[],
                                  marketplace=self.marketplace,
                                  retry_wait_time=250,
                                  kwargs=dict({"name": "consumer1"}))
        self.product0 = Coffee('Arabica', 12, 6, 'MEDIUM')
        self.product1 = Coffee('Cappucino', 10, 12, 'LOW')
        self.product2 = Tea('Complex', 9, 'White')
        self.product3 = Tea('Honey tea', 11, 'Sweet')
        self.producer0 = Producer(products=[],
                                  marketplace=self.marketplace,
                                  republish_wait_time=120,
                                  kwargs={})
        self.producer1 = Producer(products=[],
                                  marketplace=self.marketplace,
                                  republish_wait_time=176,
                                  kwargs={})

    def test___init__(self):
        
        self.assertEqual(self.marketplace.queue_size_per_producer,
                         self.size_marketplace)
        self.assertEqual(self.marketplace.cart_counter, 0)
        self.assertEqual(self.marketplace.producer_counter, 2)
        self.assertEqual(self.producer0.id_producer, 0)
        self.assertEqual(self.producer1.id_producer, 1)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.producer_counter, 3)

    def test_publish(self):
        


        self.assertEqual(self.marketplace.publish(0, self.product1), True)
        self.assertEqual(self.marketplace.publish(0, self.product2), True)
        self.assertEqual(self.marketplace.publish(0, self.product3), False)
        self.assertEqual(self.marketplace.publish(1, self.product0), True)
        self.assertEqual(self.marketplace.publish(1, self.product3), True)

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.publish(0, self.product1)
        self.marketplace.publish(0, self.product2)
        self.marketplace.publish(1, self.product0)
        self.marketplace.publish(1, self.product3)

        self.assertTrue(self.marketplace.add_to_cart(0, self.product0))
        self.assertFalse(self.marketplace.add_to_cart(1, self.product0))
        self.assertTrue(self.marketplace.add_to_cart(0, self.product1))
        self.assertTrue(self.marketplace.add_to_cart(0, self.product3))
        self.assertTrue(self.marketplace.add_to_cart(1, self.product2))

    def test_remove_from_cart(self):
        
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.publish(0, self.product1)
        self.marketplace.publish(0, self.product2)
        self.marketplace.publish(1, self.product0)
        self.marketplace.publish(1, self.product3)
        self.marketplace.add_to_cart(0, self.product0)
        self.marketplace.add_to_cart(0, self.product1)
        self.marketplace.add_to_cart(0, self.product3)
        self.marketplace.add_to_cart(1, self.product2)

        self.marketplace.remove_from_cart(1, self.product2)
        self.assertFalse(self.product2 in self.marketplace.carts[1])

        self.marketplace.remove_from_cart(0, self.product3)
        self.assertFalse(self.product3 in self.marketplace.carts[0])

    def test_place_order(self):
        
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.publish(0, self.product1)
        self.marketplace.publish(0, self.product2)
        self.marketplace.publish(1, self.product0)
        self.marketplace.publish(1, self.product3)
        self.marketplace.add_to_cart(0, self.product0)
        self.marketplace.add_to_cart(0, self.product1)
        self.marketplace.add_to_cart(0, self.product3)
        self.marketplace.add_to_cart(1, self.product2)
        self.marketplace.remove_from_cart(1, self.product2)
        self.marketplace.remove_from_cart(0, self.product3)

        self.assertEqual(self.marketplace.place_order(0),
                         [self.product0, self.product1])
        self.assertEqual(len(self.marketplace.place_order(1)), 0)
        self.assertEqual(len(self.marketplace.place_order(2)), 0)


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_producer = marketplace.register_producer()

    def run(self):
        products = self.products
        while True:
            for product in products:
                contor = 0
                while contor < product[1]:
                    
                    if self.marketplace.publish(self.id_producer, product[0]):
                        time.sleep(product[2])
                        contor = contor + 1
                    
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
