


from threading import Thread, Lock
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace


        self.retry_wait_time = retry_wait_time
        self.cart_id = 0

    def run(self):
        for cart in self.carts:
            lock = Lock()
            lock.acquire()
            self.cart_id = self.marketplace.new_cart()
            lock.release()

            for ops in cart:
                type_operation = ops['type']
                product = ops['product']
                quantity = ops['quantity']
                i = 0

                if type_operation == "add":
                    while i < quantity:
                        status = self.marketplace.add_to_cart(self.cart_id, product)
                        if not status:
                            time.sleep(self.retry_wait_time)
                        else:
                            i += 1
                else:
                    while i < quantity:
                        self.marketplace.remove_from_cart(self.cart_id, product)
                        i += 1

            placed_order_cart = self.marketplace.place_order(self.cart_id)

            lock = Lock()
            for product_bought in placed_order_cart:
                lock.acquire()
                print("{} bought {}".format(self.name, product_bought))
                lock.release()

import logging
from logging.handlers import RotatingFileHandler
import time
import unittest
from dataclasses import dataclass

class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.count_producers = 0  
        self.carts = []  
        self.producer_products = []  
        self.reserved_products = []  
        logger = logging.getLogger('my_logger') 
        logger.setLevel(logging.INFO) 
        handler = RotatingFileHandler('marketplace.log', maxBytes=2000, backupCount=10)
        formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
        handler.setFormatter(formatter)
        logging.Formatter.converter = time.gmtime
        logger.addHandler(handler)

        logger.info("Marketplace created")

    def register_producer(self):
        
        logger = logging.getLogger('my_logger')
        logger.info("Producer registration started")

        self.producer_products.append([])
        self.reserved_products.append([])
        self.count_producers = self.count_producers + 1

        logger.info("Producer registration finished")
        return self.count_producers - 1

    def publish(self, producer_id, product):
        

        logger = logging.getLogger('my_logger')
        logger.info("Product publishing started")

        if len(self.producer_products[producer_id]) < self.queue_size_per_producer:
            self.producer_products[producer_id].append(product)

            logger.info("Product publishing finished successfully")
            return True

        logger.info("Product publishing: Caller should wait")
        return False

    def new_cart(self):
        

        logger = logging.getLogger('my_logger')
        logger.info("Cart creation started")

        self.carts.append([])
        logger.info("Cart creation finished")
        return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        
        
        

        logger = logging.getLogger('my_logger')
        logger.info("Product adding in cart started")

        for i in range(self.count_producers):


            if product in self.producer_products[i]:
                self.carts[cart_id].append(product)
                self.reserved_products[i].append(product)
                self.producer_products[i].remove(product)
                return True

        logger.info("Product added in cart successfully")
        return False

    def remove_from_cart(self, cart_id, product):
        

        logger = logging.getLogger('my_logger')
        logger.info("Product removing started")



        self.carts[cart_id].remove(product)

        
        for i in range(self.count_producers):
            if product in self.reserved_products[i]:
                self.reserved_products[i].remove(product)
                self.producer_products[i].append(product)
                return True

        logger.info("Product removing finished")
        return False

    def place_order(self, cart_id):
        

        logger = logging.getLogger('my_logger')
        logger.info("Order placing finished successfully")
        return self.carts[cart_id]


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        logging.disable(logging.CRITICAL)
        self.marketplace = Marketplace(10)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 0,
                         'wrong producer id')
        self.assertEqual(len(self.marketplace.producer_products), 1,
                         'wrong producer products size')

    def test_publish(self):
        
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.assertTrue(self.marketplace.publish(0, product),
                        'product not published')
        self.assertEqual(len(self.marketplace.producer_products[0]), 1,
                         'wrong producer products size')

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0,
                         'wrong cart id')
        self.assertEqual(len(self.marketplace.carts), 1,
                         'wrong carts size')

    def test_add_to_cart(self):
        
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product),
                        'product not added to cart')
        self.assertEqual(len(self.marketplace.carts[0]), 1,
                         'wrong cart size')
        self.assertEqual(len(self.marketplace.producer_products[0]), 0,
                         'wrong producer products size')

    def test_remove_from_cart(self):
        
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.assertTrue(self.marketplace.remove_from_cart(cart_id, product),
                        'product not removed from cart')
        self.assertEqual(len(self.marketplace.carts[0]), 0,
                         'wrong cart size')
        self.assertEqual(len(self.marketplace.producer_products[0]), 1,
                         'wrong producer products size')

    def test_place_order(self):
        
        self.marketplace.register_producer()
        product = Product('prod1', 10)
        self.marketplace.publish(0, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.place_order(cart_id), [product],
                         'wrong order')


from threading import Thread, Lock
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)


        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = 0

    def run(self):
        lock = Lock()
        lock.acquire()
        self.producer_id = self.marketplace.register_producer()
        lock.release()

        while True:
            for product in self.products:
                product_id = product[0]
                quantity = product[1]
                waiting_time = product[2]
                i = 0

                while i < quantity:
                    status = self.marketplace.publish(self.producer_id, product_id)
                    if not status:
                        time.sleep(self.republish_wait_time)
                    else:
                        i += 1
                        time.sleep(waiting_time)
