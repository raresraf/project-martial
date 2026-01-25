


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        
        
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()
            
            for operation in cart:
                
                if operation["type"] == "add":
                    for _ in range(operation["quantity"]):
                        while not self.marketplace.add_to_cart(cart_id, operation["product"]):
                            
                            sleep(self.retry_wait_time)
                elif operation["type"] == "remove":
                    
                    for _ in range(operation["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
            
            order = self.marketplace.place_order(cart_id)
            
            for product in order:
                print("{0} bought {1}".format(self.name, product))

import time
from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea


class Marketplace:
    
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        
        self.producers_queue = {}
        
        self.carts = {}
        
        self.producer_id = 0
        
        self.cart_id = 0
        
        self.producer_id_lock = Lock()
        
        self.cart_id_lock = Lock()
        
        
        
        self.producers_locks = {}
        
        
        self.products_producers = {}
        
        
        
        
        
        self.products_locks = {}
        
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.INFO)
        self.handler = RotatingFileHandler("marketplace.log", maxBytes=1024 * 512, backupCount=20)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.formatter.converter = time.gmtime
        self.logger.addHandler(self.handler)

    def register_producer(self):
        
        self.logger.info("Entered register_producer()!")
        
        self.producer_id_lock.acquire()
        
        producer_id_string = "prod{0}".format(self.producer_id)
        
        self.producers_queue[producer_id_string] = 0
        
        self.producers_locks[producer_id_string] = Lock()
        
        self.producer_id += 1
        
        self.producer_id_lock.release()
        self.logger.info("Finished register_producer(): returned producer_id: %s!",
                         producer_id_string)
        return producer_id_string

    def publish(self, producer_id, product):
        
        self.logger.info("Entered publish(%s, %s)!", producer_id, product)
        
        self.producers_locks[producer_id].acquire()
        
        queue_size = self.producers_queue[producer_id]
        
        if queue_size == self.queue_size_per_producer:
            
            self.producers_locks[producer_id].release()
            self.logger.info("Finished publish(%s, %s): Queue is Full!",
                             producer_id, product)
            return False
        
        if product not in self.products_producers:
            self.products_locks[product] = Lock()
            self.products_locks[product].acquire()
            self.products_producers[product] = []
        else:
            self.products_locks[product].acquire()
        self.products_producers[product].append(producer_id)
        self.products_locks[product].release()
        
        self.producers_queue[producer_id] += 1
        


        self.producers_locks[producer_id].release()
        self.logger.info("Finished publish(%s, %s): Published product!",
                         producer_id, product)
        return True

    def new_cart(self):
        
        self.logger.info("Entered new_cart()!")
        
        self.cart_id_lock.acquire()
        cart_id = self.cart_id
        
        self.carts[cart_id] = []
        
        self.cart_id += 1
        
        self.cart_id_lock.release()
        self.logger.info("Finished new_cart(): New cart: %d!", cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        self.logger.info("Entered add_to_cart(%d, %s)!", cart_id, product)
        
        if cart_id not in self.carts:
            self.logger.info("Finished add_to_cart(%d, %s): Cart doesn't exist!",
                             cart_id, product)
            return False
        
        if product not in self.products_producers:
            self.logger.info("Finished add_to_cart(%d, %s): Product is not available!",
                             cart_id, product)
            return False
        self.products_locks[product].acquire()
        if not self.products_producers[product]:
            self.products_locks[product].release()
            self.logger.info("Finished add_to_cart(%d, %s): Product is not available!",
                             cart_id, product)
            return False
        
        
        producer_id = self.products_producers[product].pop(0)
        self.products_locks[product].release()
        
        
        
        self.carts[cart_id].append({"product": product,
                                    "producer_id": producer_id})
        self.logger.info("Finished add_to_cart(%d, %s): Product added to cart!",
                         cart_id, product)
        return True

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("Entered remove_from_cart(%d, %s)!", cart_id, product)
        
        if cart_id not in self.carts:
            self.logger.info("Finished remove_from_cart(%d, %s): Cart doesn't exist!",
                             cart_id, product)
            return False
        
        cart_list = self.carts[cart_id]
        
        for cart_element in cart_list:
            if cart_element["product"] == product:
                
                producer_id = cart_element["producer_id"]
                self.products_producers[product].append(producer_id)
                
                self.carts[cart_id].remove(cart_element)
                self.logger.info("Finished remove_from_cart(%d, %s): Product removed from cart!",
                                 cart_id, product)
                return True
        self.logger.info("Finished remove_from_cart(%d, %s): Product not found in cart!",
                         cart_id, product)
        return False

    def place_order(self, cart_id):
        
        self.logger.info("Entered place_order(%d)!", cart_id)
        result = []
        
        if cart_id not in self.carts:
            self.logger.info("Finished place_order(%d): Cart doesn't exist!", cart_id)
            return None
        
        cart_list = self.carts[cart_id]
        
        for cart_element in cart_list:
            product = cart_element["product"]
            result.append(product)
            producer_id = cart_element["producer_id"]
            
            self.producers_locks[producer_id].acquire()
            self.producers_queue[producer_id] -= 1
            self.producers_locks[producer_id].release()
        
        self.carts[cart_id] = []
        self.logger.info("Finished place_order(%d): Order placed: %s!", cart_id, result)
        return result


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        
        self.marketplace = Marketplace(5)
        self.product0 = Coffee(name="Indonezia", acidity="5.05", roast_level="MEDIUM", price=1)
        self.product1 = Tea(name="Linden", type="Herbal", price=9)
        self.product2 = Coffee(name="Ethiopia", acidity="5.09", roast_level="MEDIUM", price=10)
        self.product3 = Coffee(name="Arabica", acidity="5.02", roast_level="MEDIUM", price=9)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 'prod0',
                         'Incorrect producer_id assigned for first producer!')
        self.assertEqual(self.marketplace.register_producer(), 'prod1',
                         'Incorrect producer_id assigned for second producer!')
        self.assertEqual(self.marketplace.register_producer(), 'prod2',
                         'Incorrect producer_id assigned for third producer!')

    def test_publish(self):
        
        self.test_register_producer()
        
        for _ in range(3):
            check = self.marketplace.publish('prod0', self.product0)
            self.assertTrue(check, 'Producer prod0 should be able to publish product!')
        
        for _ in range(2):
            check = self.marketplace.publish('prod0', self.product1)
            self.assertTrue(check, 'Producer prod0 should be able to publish product!')
        
        check = self.marketplace.publish('prod0', self.product0)
        self.assertFalse(check, 'Producer prod0 should not be able to publish product!')
        
        for _ in range(2):
            check = self.marketplace.publish('prod1', self.product2)
            self.assertTrue(check, 'Producer prod1 should be able to publish product!')
        
        check = self.marketplace.publish('prod1', self.product3)
        self.assertTrue(check, 'Producer prod1 should be able to publish product!')
        
        check = self.marketplace.publish('prod1', self.product1)
        self.assertTrue(check, 'Producer prod1 should be able to publish product!')
        
        self.assertEqual(self.marketplace.producers_queue['prod0'], 5,
                         'Producer prod0 queue should be full!')
        self.assertEqual(self.marketplace.producers_queue['prod1'], 4,
                         'Producer prod1 queue size should be 4!')
        
        self.assertEqual(len(self.marketplace.products_producers[self.product0]), 3,
                         'Product0 should be available in quantity = 3!')
        self.assertEqual(len(self.marketplace.products_producers[self.product1]), 3,
                         'Product1 should be available in quantity = 3!')
        self.assertEqual(len(self.marketplace.products_producers[self.product2]), 2,
                         'Product2 should be available in quantity = 2!')
        self.assertEqual(len(self.marketplace.products_producers[self.product3]), 1,
                         'Product3 should be available in quantity = 1!')

    def test_new_cart(self):
        
        self.test_publish()
        
        self.assertEqual(self.marketplace.new_cart(), 0,
                         'Incorrect cart_id assigned for first cart!')
        self.assertEqual(self.marketplace.new_cart(), 1,
                         'Incorrect cart_id assigned for second cart!')
        self.assertEqual(self.marketplace.new_cart(), 2,
                         'Incorrect cart_id assigned for third cart!')
        self.assertEqual(self.marketplace.new_cart(), 3,
                         'Incorrect cart_id assigned for fourth cart!')
        
        for i in range(4):
            self.assertEqual(self.marketplace.carts[i], [],
                             'Cart should be empty!')

    def test_add_to_cart(self):
        
        self.test_new_cart()
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.product0),
                        'Cannot add product0 to cart!')
        
        for _ in range(3):
            self.assertTrue(self.marketplace.add_to_cart(0, self.product1),
                            'Cannot add product1 to cart!')
        
        
        self.assertFalse(self.marketplace.add_to_cart(0, self.product1),
                         'Should not be able to add product1 to cart!')
        
        self.assertTrue(self.marketplace.add_to_cart(0, self.product2),
                        'Cannot add product2 to cart!')
        
        self.assertEqual(len(self.marketplace.carts[0]), 5,
                         'Wrong number of products added to cart!')
        
        self.assertTrue(self.marketplace.add_to_cart(1, self.product2),
                        'Cannot add product2 to cart!')
        
        
        self.assertFalse(self.marketplace.add_to_cart(1, self.product2),
                         'Should not be able to add product2 to cart!')
        
        self.assertEqual(len(self.marketplace.carts[1]), 1,
                         'Wrong number of products added to cart!')
        
        self.assertEqual(len(self.marketplace.products_producers[self.product0]), 2,
                         'Product0 should be available in quantity = 0!')
        self.assertEqual(len(self.marketplace.products_producers[self.product1]), 0,
                         'Product1 should be available in quantity = 3!')
        self.assertEqual(len(self.marketplace.products_producers[self.product2]), 0,
                         'Product2 should be available in quantity = 2!')
        self.assertEqual(len(self.marketplace.products_producers[self.product3]), 1,
                         'Product3 should be available in quantity = 1!')

    def test_remove_from_cart(self):
        
        self.test_add_to_cart()
        
        self.assertTrue(self.marketplace.remove_from_cart(0, self.product1),
                        'Cannot remove product1 from cart!')
        
        self.assertEqual(len(self.marketplace.carts[0]), 4,
                         'Wrong number of products in cart0!')
        
        
        self.assertFalse(self.marketplace.remove_from_cart(0, self.product3),
                         'Should not be able to remove this product!')
        
        self.assertEqual(len(self.marketplace.products_producers[self.product1]), 1,
                         'Product1 should be available in quantity = 1!')

    def test_place_order(self):
        
        self.test_remove_from_cart()
        
        self.assertEqual(self.marketplace.place_order(0),
                         [self.product0, self.product1, self.product1, self.product2],
                         'Wrong cart list!')
        
        self.assertEqual(self.marketplace.producers_queue['prod0'], 3,
                         'Producer prod0 queue contain 3 products!')
        self.assertEqual(self.marketplace.producers_queue['prod1'], 2,
                         'Producer prod1 queue should contain 2 products!')
        
        self.assertEqual(self.marketplace.carts[0], [],
                         'Cart0 should be empty!')


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        
        Thread.__init__(self, daemon=True)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        
        
        producer_id = self.marketplace.register_producer()
        
        while True:
            
            for element in self.products:
                
                product = element[0]
                quantity = element[1]
                production_time = element[2]
                
                sleep(production_time)
                
                for _ in range(quantity):
                    while not self.marketplace.publish(producer_id, product):
                        
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
