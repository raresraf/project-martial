


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = 0
        self.is_added = True
        self.consumer_carts = []

    def run(self):

        for i in range(len(self.carts)):
            self.cart_id = self.marketplace.new_cart()
            self.consumer_carts.append((self.cart_id, self.carts[i]))



            for j in range(len(self.carts[i])):
                if self.carts[i][j]['type'] == 'add':
                    
                    for _ in range(self.carts[i][j]['quantity']):
                        cart_product = self.carts[i][j]['product']
                        self.is_added = self.marketplace.add_to_cart(self.cart_id, cart_product)
                        while not self.is_added:
                            
                            sleep(self.retry_wait_time)
                            self.is_added = self.marketplace.add_to_cart(self.cart_id, cart_product)


                else:
                    
                    for _ in range(self.carts[i][j]['quantity']):
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])

            
            order = self.marketplace.place_order(self.cart_id)
            
            for j in range(len(order)):
                print(self.name + " bought " + str(order[j][1]))

from time import gmtime
from threading import RLock
import logging
from logging.handlers import RotatingFileHandler

LOGGER = logging.getLogger("marketplace_logger")
LOGGER.setLevel(logging.ERROR)

HANDLER = RotatingFileHandler("marketplace.log", maxBytes=10000000, backupCount=10)

FORMATTER = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
FORMATTER.converter = gmtime
HANDLER.setFormatter(FORMATTER)

LOGGER.addHandler(HANDLER)

class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        self.producers_queue_size = {}
        self.producers_queue = []

        
        self.producers = {}

        
        self.carts_queue = {}

        self.register_lock = RLock()
        self.new_cart_lock = RLock()
        self.remove_prod_lock = RLock()
        self.remove_cart_lock = RLock()

    def register_producer(self):
        
        LOGGER.info("Enter register_producer")

        self.register_lock.acquire()
        self.producer_id += 1
        self.register_lock.release()

        
        self.producers_queue_size[self.producer_id] = 0
        self.producers[self.producer_id] = []

        LOGGER.info("Exit register_producer")
        return self.producer_id

    def publish(self, producer_id, product):
        
        LOGGER.info("Enter publish(%s, %s)", producer_id, product)

        
        if self.producers_queue_size[producer_id] >= self.queue_size_per_producer:
            return False

        self.producers_queue_size[producer_id] += 1

        
        self.producers_queue.append((producer_id, product))
        self.producers[producer_id].append(product)

        LOGGER.info("Exit publish(%s, %s)", producer_id, product)
        return True

    def new_cart(self):
        
        LOGGER.info("Enter new_cart")
        self.new_cart_lock.acquire()
        self.cart_id += 1
        self.new_cart_lock.release()

        
        self.carts_queue[self.cart_id] = []

        LOGGER.info("Exit new_cart")
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        LOGGER.info("Enter add_to_cart(%s, %s)", cart_id, product)

        self.remove_prod_lock.acquire()

        
        if len(self.producers_queue) == 0:
            self.remove_prod_lock.release()
            LOGGER.info("Exit add_to_cart")
            return False

        
        for i in range(len(self.producers_queue)):
            
            if self.producers_queue[i][1] == product:
                
                
                self.carts_queue[cart_id].append((self.producers_queue[i][0], product))

                
                self.producers_queue_size[self.producers_queue[i][0]] -= 1
                self.producers[self.producers_queue[i][0]].remove(product)
                self.producers_queue.remove((self.producers_queue[i][0], product))
                self.remove_prod_lock.release()
                LOGGER.info("Exit add_to_cart")
                return True

        self.remove_prod_lock.release()
        LOGGER.info("Exit add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        
        LOGGER.info("Enter remove_from_cart(%s, %s)", cart_id, product)

        self.remove_cart_lock.acquire()

        
        product_tuple = [product_tuple_item for product_tuple_item in self.carts_queue[cart_id] if
                         product_tuple_item[1] == product]
        producer_id = product_tuple[0][0]

        
        self.producers[producer_id].append(product)
        self.producers_queue_size[producer_id] += 1
        self.producers_queue.append((producer_id, product))

        
        self.carts_queue[cart_id].remove((producer_id, product))

        self.remove_cart_lock.release()
        LOGGER.info("Exit remove_from_cart")

    def place_order(self, cart_id):
        
        LOGGER.info("Enter place_order(%s)", cart_id)
        return self.carts_queue[cart_id]
import unittest
from marketplace import Marketplace
from product import Coffee, Tea

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.marketplace = Marketplace(10)

    def test_register_producer(self):
        self.marketplace.register_producer()

        self.assertEqual(self.marketplace.producer_id, 1)
        self.assertEqual(len(self.marketplace.producers_queue_size), 1)
        self.assertEqual(self.marketplace.producers_queue_size[self.marketplace.producer_id], 0)
        self.assertIsNotNone(self.marketplace.producers[self.marketplace.producer_id])
        self.assertEqual(len(self.marketplace.producers[self.marketplace.producer_id]), 0)

    def test_publish_returns_true(self):
        self.marketplace.register_producer()

        coffee = Coffee("Arabica", 10, "5.1", "medium")
        producer_id = 1

        publish_response = self.marketplace.publish(producer_id, coffee)
        self.assertTrue(publish_response)

        self.assertEqual(self.marketplace.producers_queue_size[producer_id], 1)

        self.assertEqual(len(self.marketplace.producers_queue), 1)
        self.assertEqual(self.marketplace.producers_queue[0][0], 1)
        self.assertEqual(self.marketplace.producers_queue[0][1], coffee)

        self.assertEqual(len(self.marketplace.producers[producer_id]), 1)
        self.assertEqual(self.marketplace.producers[producer_id][0], coffee)

    def test_publish_returns_false(self):
        self.marketplace.register_producer()

        coffee = Coffee("Arabica", 10, "5.1", "medium")
        producer_id = 1

        for i in range(10):
            publish_response = self.marketplace.publish(producer_id, coffee)
            self.assertTrue(publish_response)

        publish_response = self.marketplace.publish(producer_id, coffee)
        self.assertFalse(publish_response)

    def test_new_cart(self):
        self.marketplace.new_cart()
        self.assertEqual(self.marketplace.cart_id, 1)
        self.assertEqual(len(self.marketplace.carts_queue), 1)
        self.assertIsNotNone(self.marketplace.carts_queue[1])

    def test_add_to_cart_no_products(self):
        self.marketplace.new_cart()
        coffee = Coffee("Arabica", 10, "5.1", "medium")
        add_response = self.marketplace.add_to_cart(1, coffee)
        self.assertFalse(add_response)

    def test_add_to_cart_product_available(self):
        coffee = Coffee("Arabica", 10, "5.1", "medium")
        self.marketplace.register_producer()
        producer_id = 1
        self.marketplace.publish(producer_id, coffee)

        self.marketplace.new_cart()
        add_response = self.marketplace.add_to_cart(1, coffee)
        self.assertTrue(add_response)

    def test_add_to_cart_product_not_available(self):
        coffee = Coffee("Arabica", 10, "5.1", "medium")
        self.marketplace.register_producer()
        producer_id = 1
        self.marketplace.publish(producer_id, coffee)

        tea = Tea("Earl Grey", 10, "Green")

        self.marketplace.new_cart()
        add_response = self.marketplace.add_to_cart(1, tea)
        self.assertFalse(add_response)

    def test_remove_from_cart(self):
        coffee = Coffee("Arabica", 10, "5.1", "medium")
        self.marketplace.register_producer()
        producer_id = 1
        self.marketplace.publish(producer_id, coffee)

        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, coffee)

        self.marketplace.remove_from_cart(1, coffee)

        self.assertEqual(len(self.marketplace.carts_queue[1]), 0)
        self.assertEqual(self.marketplace.producers[producer_id][0], coffee)
        self.assertEqual(self.marketplace.producers_queue_size[producer_id], 1)
        self.assertEqual(self.marketplace.producers_queue[0], (producer_id, coffee))

    def test_place_order(self):
        coffee = Coffee("Arabica", 10, "5.1", "medium")
        self.marketplace.register_producer()
        producer_id = 1
        self.marketplace.publish(producer_id, coffee)

        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, coffee)

        order = self.marketplace.place_order(1)
        self.assertEqual(order[0], (producer_id, coffee))


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = 0
        self.is_published = True

    def run(self):
        self.producer_id = self.marketplace.register_producer()


        while 1:

            
            for i in range(len(self.products)):
                
                for _ in range(self.products[i][1]):

                    
                    self.is_published = self.marketplace.publish(self.producer_id, self.products[i][0])

                    while not self.is_published:
                        
                        sleep(self.republish_wait_time)
                        self.is_published = self.marketplace.publish(self.producer_id, self.products[i][0])

                    if self.is_published:
                        
                        sleep(self.products[i][2])


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
