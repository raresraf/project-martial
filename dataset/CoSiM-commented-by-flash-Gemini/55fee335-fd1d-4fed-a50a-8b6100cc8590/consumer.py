

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        cart_id = self.marketplace.new_cart()

        for i in self.carts:
            for j in i:
                quantity = j['quantity']
                product = j['product']
                action = j['type']



                for _ in range(0, quantity):
                    if action == 'add':
                        added = self.marketplace.add_to_cart(cart_id, product)

                        
                        while not added:
                            time.sleep(self.retry_wait_time)
                            added = self.marketplace.add_to_cart(cart_id, product)

                    elif action == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)

            self.marketplace.place_order(cart_id)

import logging
import time
from logging.handlers import RotatingFileHandler


class Logger:
    
    MAX_BYTE_COUNT = 1000000
    BACKUP_COUNT = 5

    @staticmethod
    def create_logger(name, log_file):
        
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - '
                                      '%(funcName)s - %(message)s')
        formatter.converter = time.gmtime

        
        handler = RotatingFileHandler(log_file,
                                      maxBytes=Logger.MAX_BYTE_COUNT,
                                      backupCount=Logger.BACKUP_COUNT)
        handler.setFormatter(formatter)

        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        return logger

import threading
from threading import Lock
import unittest
from tema.product import Tea
from tema.logger import Logger


class TestMarketplace(unittest.TestCase):
    

    def setUp(self) -> None:
        self.marketplace = Marketplace(1)

    def test_register_producer(self):
        
        self.assertGreaterEqual(int(self.marketplace.register_producer()), 0)

    def test_publish(self):
        

        
        self.assertEqual(self.marketplace.publish(123, None), False)

        
        producer_id = self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(producer_id, None), False)

        
        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.publish(producer_id, product), True)

        
        self.assertEqual(self.marketplace.publish(producer_id, product), False)

        
        
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.publish(producer_id, product), True)

    def test_new_cart(self):
        
        self.assertGreaterEqual(int(self.marketplace.new_cart()), 0)

    def test_add_to_cart(self):
        
        
        self.assertEqual(self.marketplace.publish(12345, None), False)

        
        cart_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(cart_id, None), False)

        
        
        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), False)

        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, product)


        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)
        self.marketplace.remove_from_cart(cart_id, product)

        
        snd_cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(snd_cart_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), False)

        
        self.marketplace.remove_from_cart(snd_cart_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)

        
        self.marketplace.place_order(cart_id)
        self.marketplace.publish(producer_id, product)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)
        self.assertEqual(len(self.marketplace.place_order(cart_id)), 1)

    def test_remove_from_cart(self):
        
        
        self.assertEqual(self.marketplace.remove_from_cart(123, None), False)

        
        cart_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, None), False)

        
        product = Tea("1", 2, "3")
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), False)

        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), True)

        
        self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), False)

    def test_place_order(self):
        
        
        self.assertEqual(self.marketplace.place_order(1234), None)

        
        producer_id = self.marketplace.register_producer()
        product = Tea("1", 2, "3")
        self.marketplace.publish(producer_id, product)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product)
        result = self.marketplace.place_order(cart_id)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 1)

        
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        result = self.marketplace.place_order(cart_id)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 1)

        
        self.assertEqual(self.marketplace.place_order(cart_id), [])


class Marketplace:
    
    LOG_FILE = 'marketplace.log'
    MAX_BYTE_COUNT = 1000000
    BACKUP_COUNT = 5

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.producers = {}
        self.carts = {}
        self.register_lock = Lock()
        self.cart_lock = Lock()
        self.order_lock = Lock()
        self.crt_assignable_producer_idx = 0
        self.crt_assignable_cart_idx = 0

        self.logger = Logger.create_logger(__name__, Marketplace.LOG_FILE)

    def register_producer(self):
        
        with self.register_lock:
            producer_id = str(self.crt_assignable_producer_idx)
            self.crt_assignable_producer_idx += 1

        
        self.producers[producer_id] = [Lock(), [], 0]

        self.logger.info("Registered a new producer with ID: %s.", producer_id)

        return producer_id

    def publish(self, producer_id, product):
        
        
        if producer_id not in self.producers:
            self.logger.error("Unregistered producer ID: %s.", producer_id)
            return False

        
        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Producer: %s is trying to publish: %s", producer_id, product)

        
        if self.producers[producer_id][2] == self.queue_size_per_producer:
            self.logger.info("Producer: %s failed to "
                             "publish: %s. List is full.", producer_id, product)
            return False

        
        self.producers[producer_id][1].append(product)

        
        with self.producers[producer_id][0]:
            self.producers[producer_id][2] += 1

        self.logger.info("Producer: %s successfully published: %s.", producer_id, product)

        return True

    def new_cart(self):
        
        with self.cart_lock:
            cart_id = self.crt_assignable_cart_idx
            self.crt_assignable_cart_idx += 1

        self.carts[cart_id] = []

        self.logger.info("Generated a new cart with ID: %s", cart_id)

        return cart_id

    def add_to_cart(self, cart_id, product):
        
        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return False

        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Consumer is trying to add "
                         "product: %s to cart: %s.", product, cart_id)

        for key, value in self.producers.items():
            try:
                with value[0]:
                    
                    idx = value[1].index(product)
                    
                    found_product = value[1].pop(idx)
            except ValueError:
                continue

            
            
            self.carts[cart_id].append((key, found_product))

            self.logger.info("Consumer has successfully "
                             "added product: %s to cart: %s", product, cart_id)

            return True

        self.logger.info("Consumer failed to add product: "
                         "%s to cart: %s. Product has not been found.", product, cart_id)

        return False

    def remove_from_cart(self, cart_id, product):
        

        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return False

        if product is None:
            self.logger.error("Received None value for product.")
            return False

        self.logger.info("Consumer is trying to "
                         "remove product: %s from cart: %s.", product, cart_id)

        for i in self.carts[cart_id]:
            if i[1] == product:
                
                self.producers[i[0]][1].append(i[1])

                
                self.carts[cart_id].remove(i)

                self.logger.info("Consumer has successfully "
                                 "removed product: %s from cart: %s.", product, cart_id)

                return True

        self.logger.info("Consumer failed to remove product: "
                         "%s from cart: %s. Product is not in cart.", product, cart_id)

        return False

    def place_order(self, cart_id):
        
        if cart_id not in self.carts:
            self.logger.error("Unregistered cart id: %s", cart_id)
            return None

        self.logger.info("Consumer is trying to "
                         "place order for contents of cart: %s", cart_id)

        
        result = []

        for i in self.carts[cart_id]:
            with self.producers[i[0]][0]:
                self.producers[i[0]][2] -= 1

            with self.order_lock:
                print(str(threading.current_thread().name) + " bought " + str(i[1]))

            result.append(i[1])

        
        self.carts[cart_id] = []

        self.logger.info("Consumer is has successfully "
                         "placed order for contents of cart: %s", cart_id)

        return result

from copy import copy
import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        producer_id = self.marketplace.register_producer()

        while True:
            for i in self.products:
                
                product_template = i[0]
                quantity = i[1]
                waiting_time = i[2]

                for _ in range(0, quantity):
                    product = copy(product_template)



                    is_published = self.marketplace.publish(producer_id, product)

                    while not is_published:
                        time.sleep(self.republish_wait_time)
                        is_published = self.marketplace.publish(producer_id, product)

                    time.sleep(waiting_time)


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
