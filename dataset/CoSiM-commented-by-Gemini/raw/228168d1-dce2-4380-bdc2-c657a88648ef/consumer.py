


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.carts = carts

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for command in cart:
                if command["type"] == "add":
                    for _ in range(command["quantity"]):
                        while not self.marketplace.add_to_cart(cart_id, command["product"]):
                            sleep(self.retry_wait_time)
                elif command["type"] == "remove":
                    for _ in range(command["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, command["product"])

            items = self.marketplace.place_order(cart_id)
            for item in items:
                with self.marketplace.print_lock:
                    print(self.name, "bought", item[0])

import time
import unittest
from threading import Lock
from tema.product import *
import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    handlers=[RotatingFileHandler('tema/marketplace.log', maxBytes=1000, backupCount=10)],
    format='%(asctime)s - %(levelname)s - %(message)s',


    level=logging.INFO,
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.Formatter.converter = time.gmtime


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue = []
        self.queue_size_per_producer = queue_size_per_producer
        self.carts = []
        self.producer_ids = 0
        self.cart_ids = 0

        self.producer_lock = Lock()
        self.cart_lock = Lock()
        self.print_lock = Lock()
        self.queue_lock = Lock()

    def register_producer(self):
        
        logging.info('Method called: register_producer')
        self.producer_lock.acquire()

        curr_id = self.producer_ids
        self.queue.append([])
        self.producer_ids += 1

        self.producer_lock.release()
        logging.info('Method register_producer returned ' + str(curr_id))
        return curr_id

    def publish(self, producer_id, product):
        
        logging.info('Method called: publish; params: producer_id=' + str(producer_id) + ' product=' + str(product))



        if len(self.queue[producer_id]) >= self.queue_size_per_producer:
            logging.info('Method publish returned False')
            return False

        self.queue[producer_id].append(product)
        logging.info('Method publish returned True')
        return True

    def new_cart(self):
        
        logging.info('Method called: new_cart')
        self.cart_lock.acquire()

        cart_id = self.cart_ids
        self.carts.append([])
        self.cart_ids += 1

        self.cart_lock.release()
        logging.info('Method new_cart returned ' + str(cart_id))
        return cart_id

    def find_product(self, product):
        
        logging.info('Method called: find_product; params: product=' + str(product))
        for producer_idx, producer_queue in enumerate(self.queue):
            for item_idx, item in enumerate(producer_queue):
                if item == product:
                    logging.info('Method find_product returned ' + str(producer_idx) + ', ' + str(item_idx))
                    return producer_idx, item_idx

        logging.info('Method find_product returned -1, -1')
        return -1, -1

    def add_to_cart(self, cart_id, product):
        
        logging.info('Method called: add_to_cart; params: cart_id=' + str(cart_id) + ' product=' + str(product))

        self.queue_lock.acquire()

        producer_idx, item_idx = self.find_product(product)

        if producer_idx == -1 and item_idx == -1:
            self.queue_lock.release()
            logging.info('Method add_to_cart returned False')
            return False

        self.carts[cart_id].append(
            [self.queue[producer_idx].pop(item_idx), producer_idx]
        )



        self.queue_lock.release()
        logging.info('Method add_to_cart returned True')
        return True

    def remove_from_cart(self, cart_id, product):
        
        logging.info('Method called: remove_from_cart; params: cart_id=' + str(cart_id) + ' product=' + str(product))

        for idx, prod in enumerate(self.carts[cart_id]):
            if prod[0] == product:
                self.publish(prod[1], product)


                self.carts[cart_id].pop(idx)
                break
        logging.info('Method remove_from_cart returned void')

    def place_order(self, cart_id):
        
        logging.info('Method called: place_order; params: cart_id=' + str(cart_id))
        logging.info('Method place_order returned ' + str(self.carts[cart_id]))
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    def setUp(self):
        self.marketplace = Marketplace(15)
        self.marketplace.queue = [
            [Tea(name='Linden', price=9, type='Herbal'), Tea(name='Linden', price=9, type='Herbal'),
             Tea(name='Test', price=9, type='Herbal'), Tea(name='Linden', price=9, type='Herbal'),
             Tea(name='Linden', price=9, type='Herbal'),
             Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')]]
        self.marketplace.carts = [[[Tea(name='Test1', price=9, type='Herbal'), 0],
                                   [Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM'), 0],
                                   [Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM'), 0]]]

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), 0)

    def test_publish(self):
        prod_id = self.marketplace.register_producer()
        prod = Coffee(name='Test_Publish', price=1, acidity=5.05, roast_level='MEDIUM')
        self.marketplace.publish(prod_id, prod)
        self.assertTrue(prod in self.marketplace.queue[prod_id])

    def test_new_cart(self):
        first_length = len(self.marketplace.carts)
        self.marketplace.new_cart()
        self.assertEqual(first_length + 1, len(self.marketplace.carts))

    def test_find_product(self):
        prod = Tea(name='Test', price=9, type='Herbal')
        prod_idx, item_idx = self.marketplace.find_product(prod)
        self.assertEqual(prod_idx, 0)
        self.assertEqual(item_idx, 2)

    def test_add_to_cart(self):
        first_len = len(self.marketplace.queue[0])
        prod = Tea(name='Test', price=9, type='Herbal')
        self.marketplace.add_to_cart(0, prod)
        self.assertEqual(len(self.marketplace.queue[0]), first_len - 1)
        prod_idx, item_idx = self.marketplace.find_product(prod)
        self.assertEqual(prod_idx, -1)
        self.assertEqual(item_idx, -1)

    def test_remove_from_cart(self):
        prod = Tea(name='Test1', price=9, type='Herbal')
        first_len = len(self.marketplace.carts[0])
        self.marketplace.remove_from_cart(0, prod)
        self.assertEqual(len(self.marketplace.carts[0]), first_len - 1)

    def test_place_order(self):
        self.assertEqual(self.marketplace.place_order(0), self.marketplace.carts[0])


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.id = None
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        self.id = self.marketplace.register_producer()

        
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    while not self.marketplace.publish(self.id, product[0]):
                        sleep(self.republish_wait_time)

                    sleep(product[2])


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
