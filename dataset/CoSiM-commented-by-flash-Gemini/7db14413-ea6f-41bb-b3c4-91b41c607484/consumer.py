


from time import sleep
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, name=kwargs['name'])
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                if operation['type'] == 'add':
                    for _ in range(operation['quantity']):
                        while not self.marketplace.add_to_cart(
                                cart_id, operation['product']):
                            sleep(self.retry_wait_time)

                if operation['type'] == 'remove':
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(
                            cart_id, operation['product'])

            products = self.marketplace.place_order(cart_id)
            for prod in products:
                with self.marketplace.printing_lock:
                    print(f'{self.name} bought {prod}')

from uuid import uuid4
import unittest


import logging
from logging.handlers import RotatingFileHandler

from threading import Lock
import time

from .product import Product


def logger_set_up():
    
    logging.basicConfig(
        handlers=[RotatingFileHandler(
            'marketplace.log', maxBytes=10000, backupCount=10)],
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')

    logging.Formatter.converter = time.gmtime


class Marketplace:
    

    def __init__(self, queue_size_per_producer: int):
        
        logger_set_up()
        self.queue_size_per_producer = queue_size_per_producer

        self.producers_queues: dict[str, list[Product]] = {}
        self.available_products: dict[Product, int] = {}
        self.carts: dict[int, list] = {}
        self.customer_lock = Lock()
        self.producer_lock = Lock()
        self.printing_lock = Lock()

    def register_producer(self):
        
        logging.info('register producer started.')
        with self.producer_lock:
            p_id = uuid4().hex
            self.producers_queues[p_id] = []
        logging.info('register producer finished. Returned %s.', p_id)
        return p_id



    def publish(self, producer_id: str, product: Product):
        

        logging.info(
            'publish started. Parameters: producer_id = %s, product = %s.', producer_id, product)
        

        if len(self.producers_queues[producer_id]) == self.queue_size_per_producer:
            logging.info('publish finished. Returned False.')
            return False

        
        self.producers_queues[producer_id].append(product)

        
        if product not in self.available_products:
            self.available_products[product] = 1
        else:
            self.available_products[product] += 1

        logging.info('publish finished. Returned True.')
        return True

    def new_cart(self):
        
        logging.info('new_cart started.')
        with self.customer_lock:
            cart_id = uuid4().int
            self.carts[cart_id] = []
        logging.info('new_cart finished. Returned %i', cart_id)


        return cart_id

    def add_to_cart(self, cart_id: int, product: Product):
        
        logging.info(
            'add_to_cart started. Parameters: cart_id=%i, product=%s.', cart_id, product)

        with self.customer_lock:

            
            if product not in self.available_products or self.available_products[product] == 0:
                logging.info('add_to_cart finished. Returned False.')
                return False

            
            self.available_products[product] -= 1

            
            self.carts[cart_id].append(product)

        logging.info('add_to_cart finished. Returned True.')
        logging.debug('added')


        return True

    def remove_from_cart(self, cart_id: int, product: Product):
        
        logging.info(
            'remove_from_cart started. Parameters: cart_id=%i, product=%s.', cart_id, product)
        with self.customer_lock:
            
            self.carts[cart_id].remove(product)

            
            self.available_products[product] += 1
        logging.info('remove_from_cart finished.')



    def place_order(self, cart_id: int):
        
        logging.info('place_order started. Parameters: cart_id=%i.', cart_id)
        
        bought_items = []

        for product in self.carts[cart_id]:
            for producer_queue in self.producers_queues.values():
                if product in producer_queue:
                    bought_items.append(product)
                    producer_queue.remove(product)
                    break

        logging.info('place_order finished. Returned %s.', bought_items)
        return bought_items


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        
        self.marketplace = Marketplace(1)

    def test_register_producer_return_str(self):
        
        p_id = self.marketplace.register_producer()
        self.assertEqual(type(p_id), str)

    def test_new_cart_return_int(self):
        
        c_id = self.marketplace.new_cart()
        self.assertEqual(type(c_id), int)

    def test_publish_if_queue_not_full_then_return_true(self):
        
        p_id = self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(p_id, Product('Tea', 11)),
                         True)

    def test_publish_if_queue_full_then_return_true(self):
        
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11))
        self.assertEqual(self.marketplace.publish(
            p_id, Product('Tea', 11)), False)

    def test_add_to_cart_if_product_not_available_return_false(self):
        
        c_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(
            c_id, Product('Tea', 11)), False)

    def test_add_to_cart_if_product_available_return_true(self):
        
        c_id = self.marketplace.new_cart()
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11))

        self.assertEqual(self.marketplace.add_to_cart(
            c_id, Product('Tea', 11)), True)

    def test_remove_from_cart(self):
        
        c_id = self.marketplace.new_cart()
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11))
        self.marketplace.add_to_cart(
            c_id, Product('Tea', 11))
        self.marketplace.remove_from_cart(c_id, Product('Tea', 11))
        self.assertEqual(len(self.marketplace.carts[c_id]), 0)

    def test_place_order(self):
        
        p_id = self.marketplace.register_producer()
        c_id = self.marketplace.new_cart()
        self.marketplace.publish(p_id, Product('Tea', 11))
        self.marketplace.add_to_cart(
            c_id, Product('Tea', 11))
        self.marketplace.place_order(c_id)
        self.assertEqual(len(self.marketplace.producers_queues[p_id]), 0)


from time import sleep
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.p_id = ''

    def run(self):
        self.p_id = self.marketplace.register_producer()
        while True:
            for prod_info in self.products:
                for _ in range(prod_info[1]):
                    sleep(prod_info[2])
                    while not self.marketplace.publish(self.p_id, prod_info[0]):
                        sleep(self.republish_wait_time)
