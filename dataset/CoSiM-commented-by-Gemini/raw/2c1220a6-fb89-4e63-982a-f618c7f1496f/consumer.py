

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.carts = carts

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()  
            for item in cart:
                operation_type = item["type"]
                prod = item["product"]
                quantity = item["quantity"]
                i = 0
                while i < quantity:
                    if operation_type == "add":
                        ret_val = self.marketplace.add_to_cart(cart_id, prod)
                    elif operation_type == "remove":
                        ret_val = self.marketplace.remove_from_cart(cart_id, prod)
                    else:
                        raise ValueError(
                            f'Invalid operation type: {operation_type}.'
                            f'The operation type must be add or remove.')

                    if ret_val or ret_val is None:
                        i = i + 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)

import unittest
import logging

from threading import Lock, currentThread
from logging.handlers import RotatingFileHandler
from time import gmtime

from tema.product import Tea


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.last_producer_id = -1  
        self.last_cart_id = -1  

        self.last_producer_lock = Lock()
        self.last_cart_lock = Lock()
        self.producer_lock = Lock()
        self.print_lock = Lock()

        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {} 
        self.available_products = {} 
        self.carts = {}  
        self.all_products = []  
        self.unavailable_products = {}  

        
        self.logger = logging.getLogger('marketplace_logger')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname) '
                                      '- s%(funcName)21s() - %(message)s')
        logging.Formatter.converter = gmtime  
        self.handler = RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=10)
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def register_producer(self):
        
        with self.last_producer_lock:
            self.last_producer_id = self.last_producer_id + 1
            crt_id = self.last_producer_id

        self.producers[str(crt_id)] = 0
        self.logger.info(f'Registered producer with id: {crt_id}')
        return str(crt_id)

    def publish(self, producer_id, product):
        
        self.logger.info(f'Producer with id {producer_id} wants to publish product {product}')
        
        if self.producers[producer_id] >= self.queue_size_per_producer:
            self.logger.info(f'Producer with id {producer_id} can not publish product {product}'
                             f'because they reached the queue size per producer')
            return False

        
        with self.producer_lock:
            self.producers[producer_id] = self.producers[producer_id] + 1

        
        if product not in self.available_products:
            self.available_products[product] = []
        self.available_products[product].append(producer_id)
        


        self.all_products.append(product)
        self.logger.info(f'Producer with id {producer_id} published product {product}')
        return True

    def new_cart(self):
        
        with self.last_cart_lock:
            self.last_cart_id = self.last_cart_id + 1
            crt_cart_id = self.last_cart_id

        self.carts[crt_cart_id] = []


        self.logger.info(f'Registered new cart with id: {crt_cart_id}')
        return crt_cart_id

    def add_to_cart(self, cart_id, product):
        
        self.logger.info(f'A consumer wants to add in cart with id {cart_id} the product {product}')
        if product in self.all_products:
            
            self.carts[cart_id].append(product)
            with self.producer_lock:
                self.producers[self.available_products[product][- 1]] -= 1

            
            
            self.all_products.remove(product)
            if product not in self.unavailable_products:
                self.unavailable_products[product] = []
            self.unavailable_products[product].append(self.available_products[product].pop())
            self.logger.info(f'A consumer added in cart with id {cart_id} the product {product}')
            return True

        self.logger.info(f'A consumer could not add in cart '
                         f'with id {cart_id} the product {product},'
                         f'because the product is not available')
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info(f'A consumer wants to remove from cart'
                         f' with id {cart_id} the product {product}')
        if product in self.carts[cart_id]:
            
            self.carts[cart_id].remove(product)

            
            with self.producer_lock:
                self.producers[self.unavailable_products[product][-1]] += 1

            
            self.all_products.append(product)
            self.available_products[product]\
                .append(self.unavailable_products[product].pop())
            self.logger.info(f'A consumer removed from cart '
                             f'with id {cart_id} the product {product}')

        self.logger.error(f'A consumer wanted to remove from cart '
                          f'with id {cart_id} the product {product},'
                          f'but the product is not in that cart')

    def place_order(self, cart_id):
        
        self.logger.info(f'A consumer wants to place an order for cart with id {cart_id}')
        for product in self.carts.pop(cart_id, None):
            with self.print_lock:
                print(f'{currentThread().getName()} bought {product}')
        self.logger.info(f'A consumer placed an order for cart with id {cart_id}.')


class TestMarketplace(unittest.TestCase):
    def setUp(self):
        self.marketplace = Marketplace(1)

    def test_register_producer_with_one_producer(self):
        producer_id = self.marketplace.register_producer()
        self.assertEqual(producer_id, '0')
        self.assertEqual(str(self.marketplace.last_producer_id), producer_id)

    def test_register_producer_with_multiple_producers(self):
        producer_id1 = self.marketplace.register_producer()
        producer_id2 = self.marketplace.register_producer()
        producer_id3 = self.marketplace.register_producer()

        self.assertEqual(producer_id1, '0')
        self.assertEqual(producer_id2, '1')
        self.assertEqual(producer_id3, '2')
        self.assertEqual(str(self.marketplace.last_producer_id), producer_id3)

    def test_publish_without_producer_limit(self):
        product = Tea(name="Linden", type="Herbal", price=9)

        producer_id = self.marketplace.register_producer()
        ret_val = self.marketplace.publish(producer_id, product)

        self.assertTrue(ret_val)
        self.assertEqual(self.marketplace.producers[producer_id], 1)
        self.assertIsNotNone(self.marketplace.available_products)
        self.assertIsNotNone(self.marketplace.all_products)

    def test_publish_with_producer_limit(self):
        product1 = Tea(name="Linden", type="Herbal", price=9)
        product2 = Tea(name="Lipton", type="Herbal", price=10)

        producer_id = self.marketplace.register_producer()
        ret_val = self.marketplace.publish(producer_id, product1)
        ret_val = self.marketplace.publish(producer_id, product2)

        self.assertFalse(ret_val)
        self.assertEqual(self.marketplace.producers[producer_id], 1)
        self.assertNotIn(product2, self.marketplace.available_products)
        self.assertNotIn(product2, self.marketplace.all_products)

    def test_new_cart_with_one_cart(self):
        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, 0)
        self.assertEqual(self.marketplace.last_cart_id, cart_id)

    def test_new_cart_with_multiple_carts(self):
        cart_id1 = self.marketplace.new_cart()
        cart_id2 = self.marketplace.new_cart()
        cart_id3 = self.marketplace.new_cart()

        self.assertEqual(cart_id1, 0)
        self.assertEqual(cart_id2, 1)
        self.assertEqual(cart_id3, 2)
        self.assertEqual(self.marketplace.last_cart_id, cart_id3)

    def test_add_to_cart_for_available_product(self):
        product = Tea(name="Linden", type="Herbal", price=9)

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, product)
        ret_val = self.marketplace.add_to_cart(cart_id, product)

        self.assertTrue(ret_val)
        self.assertIn(product, self.marketplace.carts[cart_id])
        self.assertNotIn(product, self.marketplace.all_products)
        self.assertNotIn(producer_id, self.marketplace.available_products[product])
        self.assertEqual(self.marketplace.producers[producer_id], 0)

    def test_add_to_cart_for_unavailable_product(self):
        product = Tea(name="Linden", type="Herbal", price=9)

        cart_id = self.marketplace.new_cart()
        ret_val = self.marketplace.add_to_cart(cart_id, product)

        self.assertFalse(ret_val)
        self.assertNotIn(product, self.marketplace.carts[cart_id])

    def test_remove_from_cart(self):
        product = Tea(name="Linden", type="Herbal", price=9)

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.remove_from_cart(cart_id, product)

        self.assertNotIn(product, self.marketplace.carts[cart_id])
        self.assertNotIn(product, self.marketplace.unavailable_products[product])
        self.assertIn(product, self.marketplace.all_products)
        self.assertIn(producer_id, self.marketplace.available_products[product])
        self.assertEqual(self.marketplace.producers[producer_id], 1)

    def test_place_order(self):
        product = Tea(name="Linden", type="Herbal", price=9)

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.place_order(cart_id)



        self.assertNotIn(cart_id, self.marketplace.carts)

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        self.unique_id = marketplace.register_producer()

    def run(self):
        while True:
            for crt_product_info in self.products:
                (product, quantity, wait_time) = crt_product_info
                count = 0
                while count < quantity:
                    has_to_wait = not self.marketplace.publish(self.unique_id, product)
                    if has_to_wait:
                        time.sleep(wait_time)
                    else:
                        time.sleep(self.republish_wait_time)
                        count = count + 1


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
