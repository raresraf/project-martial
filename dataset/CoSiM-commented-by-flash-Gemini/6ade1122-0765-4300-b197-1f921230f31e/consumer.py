

import logging


from threading import Thread, Lock
from time import sleep

from tema.marketplace import Marketplace


class Consumer(Thread):
    

    __id = 1
    __id_lock = Lock()



    def __init__(self, carts, marketplace: Marketplace, retry_wait_time: int, **kwargs):
        

        Thread.__init__(self, **kwargs)

        with Consumer.__id_lock:
            self.__id = Consumer.__id
            Consumer.__id += 1

        self.__marketplace = marketplace
        self.__retry_wait_time = retry_wait_time
        self.__carts = carts

        self.__cart_id = None

    def add_product(self, product):
        

        status = self.__marketplace.add_to_cart(self.__cart_id, product)

        while not status:
            sleep(self.__retry_wait_time)
            status = self.__marketplace.add_to_cart(self.__cart_id, product)

    def add_products(self, product, quantity):
        

        for _ in range(quantity):
            self.add_product(product)

    def remove_product(self, product):
        
        self.__marketplace.remove_from_cart(self.__cart_id, product)

    def remove_products(self, product, quantity):
        

        for _ in range(quantity):
            self.remove_product(product)

    def buy_cart(self, cart):
        

        self.__cart_id = self.__marketplace.new_cart()
        logging.info(cart)

        for action in cart:
            if action['type'] == 'add':
                self.add_products(action['product'], action['quantity'])
            else:
                self.remove_products(action['product'], action['quantity'])

        products = self.__marketplace.place_order(self.__cart_id)
        for product in products:
            print(f'cons{self.__id} bought {product}'.strip())

        self.__cart_id = None

    def run(self):
        for cart in self.__carts:
            self.buy_cart(cart)

import logging
from logging.handlers import RotatingFileHandler
from queue import Queue
from threading import Lock
from typing import Dict, List
from unittest import TestCase

from tema.product import Product, Tea

LOGGER = logging.getLogger('MARKETPLACE')
HANDLER = RotatingFileHandler('marketplace.log', maxBytes=102, backupCount=10)
LOGGER.addHandler(HANDLER)


class Cart:
    

    __last_id = 1
    __id_lock = Lock()

    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.amount: Dict[str, int] = {}

        with Cart.__id_lock:
            self.__id = Cart.__last_id
            Cart.__last_id += 1

    def add(self, product):
        

        if product.name not in self.amount:
            self.amount[product.name] = 0

        self.amount[product.name] += 1
        self.products[product.name] = product

    def remove(self, product):
        

        self.amount[product.name] -= 1

    def list(self) -> List:
        

        result = []

        for product_name in self.amount:
            result += [self.products[product_name]] * self.amount[product_name]

        return result

    def get_id(self):
        
        return self.__id


class Marketplace:
    

    __producer_id = 0
    __producer_id_lock = Lock()



    def __init__(self, queue_size_per_producer: int):
        

        self.producer_capacity = queue_size_per_producer
        self.producers = {}
        self.prod_queue = {}

        self.all_products = {}
        self.reserved_products = {}
        self.carts = {}

        self.lock = Lock()

    def register_producer(self) -> str:
        

        with Marketplace.__producer_id_lock:
            producer_id = f"producer{Marketplace.__producer_id}"
            Marketplace.__producer_id += 1

        self.producers[producer_id] = {}

        LOGGER.info("register_producer -> %s", producer_id)
        return producer_id

    def publish(self, producer_id: str, product) -> bool:
        

        with self.lock:
            if product.name not in self.all_products:
                self.all_products[product.name] = 0

            if product.name not in self.producers[producer_id]:
                self.producers[producer_id][product.name] = 0

            if self.producers[producer_id][product.name] == self.producer_capacity:
                LOGGER.info("publish(%s, %s) -> False", producer_id, product)
                return False

            if product.name not in self.prod_queue:
                self.prod_queue[product.name] = Queue()

            self.producers[producer_id][product.name] += 1
            self.all_products[product.name] += 1
            self.prod_queue[product.name].put(producer_id)

        LOGGER.info("publish(%s, %s) -> True", producer_id, product)
        return True

    def new_cart(self) -> int:
        

        cart = Cart()
        self.carts[cart.get_id()] = cart

        LOGGER.info("new_cart -> %s", cart.get_id())
        return cart.get_id()

    def add_to_cart(self, cart_id: int, product) -> bool:
        

        with self.lock:
            if product.name not in self.all_products:
                self.all_products[product.name] = 0
            if product.name not in self.reserved_products:
                self.reserved_products[product.name] = 0

            if self.all_products[product.name] == self.reserved_products[product.name]:
                LOGGER.info("add_to_cart(%s, %s) -> False", cart_id, product.name)
                return False

            cart = self.carts[cart_id]

            cart.add(product)
            self.reserved_products[product.name] += 1

        LOGGER.info("add_to_cart(%s, %s) -> True", cart_id, product.name)
        return True

    def remove_from_cart(self, cart_id: int, product):
        

        LOGGER.info("remove_from_cart(%s, %s)", cart_id, product.name)
        with self.lock:
            cart = self.carts[cart_id]

            cart.remove(product)

    def place_order(self, cart_id):
        

        cart = self.carts[cart_id]
        products = cart.list()
        amount = cart.amount

        with self.lock:
            for product_id in cart.amount:
                self.all_products[product_id] -= amount[product_id]
                self.reserved_products[product_id] -= amount[product_id]

                while amount[product_id] > 0:
                    producer_id = self.prod_queue[product_id].get()
                    self.producers[producer_id][product_id] -= 1

                    amount[product_id] -= 1

        LOGGER.info("place_order(%s) -> %s", cart_id, products)
        return products


class TestMarketplace(TestCase):
    

    PRODUCER_COUNT = 10
    PRODUCER_QUEUE_SIZE = 10

    def setUp(self) -> None:
        self.marketplace = Marketplace(TestMarketplace.PRODUCER_QUEUE_SIZE)

    def test_register_producer(self):
        

        producers = set()
        for _ in range(TestMarketplace.PRODUCER_COUNT):
            producer_id = self.marketplace.register_producer()
            producers.add(producer_id)

        self.assertEqual(len(producers), TestMarketplace.PRODUCER_COUNT)

    def test_publish(self):
        

        producer_id = self.marketplace.register_producer()

        product_name = 'test_tea'
        product_price = 10
        product_type = 'test_type'
        product = Tea(product_name, product_price, product_type)

        self.marketplace.publish(producer_id, product)

        queue_producer_id = self.marketplace.prod_queue[product_name].get()
        self.assertEqual(queue_producer_id, producer_id)

        self.assertEqual(self.marketplace.all_products[product_name], 1)
        self.assertEqual(self.marketplace.producers[producer_id][product_name], 1)

    def test_new_cart(self):
        

        cart_id_1 = self.marketplace.new_cart()
        cart_id_2 = self.marketplace.new_cart()

        self.assertNotEqual(cart_id_1, cart_id_2)

    def test_add_to_cart(self):
        

        producer_id = self.marketplace.register_producer()

        product_name = 'test_tea'
        product_price = 10
        product_type = 'test_type'
        product = Tea(product_name, product_price, product_type)

        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, product))

        self.marketplace.publish(producer_id, product)
        self.assertTrue(self.marketplace.add_to_cart(cart_id, product))

    def test_remove_from_cart(self):
        

        product_name = 'test_tea'
        product_price = 10
        product_type = 'test_type'
        product = Tea(product_name, product_price, product_type)

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.marketplace.remove_from_cart(cart_id, product)

        self.assertEqual(len(self.marketplace.carts[cart_id].list()), 0)

    def test_order(self):
        

        producer_id = self.marketplace.register_producer()

        product_name = 'test_tea'
        product_price = 10
        product_type = 'test_type'
        product = Tea(product_name, product_price, product_type)

        cart_id = self.marketplace.new_cart()

        self.marketplace.publish(producer_id, product)
        self.marketplace.add_to_cart(cart_id, product)
        products = self.marketplace.place_order(cart_id)

        self.assertEqual(products, [product])
        self.assertEqual(self.marketplace.all_products[product_name], 0)

from threading import Thread
from time import sleep

from tema.marketplace import Marketplace


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.__orders = products
        self.__marketplace: Marketplace = marketplace
        self.__wait_time = republish_wait_time

        self.__id = marketplace.register_producer()

    @staticmethod
    def prepare_product(product, prepare_time):
        

        sleep(prepare_time)

        return product

    def publish_product(self, product, prepare_time):
        

        self.prepare_product(product, prepare_time)

        status = self.__marketplace.publish(self.__id, product)

        while not status:
            sleep(self.__wait_time)
            status = self.__marketplace.publish(self.__id, product)

    def publish_products(self, product, quantity, prepare_time):
        
        for _ in range(quantity):
            self.publish_product(product, prepare_time)

    def run(self):
        while True:
            for order in self.__orders:


                product, quantity, prepare_time = order

                self.publish_products(product, quantity, prepare_time)


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
