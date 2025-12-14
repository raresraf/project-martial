


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        cart_id = self.marketplace.new_cart()


        for product in self.carts:
            for attribute in product:
                command = attribute.get("type")
                product = attribute.get("product")
                quantity = attribute.get("quantity")
                if command == "remove":
                    i = 0
                    while i < quantity:
                        self.marketplace.remove_from_cart(cart_id, product)
                        i += 1
                elif command == "add":
                    i = 0
                    while i < quantity:
                        no_wait = self.marketplace.add_to_cart(cart_id, product)
                        if no_wait:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)
        order = self.marketplace.place_order(cart_id)
        for prod in order:
            print(self.name, "bought", prod)


from threading import Lock
from logging.handlers import RotatingFileHandler
import logging
import time

LOGGER = logging.getLogger('marketplace_logger')
LOGGER.setLevel(logging.INFO)

FORMATTER = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
FORMATTER.converter = time.gmtime()

HANDLER = RotatingFileHandler('marketplace.log', maxBytes=5000, backupCount=10)
HANDLER.setFormatter(FORMATTER)

LOGGER.addHandler(HANDLER)


class Marketplace:
    
    def __init__(self, queue_size_per_producer: int):
        
        self.lock_consumer = Lock()
        self.lock_producer = Lock()

        self.producers = [[]]
        self.carts = [[]]
        self.no_producers = 0
        self.no_carts = 0

        self.queue_size_per_producer = queue_size_per_producer



    def register_producer(self):
        
        LOGGER.info("A new producer is registered.")
        self.no_producers += 1
        self.producers.append([])
        LOGGER.info("Producer with id %s registerd.", self.no_producers)

        return self.no_producers

    def publish(self, producer_id, product):
        
        LOGGER.info('Producer with id %d is publishig the product: %s', producer_id, product)
        if producer_id > self.no_producers:
            LOGGER.error('Producer with id: %d does not exist', producer_id)
            raise ValueError("Producer does not exist!")

        product_list = self.producers[producer_id]
        with self.lock_producer:
            if len(product_list) >= self.queue_size_per_producer:
                can_publish = False
            else:
                product_list.append(product)
                can_publish = True
        LOGGER.info("Producer published: %s", str(can_publish))

        return can_publish

    def new_cart(self):
        
        LOGGER.info("New cart with id %d is being created.", self.no_carts + 1)
        self.no_carts += 1
        self.carts.append([])

        return self.no_carts

    def add_to_cart(self, cart_id, product):
        
        LOGGER.info("Cart with id %d is adding %s.", cart_id, product)
        can_add = False
        index = -1
        with self.lock_consumer:
            for i in range(0, self.no_producers):
                for prod_in_list in self.producers[i]:
                    if prod_in_list == product:
                        index = i
                        break
            if index >= 0:
                self.carts[cart_id].append(product)
                can_add = True

        if can_add:
            LOGGER.info("Product was added to the cart.")
        else:
            LOGGER.info("Product could not be added to the cart.")

        return can_add

    def remove_from_cart(self, cart_id, product):
        
        LOGGER.info("Cart with id %d is removing product %s.", cart_id, product)
        found = False
        with self.lock_consumer:
            if product in self.carts[cart_id]:
                found = True
            if found:
                self.carts[cart_id].remove(product)

    def place_order(self, cart_id):
        
        LOGGER.info("Cart with id %d placed an order.", cart_id)
        if cart_id > self.no_carts:
            LOGGER.error("Cart with id %d is invalid!", cart_id)
            raise ValueError("Cart does not exist!")

        for prod in self.carts[cart_id]:
            for producer in self.producers:
                if prod in producer:
                    producer.remove(prod)
                    break

        LOGGER.info("Product list: %s.", self.carts[cart_id])

        return self.carts[cart_id].copy()

import unittest
from marketplace import Marketplace
from product import Product


class MarketplaceTestCase(unittest.TestCase):
    

    product_test = Product("coffee", 10)
    marketplace = Marketplace(4)

    def test_place_order_exception(self):
        
        marketplace = Marketplace(2)
        self.assertRaises(ValueError, marketplace.place_order, 1)

    def test_place_order(self):
        
        self.marketplace.carts = [[self.product_test]]
        self.marketplace.no_carts = 1
        response = self.marketplace.place_order(0)
        expected = [self.product_test]
        self.assertEqual(response, expected)

    def test_register_producer(self):
        
        marketplace = Marketplace(10)
        result = marketplace.register_producer()
        self.assertEqual(result, 1)

    def test_publish_exception(self):
        
        marketplace = Marketplace(5)
        self.assertRaises(ValueError, marketplace.publish, 2, Product("coffee", 10))

    def test_publish_method(self):
        
        self.marketplace.register_producer()
        result = self.marketplace.publish(1, self.product_test)
        self.assertEqual(result, True)

    def test_publish_method_false(self):
        
        self.marketplace.register_producer()
        for _ in range(0, 4):
            self.marketplace.publish(1, self.product_test)
        response = self.marketplace.publish(1, self.product_test)
        self.assertEqual(response, False)

    def test_new_cart(self):
        
        marketplace = Marketplace(2)
        result = marketplace.new_cart()
        self.assertEqual(result, 1)

    def test_add_cart(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.product_test)
        self.marketplace.new_cart()
        result = self.marketplace.add_to_cart(1, self.product_test)
        self.assertEqual(result, True)

    def test_add_cart_false(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        result = self.marketplace.add_to_cart(1, self.product_test)
        self.assertEqual(result, False)

    def test_remove_from_cart(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.product_test)
        self.marketplace.new_cart()
        self.assertIsNone(self.marketplace.remove_from_cart(0, self.product_test))


if __name__ == '__main__':
    unittest.main()


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace


        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        while True:
            prod_id = self.marketplace.register_producer()
            for product in self.products:
                i = 0
                self.publish(i, prod_id, product)

    def publish(self, i, prod_id, product):
        
        while i < product[1]:
            no_wait = self.marketplace.publish(prod_id, product[0])
            if no_wait:
                i += 1
                time.sleep(product[2])
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
