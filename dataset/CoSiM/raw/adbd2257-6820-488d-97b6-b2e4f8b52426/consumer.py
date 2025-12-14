


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        for cart in self.carts:
            id_cart = self.marketplace.new_cart()
            for curr_op in cart:
                for _ in range(curr_op["quantity"]):
                    if curr_op["type"] == "add":
                        while True:
                            check = self.marketplace.add_to_cart(id_cart, curr_op["product"])
                            if not check:
                                sleep(self.retry_wait_time)
                            else:
                                break
                    elif curr_op["type"] == "remove":
                        self.marketplace.remove_from_cart(id_cart, curr_op["product"])
            self.marketplace.place_order(id_cart)

import threading
import unittest
import logging
import logging.handlers
from tema.product import Product

LOG_FILE_NAME = 'marketplace.log'
LOGGING_LEVEL = logging.DEBUG

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.allowed_quantity = queue_size_per_producer
        self.producers = {}
        self.last_prod_id = 0
        self.last_cart_id = 0
        self.consumers = {}
        self.market = {}
        self.all_products = []

        self.reg_p_lock = threading.Lock()
        self.publish_lock = threading.Lock()
        self.new_cart_lock = threading.Lock()
        self.add_to_cart_lock = threading.Lock()
        self.logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE_NAME,
                                                            interval=30, backupCount=10)
        handler.setFormatter(formatter)
        self.logger = logging.getLogger() 
        self.logger.addHandler(handler)
        self.logger.setLevel(LOGGING_LEVEL)



    def register_producer(self):
        
        self.reg_p_lock.acquire()
        self.logger.info("-- Intrare in metoda register_producer")
        self.last_prod_id += 1
        self.producers[self.last_prod_id] = 0
        self.reg_p_lock.release()
        self.logger.info("-- Iesire din metoda register_producer")
        return self.last_prod_id

    def publish(self, producer_id, product):
        

        self.publish_lock.acquire()
        self.logger.info("-- Intrare in metoda publish cu param producer_id = %s si product = %s",
                         str(producer_id), str(product))
        self.market[product] = producer_id
        self.all_products.append(product)
        old_val = self.producers.get(producer_id)
        self.producers[producer_id] = old_val + 1
        self.publish_lock.release()

        if self.producers[producer_id] > self.allowed_quantity:
            self.producers[producer_id] = old_val
            self.logger.info("-- Iesire din metoda publish cu rezultatul False")
            return False

        self.logger.info("-- Iesire din metoda publish cu rezultatul True")
        return True

    def new_cart(self):
        
        self.new_cart_lock.acquire()
        self.logger.info("-- Intrare in metoda new_cart")
        self.last_cart_id += 1
        self.consumers[self.last_cart_id] = []
        self.logger.info("-- Iesire din metoda new_cart")
        self.new_cart_lock.release()
        return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        
        self.add_to_cart_lock.acquire()
        self.logger.info("-- Intrare in metoda add_to_cart cu param cart_id = %s si product = %s",
                         str(cart_id), str(product))
        if product not in self.market or product not in self.all_products:
            self.logger.info("-- Iesire din metoda add_to_cart -> Nu exista produsul")
            self.add_to_cart_lock.release()
            return False

        self.consumers[cart_id].append(product)
        prod = self.market[product]
        self.all_products.remove(product)
        self.producers[prod] -= 1


        self.logger.info("-- Iesire triumfatoare din add_to_cart")
        self.add_to_cart_lock.release()
        return True

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("-- Intrare in metoda remove_from_cart cu param cart_if = %s si %s",
                         str(cart_id), str(product))
        self.consumers[cart_id].remove(product)
        prod = self.market.get(product)
        self.all_products.append(product)
        self.producers[prod] += 1
        self.logger.info("-- Iesire din metoda remove_from_cart")

    def place_order(self, cart_id):
        
        self.logger.info("-- Intrare in metoda place_order cu param cart_id = %s ", str(cart_id))
        for prod in self.consumers[cart_id]:
            print(threading.currentThread().getName() + " bought " + str(prod))
        self.logger.info("-- Iesire din metoda place_order")
        return self.consumers[cart_id]

class TestMarketPlace(unittest.TestCase):
    
    def test_register_producer(self):
        
        market = Marketplace(12)
        self.assertEqual(1, market.register_producer())

    def test_publish(self):
        
        market = Marketplace(2)
        market.register_producer()
        self.assertEqual(True, market.publish(1, Product("TeaName", 23.0)))

    def test_new_cart(self):
        
        market = Marketplace(2)
        first_cart = market.new_cart()
        second_cart = market.new_cart()
        self.assertEqual(1, first_cart)
        self.assertEqual(2, second_cart)

    def test_add_to_cart(self):
        
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutz", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutz", 5.0))
        self.assertEqual(0, len(market.all_products))

    def test_remove_from_cart(self):
        
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.remove_from_cart(cart_id, Product("Cafelutzaaa", 5.0))
        self.assertEqual(1, len(market.all_products))

    def test_place_order(self):
        
        market = Marketplace(2)
        market.register_producer()
        cart_id = market.new_cart()
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.publish(1, Product("Ceiut", 3.0))
        market.publish(1, Product("Cafelutzaaa", 5.0))
        market.publish(1, Product("Ceiut", 3.0))
        market.publish(1, Product("Altceva", 3.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Cafelutzaaa", 5.0))
        market.add_to_cart(cart_id, Product("Ceiut", 3.0))
        self.assertEqual(3, len(market.place_order(cart_id)))


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.published_products = 0
        Thread.__init__(self, **kwargs)

    def run(self):
        id_prod = self.marketplace.register_producer()

        while 1:
            for (product, quantity, production_time) in self.products:
                for i in range(quantity):
                    check = self.marketplace.publish(id_prod, product)
                    if check:
                        time.sleep(production_time)
                    else:
                        time.sleep(self.republish_wait_time)
                        i += 1
                    i -= 1


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
