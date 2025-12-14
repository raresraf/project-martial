


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)
        self.name = kwargs["name"]

    def run(self):
        id_cart = self.marketplace.new_cart()

        for cart in self.carts:
            for com in cart:
                type_com = com['type']
                quantity = com['quantity']
                product = com['product']
                if type_com == "remove":
                    for i in range(quantity):
                        self.marketplace.remove_from_cart(id_cart, product)
                elif type_com == "add":
                    i = 0
                    while 1:
                        if i >= quantity:
                            break
                        while not self.marketplace.add_to_cart(id_cart, product):
                            sleep(self.retry_wait_time)
                        i += 1
        for prod in self.marketplace.place_order(id_cart):
            to_print = "{} bought {}".format(self.name, prod)
            print(to_print)

from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import time
import unittest

class TestMarketplace(unittest.TestCase):
    
    def test_register_producer(self):
        
        market = Marketplace(10)
        self.assertEquals(market.register_producer(), 0, "Expected to return id 0")

    def test_publish(self):
        
        market = Marketplace(10)
        producter_id = market.register_producer()
        self.assertTrue(market.publish(producter_id, "Tea"), "Expected to return True")

    def test_new_cart(self):
        market = Marketplace(10)
        self.assertEquals(market.new_cart(), 0, "Expected to return id 0")

    def test_remove_from_cart(self):
        market = Marketplace(10)
        cart_id = market.new_cart()
        market.add_to_cart(cart_id, "Tea")
        self.assertTrue(market.remove_from_cart(cart_id, "Tea"), "Expected to return True")

    def test_place_order(self):
        market = Marketplace(10)
        cart_id = market.new_cart()
        market.add_to_cart(cart_id, "Tea")
        self.assertEquals(market.place_order(cart_id, "Tea")[0], "Tea", "Expected to return Tea")


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.mutex = Lock()
        self.producers = []
        self.carts = []
        self.producers_no = -1
        self.consumers_no = -1


        logging.Formatter.converter = time.gmtime
        log_formatter = logging.Formatter("%(asctime)s:%(levelname)s: \
                                         %(filename)s::%(funcName)s:%(lineno)d %(message)s")
        logger = logging.getLogger()
        logger.propagate = False
        handler_file = RotatingFileHandler('marketplace.log', maxBytes=4096, backupCount=1)

        handler_file.setFormatter(log_formatter)
        logger.addHandler(handler_file)
        logger.setLevel(logging.INFO)

        self.logger = logger

    def register_producer(self):
        
        self.mutex.acquire()
        self.producers_no += 1
        self.producers.append([])
        producer_id = self.producers_no
        self.mutex.release()
        self.logger.info("New producer: {}".format(producer_id))

        return producer_id

    def publish(self, producer_id, product):
        

        publish_state = False

        self.mutex.acquire()
        if len(self.producers[producer_id]) <= self.queue_size_per_producer:
            self.producers[producer_id].append(product)
            publish_state = True
        self.mutex.release()

        self.logger.info("state: {} \
            add product {} of producer: {}".format(publish_state, product, producer_id))

        return publish_state

    def new_cart(self):
        
        self.mutex.acquire()
        cart = []
        self.carts.append(cart)
        self.consumers_no += 1
        cart_id = self.consumers_no
        self.mutex.release()
        self.logger.info("New char id: {}".format(cart_id))

        return cart_id

    def add_to_cart(self, cart_id, product):
        

        self.mutex.acquire()
        i = 0
        for prod in self.producers:
            for produs in prod:
                if product == produs:
                    self.carts[cart_id].append([product, i])
                    prod.remove(product)
                    self.mutex.release()
                    self.logger.info("add product {} to cart id: {}".format(product, cart_id))
                    return True
            i += 1
        self.mutex.release()
        self.logger.error("add product {} to cart id: {}".format(product, cart_id))

        return False

    def remove_from_cart(self, cart_id, product):
        
        self.mutex.acquire()


        for produs in self.carts[cart_id]:
            if produs[0] == product:
                self.producers[produs[1]].append(produs[0])
                self.carts[cart_id].remove(produs)
                break
        self.mutex.release()
        self.logger.info("remove product {} to cart id: {}".format(product, cart_id))
        return True

    def place_order(self, cart_id):
        
        products = [x[0] for x in self.carts[cart_id]]
        self.logger.info("place order {} to cart id: {}".format(products, cart_id))

        return products


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        producer_id = self.marketplace.register_producer()

        while 1:
            for elem in self.products:
                (id_prod, cantitate, timp_asteptare) = elem
                sleep_time = cantitate * timp_asteptare
                sleep(sleep_time)
                i = 0
                while 1:
                    if i >= cantitate:
                        break
                    while not self.marketplace.publish(producer_id, id_prod):
                        sleep(self.republish_wait_time)
                    i += 1


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
