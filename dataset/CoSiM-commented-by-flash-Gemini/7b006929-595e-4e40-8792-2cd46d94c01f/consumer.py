

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        for cart in self.carts:
            for data in cart:
                for i in range(data["quantity"]):
                    ret = False
                    while not ret:
                        if data["type"] == "add":
                            ret = self.marketplace.add_to_cart(self.cart_id, data["product"])
                        else:
                            ret = self.marketplace.remove_from_cart(self.cart_id, data["product"])
                        if not ret:
                            time.sleep(self.retry_wait_time)
        self.marketplace.place_order(self.cart_id)

import time
from threading import Lock
import logging
import logging.handlers


class Marketplace:
    
    formatter = logging.Formatter('(%(asctime)s %(threadName)-9s) %(funcName)s %(message)s')
    formatter.converter = time.gmtime
    handler = logging.handlers.RotatingFileHandler('marketplace.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    MARK = []

    
    GET_PROD = {}

    
    PROD = {}

    
    
    
    CONS = {}

    lock = Lock()

    def __init__(self, queue_size_per_producer):
        
        self.logger.info(f" <- queue_zie_per_producer = {queue_size_per_producer}")
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        
        self.lock.acquire()
        producer_id = len(self.PROD)


        self.PROD[producer_id] = self.queue_size_per_producer
        self.lock.release()
        self.logger.info(f" -> producer_id = {producer_id}")
        return producer_id

    def publish(self, producer_id, product):
        
        self.logger.info(f" <- producer_id = {producer_id}, product = {product}")
        self.lock.acquire()
        if self.PROD[producer_id] > 0:
            self.PROD[producer_id] -= 1
            self.MARK.append(product)
            self.GET_PROD[product] = producer_id
            self.lock.release()
            self.logger.info(f" -> True")
            return True
        self.lock.release()
        self.logger.info(f" -> False")
        return False

    def new_cart(self):
        
        self.lock.acquire()
        cart_id = len(self.CONS)
        self.CONS[cart_id] = {}


        self.lock.release()
        self.logger.info(f" -> cart_id = {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        try:
            self.MARK.remove(product)
        except ValueError:
            self.lock.release()
            self.logger.info(f" -> False")
            return False
        producer_id = self.GET_PROD[product]
        try:
            self.CONS[cart_id][producer_id].append(product)
        except KeyError:
            self.CONS[cart_id][producer_id] = []
            self.CONS[cart_id][producer_id].append(product)
        self.PROD[producer_id] += 1


        self.lock.release()
        self.logger.info(f" -> True")
        return True

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        for entry in self.CONS[cart_id]:
            for search_product in self.CONS[cart_id][entry]:
                if product == search_product:
                    self.CONS[cart_id][entry].remove(search_product)
                    self.MARK.append(product)
                    self.PROD[entry] -= 1
                    self.lock.release()
                    self.logger.info(f" -> True")


                    return True
        self.lock.release()
        self.logger.info(f" -> False")
        return False

    def place_order(self, cart_id):
        
        self.logger.info(f" <- cart_id = {cart_id}")
        for prod_list in self.CONS[cart_id].values():
            for prod in prod_list:
                self.lock.acquire()
                print(f'cons{cart_id + 1} bought {prod}')
                self.lock.release()

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for product in self.products:
                for i in range(product[1]):
                    ret = False
                    while not ret:
                        ret = self.marketplace.publish(self.producer_id, product[0])
                        time.sleep(product[2])
            time.sleep(self.republish_wait_time)
