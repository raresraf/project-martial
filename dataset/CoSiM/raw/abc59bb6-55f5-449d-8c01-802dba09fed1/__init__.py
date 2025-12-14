


from threading import Thread
from time import sleep
from typing import List

from tema.marketplace import Marketplace


class Consumer(Thread):
    

    def __init__(self, carts: List, marketplace: Marketplace, retry_wait_time: float, **kwargs):
        
        Thread.__init__(self, kwargs=kwargs)

        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self.logger = marketplace.logger

    def run(self):
        
        log_msg = "Started consumer " + str(self.kwargs['name'])
        self.marketplace.log(log_msg, str(self.kwargs['name']))

        
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()
            self.marketplace.assign_owner(cart_id, str(self.kwargs['name']))
            cart_iter = iter(cart)
            log_msg = "NEW CART" + str(cart_id)
            self.marketplace.log(log_msg, str(self.kwargs['name']))

            
            req_item = next((item for item in cart_iter), None)

            
            while req_item is not None:

                if req_item['type'] == 'add':
                    res = self.marketplace.add_to_cart(
                        cart_id, req_item['product'])

                    if res:
                        req_item['quantity'] -= 1

                        if req_item['quantity'] == 0:
                            req_item = next((item for item in cart_iter), None)
                    else:
                        sleep(self.retry_wait_time)

                elif req_item['type'] == 'remove':
                    self.marketplace.remove_from_cart(
                        cart_id, req_item['product'])

                    req_item['quantity'] -= 1

                    if req_item['quantity'] == 0 :
                        req_item=next((item for item in cart_iter), None)

            
            self.marketplace.place_order(cart_id)

        self.marketplace.sign_out(str(self.kwargs['name']))

import logging
from logging.handlers import RotatingFileHandler

from threading import Lock, Semaphore
import unittest
from tema.product import Coffee, Product


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.q_limit = queue_size_per_producer
        self.producers = []
        self.carts = []
        self.consumers = []
        self.lock = Lock()

        
        logger = logging.getLogger("log_asc")
        logger.setLevel(logging.INFO)
        rfh = RotatingFileHandler('my_log.log', mode='w')
        rfh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        rfh.setFormatter(formatter)
        logger.addHandler(rfh)
        self.logger = logger
        self.mname = "MK"
        self.all_completed = False

    def log(self, msg, src):
        
        self.logger.info(src + ":" + msg)

    def register_producer(self):
        
        new_producer = {
            'id': len(self.producers),
            'queue': [],
            'empty_sem': Semaphore(value=self.q_limit),
            'full_sem': Semaphore(0)
        }
        self.producers.append(new_producer)

        log_msg = "REG PROD [" + str(new_producer['id']) + ']'
        self.log(log_msg, self.mname)
        return new_producer['id']



    def publish(self, producer_id: int, product: Product):
        

        prod_queue = self.producers[producer_id]['queue']
        prod_esem = self.producers[producer_id]['empty_sem']
        prod_fsem = self.producers[producer_id]['full_sem']

        acquired = prod_esem.acquire(blocking=False)
        if not acquired:
            log_msg = "REJ PUB REQ S:PROD[" + \
                str(producer_id) + "] " + str(product)
            self.log(log_msg, self.mname)
            return False

        prod_queue.append([product, True])
        log_msg = "ACC PUB REQ S:PROD[" + \
            str(producer_id) + "] " + \
            str(product) + " SLOTS [" + \
            str(self.q_limit - len(prod_queue)) + \
            "]"
        self.log(log_msg, self.mname)
        prod_fsem.release()
        return True

    def new_cart(self):
        
        self.lock.acquire()
        new_cart = {
            'id': len(self.carts),
            'items': [],
            'completed': False,
            'owner': ""
        }
        self.carts.append(new_cart)
        self.lock.release()

        
        log_msg = "REG CART [" + str(new_cart['id']) + "]"
        self.log(log_msg, self.mname)
        return new_cart['id']

    def assign_owner(self, cart_id: int, owner: str):
        
        for cart in self.carts:
            if cart['id'] == cart_id:
                cart['owner'] = owner

        if owner not in self.consumers:
            self.consumers.append(owner)

    def product_search(self, name: str):
        
        item_prod = None
        for producer in self.producers:
            for prod in producer['queue']:

                if prod[0].name == name and prod[1]:
                    item_prod = (prod, producer)
                    return item_prod



        return None

    def add_to_cart(self, cart_id: int, product: Product):
        

        log_msg = "ADD REQ [" + self.carts[cart_id]['owner'] + \
            "][C" + str(cart_id) + "] " + str(product)

        
        c_iter = iter(self.carts)
        cart = next((c for c in c_iter if c['id'] == cart_id), None)

        
        item_prod = self.product_search(product.name)
        
        
        if item_prod is not None:

            req_item = item_prod[0]
            if req_item[1]:
                req_item[1] = False

                
                cart['items'].append(item_prod)
                log_msg = "ACC " + log_msg
                self.log(log_msg, self.mname)
                return True

        log_msg = "REJ " + log_msg
        self.log(log_msg, self.mname)


        return False

    def remove_from_cart(self, cart_id: int, product: Product):
        

        
        req_prod_name = product.name
        prod_to_remove = None
        for prod in self.carts[cart_id]['items']:
            if prod[0][0].name == req_prod_name:
                prod[0][1] = True
                prod_to_remove = prod

        before_remove = len(self.carts[cart_id]['items'])
        self.carts[cart_id]['items'].remove(prod_to_remove)
        
        after_remove = len(self.carts[cart_id]['items'])
        log_msg = "DEL REQ " + str(prod_to_remove) + \
            str(before_remove) + " " + str(after_remove)
        self.log(log_msg, self.mname)



    def place_order(self, cart_id: int):
        
        
        self.carts[cart_id]['completed'] = True

        
        log_msg = "\n"
        for item in self.carts[cart_id]['items']:
            producer = item[1]
            prod_esem = producer['empty_sem']
            prod_fsem = producer['full_sem']

            prod_fsem.acquire()
            if item[0] in producer['queue']:
                producer['queue'].remove(item[0])
            else:
                err_log = "ERR COULD NOT FIND " + str(item[0])
                self.log(err_log, self.mname)
            prod_esem.release()

        
        for item in self.carts[cart_id]['items']:
            log_msg += self.carts[cart_id]['owner'] + \
                ' bought ' + str(item[0][0]) + '\n'

        print(log_msg[1:-1])
        self.log(log_msg, self.mname)

    def sign_out(self, cons: str):
        
        self.consumers.remove(cons)
        log_msg = "LOGOUT " + cons + " REMAINING " + str(len(self.consumers))
        self.log(log_msg, self.mname)


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(3)

    def test_1_register_producer(self):
        
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)
        ret = market.register_producer()
        self.assertEqual(ret, 1)

    def test_2_new_cart(self):
        
        market = self.marketplace

        ret = market.new_cart()
        self.assertEqual(ret, 0)
        ret = market.new_cart()
        self.assertEqual(ret, 1)

    def test_3_publish(self):
        
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        ret = market.publish(0, Coffee("TestCoff2", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        ret = market.publish(0, Coffee("TestCoff3", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        ret = market.publish(0, Coffee("TestCoff4", 10, "0.01", "Medium"))
        self.assertEqual(ret, False)

    def test_4_add_to_cart(self):
        
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        ret = market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        ret = market.add_to_cart(0, Coffee("TestCoff2", 10, "0.01", "Medium"))
        self.assertEqual(ret, False)

    def test_5_remove_from_cart(self):
        
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        ret = market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        market.remove_from_cart(
            0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(Coffee("TestCoff1", 10, "0.01", "Medium")
                         in market.carts[0]['items'], False)

    def test_6_place_order(self):
        
        market = self.marketplace

        ret = market.register_producer()
        self.assertEqual(ret, 0)

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        ret = market.publish(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)

        ret = market.add_to_cart(0, Coffee("TestCoff1", 10, "0.01", "Medium"))
        self.assertEqual(ret, True)
        self.assertEqual([Coffee("TestCoff1", 10, "0.01", "Medium"),
                         False] in market.producers[0]['queue'], True)
        market.place_order(0)
        self.assertEqual([Coffee("TestCoff1", 10, "0.01", "Medium"),
                         False] in market.producers[0]['queue'], False)

    def test_7_assign_owner(self):
        
        market = self.marketplace

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        market.assign_owner(0, "TestOwner")
        self.assertEqual(market.carts[0]['owner'], "TestOwner")

    def test_8_sign_out(self):
        
        market = self.marketplace

        ret = market.new_cart()
        self.assertEqual(ret, 0)

        market.assign_owner(0, "TestOwner")
        self.assertEqual(market.carts[0]['owner'], "TestOwner")

        self.assertEqual(len(market.consumers), 1)
        market.sign_out("TestOwner")
        self.assertEqual(len(market.consumers), 0)


from threading import Thread
from time import sleep
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product


class Producer(Thread):
    

    def __init__(self, products: List[Product], marketplace: Marketplace,
                 republish_wait_time: float, **kwargs):
        
        Thread.__init__(self, kwargs=kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.curr_index = 0
        self.curr_product = list(self.products[0])
        self.prod_id = -1

    def produce(self) -> Product:
        
        
        self.curr_product[1] -= 1

        
        sleep(float(self.curr_product[2]))

        
        
        if self.curr_product[1] == 0:
            self.curr_index += 1

        return self.curr_product[0]

    def run(self):
        log_msg = "Started producer " + str(self.kwargs['name'])
        self.marketplace.log(log_msg, str(self.kwargs['name']))

        self.prod_id = self.marketplace.register_producer()
        loop_flag = True
        while loop_flag:

            
            produced_item = self.produce()
            if len(self.products) > self.curr_index:
                if self.curr_product[1] == 0:
                    self.curr_product = list(self.products[self.curr_index])
            else:
                self.curr_product = list(self.products[0])
                self.curr_index = 0

            
            
            was_published = False
            while not was_published:
                was_published = self.marketplace.publish(
                    self.prod_id, produced_item)
                if not was_published:
                    sleep(self.republish_wait_time)

                if len(self.marketplace.consumers) == 0:
                    loop_flag = False
                    break


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
