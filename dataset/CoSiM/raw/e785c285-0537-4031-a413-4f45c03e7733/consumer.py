


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_request(self, requests, new_cart_id):
        
        requests_made = 1
        while True:

            if requests_made > requests["quantity"]:
                break

            if self.marketplace.add_to_cart(
                    new_cart_id, requests["product"]):
                requests_made += 1
            else:
                time.sleep(self.retry_wait_time)

    def rm_request(self, requests, new_cart_id):
        
        requests_made = 1
        while True:

            if requests_made > requests["quantity"]:
                break

            error_code = self.marketplace.remove_from_cart(
                new_cart_id, requests["product"])

            if error_code is None:
                requests_made += 1
            else:
                time.sleep(self.retry_wait_time)

    def run(self):
        
        for new_cart in self.carts:

            new_cart_id = self.marketplace.new_cart()

            
            for requests in new_cart:
                if requests["type"] == "add":
                    self.add_request(requests, new_cart_id)
                else:
                    self.rm_request(requests, new_cart_id)

            
            
            for product in self.marketplace.place_order(new_cart_id):
                print(self.name + " bought " + str(product))


import os
import threading
import time
import logging
import unittest
from logging.handlers import RotatingFileHandler
from .product import Tea, Coffee

FIELNAME = "marketplace.log"
logging.basicConfig(filename=FIELNAME,
                    format='%(asctime)s %(message)s',
                    filemode='w')

logging.Formatter.converter = time.gmtime
logger = logging.getLogger('marketplace_loger')

should_roll_over = os.path.isfile(FIELNAME)
handler = RotatingFileHandler(FIELNAME, mode='w', backupCount=10)


if should_roll_over:  
    handler.doRollover()


logger.setLevel(logging.DEBUG)


class Marketplace:
    

    def check_item_market(self, product):
        
        logger.info("In check_item %s", str(product))
        for product_complex in self.marketplace_products:
            if product == product_complex[0]:
                logger.info("Out check_item")
                return product_complex
        logger.info("Out check_item")
        return None

    def check_item_cart(self, product, cart_id):
        
        logger.info("In check_item_cart %s %s", str(product), str(cart_id))
        for product_complex in self.cart[cart_id]:
            if product == product_complex[0]:
                logger.info("Out check_item_cart")
                return product_complex
        logger.info("Out check_item_cart")
        return None

    def __init__(self, queue_size_per_producer):
        

        
        self.queue_size_per_producer = queue_size_per_producer

        
        self.marketplace_products = []

        
        self.producers_list = {}
        self.mr_of_producers = 0

        
        self.cart = {}
        self.nr_of_carts = 0



        self.lock = threading.Lock()

    def register_producer(self):
        
        logger.info("In register_producer")

        
        

        with self.lock:
            self.producers_list[self.mr_of_producers] = []
            self.mr_of_producers = self.mr_of_producers + 1
            logger.info("Out register_producer")
            return self.mr_of_producers - 1

    def publish(self, producer_id, product):
        
        logger.info("In publish %s %s", producer_id, str(product))
        if len(self.producers_list[int(producer_id)]) >= self.queue_size_per_producer:
            logger.info("Out publish")
            return False

        self.producers_list[int(producer_id)].append(product)
        self.marketplace_products.append((product, int(producer_id)))
        logger.info("Out publish")
        return True

    def new_cart(self):
        

        
        
        logger.info("In new_cart")
        with self.lock:
            self.cart[self.nr_of_carts] = []
            self.nr_of_carts = self.nr_of_carts + 1
            logger.info("Out new_cart")
            return self.nr_of_carts - 1

    def add_to_cart(self, cart_id, product):
        
        logger.info("In add_to_cart %s %s", str(cart_id), str(product))
        
        
        
        with self.lock:
            item = self.check_item_market(product)
            if item is not None:
                self.cart[cart_id].append(item)


                self.marketplace_products.remove(item)
                logger.info("Out add_to_cart")
                return True
            logger.info("Out add_to_cart")
            return False

    def remove_from_cart(self, cart_id, product):
        
        logger.info("In remove_from_cart %s %s", str(cart_id), str(product))
        item = self.check_item_cart(product, cart_id)
        
        
        if item is not None:
            self.cart[cart_id].remove(item)
            self.marketplace_products.append(item)
            logger.info("Out remove_from_cart")

    def place_order(self, cart_id):
        
        logger.info("In place_order %s", str(cart_id))

        

        product_list = []
        for product_extended in self.cart[cart_id]:
            product = product_extended[0]
            producer_id = product_extended[1]
            self.producers_list[producer_id].remove(product)
            product_list.append(product)
        logger.info("Out place_order")
        return product_list


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        
        queue_size_per_producer = 10
        self.marketplace = Marketplace(queue_size_per_producer)
        self.marketplace.register_producer()
        self.marketplace.new_cart()

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 1,
                         'incorrect id')
        self.assertEqual(self.marketplace.producers_list[1], [],
                         'incorrect empty list')
        self.assertEqual(len(self.marketplace.producers_list[1]), 0,
                         'incorrect len')

    def test_publish(self):
        
        tea = Tea("lipton", 10, "green_tea")
        coffee = Coffee("lavazza", 2, "5.05", "MEDIUM")
        self.assertEqual(self.marketplace.publish("0", tea), True,
                         'incorrect test_publish')
        self.assertEqual(self.marketplace.publish("0", coffee), True,
                         'incorrect test_publish')

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 1,
                         'incorrect new_cart')
        self.assertEqual(self.marketplace.new_cart(), 2,
                         'incorrect new_cart')

    def test_add_to_cart(self):
        
        tea = Tea("lipton", 10, "green_tea")
        coffee = Coffee("lavazza", 2, "5.05", "MEDIUM")
        coffee2 = Coffee("lavazza", 2, "5.05", "HIGH")

        self.marketplace.publish("0", tea)
        self.marketplace.publish("0", coffee2)

        self.assertEqual(self.marketplace.add_to_cart(0, tea), True,
                         'incorrect add_to_cart')
        self.assertEqual(self.marketplace.add_to_cart(0, coffee), False,
                         'incorrect add_to_cart')
        self.assertEqual(self.marketplace.add_to_cart(0, coffee2), True,
                         'incorrect add_to_cart')

    def test_remove_from_cart(self):
        
        tea = Tea("lipton", 10, "green_tea")
        coffee = Coffee("lavazza", 2, "5.05", "MEDIUM")
        coffee2 = Coffee("lavazza", 2, "5.05", "HIGH")

        self.marketplace.publish("0", tea)
        self.marketplace.publish("0", coffee2)

        self.marketplace.add_to_cart(0, tea)
        self.marketplace.add_to_cart(0, coffee2)

        self.assertEqual(self.marketplace.remove_from_cart(0, tea), None,
                         'incorrect remove_from_cart')
        self.assertEqual(self.marketplace.remove_from_cart(0, coffee), None,
                         'incorrect remove_from_cart')
        self.assertEqual(self.marketplace.remove_from_cart(0, coffee2), None,
                         'incorrect remove_from_cart')
        self.assertEqual(self.marketplace.cart[0], [],
                         'incorrect remove_from_cart')

    def test_place_order(self):
        
        tea = Tea("lipton", 10, "green_tea")
        coffee2 = Coffee("lavazza", 2, "5.05", "HIGH")

        self.marketplace.publish("0", tea)
        self.marketplace.publish("0", coffee2)

        self.marketplace.add_to_cart(0, tea)
        self.marketplace.add_to_cart(0, coffee2)

        product_list = [tea, coffee2]

        self.assertEqual(self.marketplace.place_order(0), product_list,
                         'incorrect remove_from_cart')

    def test_check_item_market(self):
        
        tea = Tea("lipton", 10, "green_tea")
        coffee = Coffee("lavazza", 2, "5.05", "MEDIUM")
        coffee2 = Coffee("lavazza", 2, "5.05", "HIGH")

        self.marketplace.publish("0", tea)
        self.marketplace.publish("0", coffee2)

        print(self.marketplace.marketplace_products)
        self.assertEqual(self.marketplace.check_item_market(tea),
                         self.marketplace.marketplace_products[0],
                         'incorrect check_item_market')
        self.assertEqual(self.marketplace.check_item_market(coffee), None,
                         'incorrect check_item_market')
        self.assertEqual(self.marketplace.check_item_market(coffee2),
                         self.marketplace.marketplace_products[1],
                         'incorrect check_item_market')

    def test_check_item_cart(self):
        
        tea = Tea("lipton", 10, "green_tea")
        coffee = Coffee("lavazza", 2, "5.05", "MEDIUM")
        coffee2 = Coffee("lavazza", 2, "5.05", "HIGH")

        self.marketplace.publish("0", tea)
        self.marketplace.publish("0", coffee2)

        self.marketplace.add_to_cart(0, tea)
        self.marketplace.add_to_cart(0, coffee2)

        print(self.marketplace.cart[0])

        self.assertEqual(self.marketplace.check_item_cart(tea, 0), self.marketplace.cart[0][0],
                         'incorrect check_item_cart')
        self.assertEqual(self.marketplace.check_item_cart(coffee, 0), None,
                         'incorrect check_item_cart')
        self.assertEqual(self.marketplace.check_item_cart(coffee2, 0), self.marketplace.cart[0][1],
                         'incorrect check_item_cart')


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.id_producer = None
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        self.id_producer = self.marketplace.register_producer()
        while 1:
            for product_process in self.products:

                
                
                product = product_process[0]
                product_nr = product_process[1]
                product_cnt = 0
                product_wait_time = product_process[2]

                while True:

                    
                    
                    if product_cnt >= product_nr:
                        break

                    
                    
                    
                    
                    if self.marketplace.publish(str(self.id_producer), product):
                        product_cnt += 1
                        time.sleep(product_wait_time)
                    else:
                        time.sleep(self.republish_wait_time)
