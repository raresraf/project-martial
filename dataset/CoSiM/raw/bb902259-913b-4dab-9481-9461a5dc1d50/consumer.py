


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        
        curr_id = self.marketplace.new_cart()
        for curr_cart in self.carts:
            for elem in curr_cart:
                
                action_type = elem["type"]
                prod_id = elem["product"]
                quantity = elem["quantity"]
                
                for i in range(quantity):
                    if action_type == "add":
                        
                        
                        while not self.marketplace.add_to_cart(curr_id, prod_id):
                            sleep(self.retry_wait_time)
                    if action_type == "remove":
                        self.marketplace.remove_from_cart(curr_id, prod_id)
                        sleep(self.retry_wait_time)
        order = self.marketplace.place_order(curr_id)
        for i in order:
            
            with self.marketplace.print_lock:
                print(f"cons{curr_id} bought {i}")

from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(asctime)s;%(message)s")
        self.rotating_file_handler = RotatingFileHandler('marketplace.log', 'w')
        self.rotating_file_handler.setLevel(logging.INFO)
        self.rotating_file_handler.setFormatter(self.formatter)
        self.log.addHandler(self.rotating_file_handler)

        self.queue_size_per_producer = queue_size_per_producer
        
        self.producers = {}
        self.no_prod = 0
        
        self.carts = {}
        self.no_carts = 0
        self.market_products = []

        self.product_lock = Lock()
        self.cart_lock = Lock()
        self.publish_lock = Lock()
        self.add_lock = Lock()
        self.print_lock = Lock()

    def register_producer(self):


        
        with self.product_lock:
            self.log.info("begin register method")
            self.no_prod += 1
            id_p = self.no_prod

        
        self.producers[id_p] = []


        self.log.info("end register method")
        return id_p

    def publish(self, producer_id, product):
        
        with self.publish_lock:
            self.log.info("begin publish method")
            
            if len(self.producers[int(producer_id)]) > self.queue_size_per_producer:
                self.log.info("end publish method with False")
                return False

            
            self.producers[int(producer_id)].append(product)
            self.market_products.append(product)
            self.log.info("end publish method with True")
            return True

    def new_cart(self):
        
        with self.cart_lock:
            self.log.info("begin new_cart method")
            self.no_carts += 1
            new_cart_id = self.no_carts

        
        self.carts[new_cart_id] = []


        self.log.info("end new_cart method")
        return new_cart_id

    def add_to_cart(self, cart_id, product):
        
        with self.add_lock:
            self.log.info("begin add_to_cart method")
            
            if product in self.market_products:
                
                
                self.carts[cart_id].append(product)
                self.market_products.remove(product)
                
                for id_p in self.producers:
                    if product in self.producers[id_p]:
                        self.producers[id_p].remove(product)
                        break


                self.log.info("end add_to_cart method with True")
                return True
        self.log.info("end add_to_cart method with False")
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.log.info("begin remove_from_cart method")
        
        
        
        for prod in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove(prod)
                self.market_products.append(prod)
                break
        self.log.info("end remove_from_cart method")

    def place_order(self, cart_id):
        
        self.log.info("begin place_order method")
        
        order = self.carts[cart_id]
        self.log.info("end place_order method")
        return order


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        
        self.marketplace.register_producer()
        print(self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')"))
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])

    def test_new_cart(self):
        
        self.marketplace.register_producer()
        print(self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')"))
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])
        print(self.marketplace.new_cart())

    def test_add(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])
        self.marketplace.new_cart()
        print(self.marketplace.market_products)
        print(self.marketplace.producers)
        print(self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')"))
        print(self.marketplace.producers)
        print(self.marketplace.carts)

    def test_remove(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])
        self.marketplace.new_cart()
        print(self.marketplace.market_products)
        print(self.marketplace.producers)
        self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')")
        print(self.marketplace.carts)
        print(self.marketplace.remove_from_cart(1, "Tea(name='Linden', price=9, type='Herbal')"))
        print(self.marketplace.carts)

    def test_place_order(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, "Tea(name='Linden', price=9, type='Herbal')")
        self.assertEqual(self.marketplace.market_products,
                         ["Tea(name='Linden', price=9, type='Herbal')"])
        self.marketplace.new_cart()
        print(self.marketplace.market_products)
        print(self.marketplace.producers)
        self.marketplace.add_to_cart(1, "Tea(name='Linden', price=9, type='Herbal')")
        print(self.marketplace.carts)
        print(self.marketplace.place_order(1))


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        
        producer_id = self.marketplace.register_producer()
        
        while True:
            for prod in self.products:
                
                product_id = prod[0]
                quantity = prod[1]
                wait_time = prod[2]
                
                i = 0
                while i < int(quantity):
                    if self.marketplace.publish(str(producer_id), product_id):
                        i += 1
                        
                        sleep(wait_time)
                
                sleep(self.republish_wait_time)
