


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = self.marketplace.new_cart()

    def run(self):
        for cart in self.carts:
            for prod in cart:
                type_op = prod["type"]
                prod_id = prod["product"]
                quantity = prod["quantity"]

                if type_op == "add":
                    for i in range(quantity):
                        while not self.marketplace.add_to_cart(self.name, str(prod_id)):
                            sleep(self.retry_wait_time)

                if type_op == "remove":
                    for i in range(quantity):
                        self.marketplace.remove_from_cart(self.name, prod_id)
                        sleep(self.retry_wait_time)

        list_of_products = self.marketplace.place_order(int(self.name))

        with self.marketplace.lock_printer:
            for product in list_of_products:
                print(f"cons{self.name} bought {product}")

from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.format = logging.Formatter("%(asctime)s %(message)s")
        self.rotating_handler = RotatingFileHandler('marketpplace.log', 'w')
        self.rotating_handler.setFormatter(self.format)
        self.log.addHandler(self.rotating_handler)

        self.queue_size_per_producer = queue_size_per_producer
        self.dict_prod = {}
        self.dict_con = {}
        self.lock_prod = Lock()
        self.lock_register = Lock()
        self.lock_con = Lock()
        self.generateId = 0
        self.cartId = 0
        self.products_list = []
        self.lock_publish = Lock()
        self.lock_printer = Lock()

    def register_producer(self):
        
        self.log.info("beginning of func register_product")
        with self.lock_register:
            self.generateId += 1



        self.dict_prod[self.generateId] = []
        self.log.info("end of func register_product")
        return self.generateId

    def publish(self, producer_id, product):
        
        self.log.info("beginning of func publish")

        with self.lock_publish:
            if self.queue_size_per_producer > len(self.dict_prod[producer_id]):
                self.dict_prod[producer_id].append(product)
                self.products_list.append(product)
                self.log.info("end of func publish -> True")
                return True
            else:
                self.log.info("end of func publish -> False")
                return False

    def new_cart(self):
        
        self.log.info("beginning of func new_cart")

        with self.lock_con:
            self.cartId += 1



        self.dict_con[self.cartId] = []
        self.log.info("end of func new_cart")
        return self.cartId

    def add_to_cart(self, cart_id, product):
        
        self.log.info("beginning of func add_to_cart")
        with self.lock_prod:
            if product in self.products_list:
                self.products_list.remove(product)
                for key in self.dict_prod:
                    for value in self.dict_prod[key]:
                        if value == product:
                            self.dict_con[int(cart_id)].append(product)
                            self.dict_prod[key].remove(value)
                            self.log.info("end of func add_to_cart -> True")
                            return True
            self.log.info("end of func add_to_cart -> False")
            return False

    def remove_from_cart(self, cart_id, product):
        
        self.log.info("beginning of func remove_from_func")
        if str(product) in self.dict_con[int(cart_id)]:
            self.dict_con[int(cart_id)].remove(str(product))


            self.products_list.append(str(product))
        self.log.info("end of func remove_from_func")

    def place_order(self, cart_id):
        
        self.log.info("func place_order")
        return self.dict_con[int(cart_id)]


class TestMarketPlace(unittest.TestCase):
    def setUp(self):
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):


        self.marketplace.register_producer()
        self.marketplace.publish(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        self.assertEqual(self.marketplace.products_list,
                         ["Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')"])

    def test_new_cart(self):
        self.assertEqual(1, self.marketplace.new_cart())

    def test_add_to_cart(self):


        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        self.marketplace.add_to_cart(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        print(f'List of products in cart: {self.marketplace.dict_con}')

    def test_remove_from_cart(self):


        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        self.marketplace.add_to_cart(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        print(f'List of products in cart: {self.marketplace.dict_con}')
        print(f'List of all available products: {self.marketplace.products_list}')
        self.marketplace.remove_from_cart(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        print(f'List of products in cart: {self.marketplace.dict_con}')
        print(f'List of all available products: {self.marketplace.products_list}')

    def test_place_order(self):


        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        self.marketplace.add_to_cart(1, "Coffee(name='Arabica', price=9, acidity=5.02, roast_level='MEDIUM')")
        print(f'List of products in cart: {self.marketplace.dict_con}')
        print(f'List of all available products: {self.marketplace.products_list}')
        print(f'Placed order: {self.marketplace.place_order(1)}')

from queue import Queue
from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id = self.marketplace.register_producer()

    def run(self):
        while True:
            for product in self.products:
                product_id = product[0]
                quantity = product[1]
                sleep_time = product[2]
                for i in range(quantity):
                    flag = self.marketplace.publish(self.id, str(product_id))
                    if flag:
                        sleep(sleep_time)
                    sleep(self.republish_wait_time)
