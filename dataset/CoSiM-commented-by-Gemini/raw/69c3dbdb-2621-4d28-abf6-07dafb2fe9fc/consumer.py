


import time
from threading import Thread
from multiprocessing import Lock

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)


    def run(self):
        for c in self.carts:
            c_id = self.marketplace.new_cart()
            for req in c:
                type = req['type']
                prod = req['product']
                qty = req['quantity']
                if type == 'add':
                    for _ in range(0, qty):

                        prod_added_flag = self.marketplace.add_to_cart(c_id, prod)

                        while prod_added_flag is False:
                            time.sleep(self.retry_wait_time)
                            prod_added_flag = self.marketplace.add_to_cart(c_id, prod)

                elif type == 'remove':
                    for _ in range(0, qty):
                       self.marketplace.remove_from_cart(c_id, prod)

            shopping_list = self.marketplace.place_order(c_id)

            self.marketplace.cons_lock.acquire()            
            for elem in shopping_list:
                print("{} bought {}".format(self.getName(), elem))
            self.marketplace.cons_lock.release()



from itertools import product
import logging


from logging.handlers import RotatingFileHandler
from multiprocessing import Lock

import unittest

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.lock_prod = Lock()
        self.lock_cart = Lock()
        self.prod_id = 0
        self.lock = Lock()

        
        self.prod_list = {}

        
        self.carts_list = {}

        self.cart_id = 0
        self.cons_lock = Lock()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        RotatingFileHandler(filename='marketplace.log', maxBytes=50000, backupCount=20)
        self.logger.addHandler(RotatingFileHandler("marketplace.log"))

    def register_producer(self):
        

        self.logger.info("Register producer was called")

        self.lock_prod.acquire()
        self.prod_id += 1
        self.prod_list[self.prod_id] = []
        self.logger.info("Register producer completed")
        self.lock_prod.release()

        return self.prod_id


    def publish(self, producer_id, product):
        

        self.logger.info("Publish was called")

        if len(self.prod_list[producer_id]) == self.queue_size_per_producer:
            self.logger.info("Publish returned False")
            return False

        self.prod_list[producer_id].append(product)
        return True


    def new_cart(self):
        

        self.logger.info("New cart was called")

        self.lock_cart.acquire() 
        self.cart_id += 1
        self.carts_list[self.cart_id] = []
        self.logger.info("New cart was created successfully")
        self.lock_cart.release()

        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        self.logger.info("Add to cart was called")

        for i in self.prod_list:
            if product in self.prod_list[i]:

                self.prod_list[i].remove(product)
                self.carts_list[cart_id].append(tuple((i, product)))
                self.logger.info("Add to cart was made successfully")
                return True
        self.logger.info("Add to cart could not be completed")
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("Remove from cart was called")

        self.lock.acquire()
        for i in self.carts_list[cart_id]:

            if i[1] == product:
                self.prod_list[i[0]].append(product)
                self.carts_list[cart_id].remove(i)
                self.logger.info("Remove from cart was successful")
                break

        self.lock.release()
        return


    def place_order(self, cart_id):
        
        self.logger.info("Place order was called")

        cart = self.carts_list[cart_id].copy()
        self.carts_list[cart_id] = []

        final_list = []
        for i in cart:
            final_list.append(i[1])
        self.logger.info("Place order created the list")
        return final_list


class TestMarketplace(unittest.TestCase):
    def setUp(self):
        self.marketplace = Marketplace(8)
        self.prod_list = {"name1":"Tea", "name2":"Coffee", "name3":"Wild berries Tea"}
        self.name_list = ["Tea", "Coffee", "Wild berries Tea"]
        self.list = ["Coffee", "Wild berries Tea"]

    def test_register_prod(self):

        rez = self.marketplace.register_producer()
        self.assertEqual(rez, 1, 'Producer not updated')

    def test_publish(self):
        self.marketplace.register_producer()
        rez = self.marketplace.publish(1, self.prod_list["name3"])
        self.assertTrue(rez, 'Product is not published')

    def test_new_cart(self):
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        rez = self.marketplace.new_cart()
        self.assertEqual(rez, 4, 'Cart id not updated')

    def test_add_to_cart(self):

        self.marketplace.register_producer()
        self.marketplace.publish(1, self.prod_list["name3"])
        self.marketplace.publish(1, self.prod_list["name2"])
        id_1 = self.marketplace.new_cart()
        id_2 = self.marketplace.new_cart()
        rez1 = self.marketplace.add_to_cart(id_1, self.prod_list["name1"])
        rez2 = self.marketplace.add_to_cart(id_1, self.prod_list["name2"])
        rez3 = self.marketplace.add_to_cart(id_1, self.prod_list["name3"])          
        self.assertEqual(id_2, 2, 'Cart id not set right')
        self.assertTrue(rez3, 'Product not added')
        self.assertFalse(rez1, 'Product not added')

        list_check = []
        for i in self.marketplace.carts_list[1]:
            list_check.append(i[1])

        self.assertListEqual(self.list, list_check, 'Not all products added are in list')

    def test_remove_from_cart(self):
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.prod_list["name1"])
        self.marketplace.publish(1, self.prod_list["name2"])
        self.marketplace.publish(1, self.prod_list["name3"])
        id_1 = self.marketplace.new_cart()
        self.marketplace.add_to_cart(id_1, self.prod_list["name1"])
        self.marketplace.add_to_cart(id_1, self.prod_list["name2"])
        self.marketplace.remove_from_cart(id_1, self.prod_list["name1"])

        list_check = []
        for i in self.marketplace.carts_list[1]:
            list_check.append(i[1])

        self.assertListEqual(['Coffee'], list_check, 'List mismatch after remove')


    def test_place_order(self):
        
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.prod_list["name1"])
        self.marketplace.publish(1, self.prod_list["name2"])
        self.marketplace.publish(1, self.prod_list["name3"])
        id_1 = self.marketplace.new_cart()
        self.marketplace.add_to_cart(id_1, self.prod_list["name1"])
        self.marketplace.add_to_cart(id_1, self.prod_list["name2"])
        self.marketplace.remove_from_cart(id_1, self.prod_list["name2"])
        self.marketplace.add_to_cart(id_1, self.prod_list["name3"])
        self.marketplace.add_to_cart(id_1, self.prod_list["name2"])
        self.marketplace.remove_from_cart(id_1, self.prod_list["name1"])

        list_check = []
        for i in self.marketplace.carts_list[1]:
            list_check.append(i[1])

        self.assertListEqual(['Wild berries Tea','Coffee'], list_check, 'List mismatch after placing order')>>>> file: producer.py


import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):

        p_id = 0
        p_id = self.marketplace.register_producer()

        while True:
            for prod in self.products:
                product_id = prod[0]
                q = prod[1]
                for _ in range(0, q):

                    if self.marketplace.publish(p_id, product_id) is True:
                        break

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
