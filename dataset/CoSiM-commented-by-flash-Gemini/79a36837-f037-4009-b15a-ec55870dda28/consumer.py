

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        pass

    def run(self):
        
        id_cart = self.marketplace.new_cart()

        
        for cart_list in self.carts:
            for cart in cart_list:
                type_command = cart.get("type")
                prod = cart.get("product")
                quantity = cart.get("quantity")
                if type_command == "add":
                    
                    while quantity > 0:
                        ret = self.marketplace.add_to_cart(id_cart, prod)

                        if ret:
                            quantity -= 1
                        else:
                            time.sleep(self.retry_wait_time)
                else:

                    while quantity > 0:
                        quantity -= 1
                        self.marketplace.remove_from_cart(id_cart, prod)

        list_prod = self.marketplace.place_order(id_cart)

        for p in list_prod:
            print(self.name, "bought", p)

        pass

import unittest
from threading import Lock
from tema.product import Coffee, Tea
import logging


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.id_consumer = 0


        self.id_producer = 0
        
        self.carts = {}
        
        self.products = {}

        
        self.lock_reg_producer = Lock()

        
        self.lock_cart = Lock()

        pass

    def register_producer(self):
        
        logging.info('Entered in register_producer')
        self.lock_reg_producer.acquire()
        self.id_producer += 1
        self.lock_reg_producer.release()
        
        self.products[self.id_producer] = []
        logging.info('Returned id_prod from register_producer')


        return self.id_producer
        pass

    def publish(self, producer_id, product):
        
        logging.info('Entered publish')
        add_product = False

        if len(self.products.get(producer_id)) < self.queue_size_per_producer:
            add_product = True
        if add_product:
            self.products.get(producer_id).append(product)
        logging.info('Returned from publish')
        return add_product

        pass

    def new_cart(self):
        
        logging.info('Entered new_cart')

        self.lock_cart.acquire()
        self.id_consumer += 1
        self.lock_cart.release()

        self.carts[self.id_consumer] = []
        logging.info('Returned from new_cart')



        return self.id_consumer

        pass

    def add_to_cart(self, cart_id, product):
        
        logging.info('Entered in add_to_cart')

        id_producer = 0
        producer_found = False
        for key in list(self.products.keys()):
            for prod in self.products.get(key):
                if prod == product:
                    producer_found = True
                    id_producer = key
                    break

        if producer_found:
            
            self.products.get(id_producer).remove(product)
            
            self.carts.get(cart_id).append([product, id_producer])
        logging.info('Returned from add_to_cart')



        return producer_found
        pass

    def remove_from_cart(self, cart_id, product):
        
        logging.info('Entered in remove_from_cart')

        for prod, id_producer in self.carts.get(cart_id):
            if prod == product:
                self.carts.get(cart_id).remove([product, id_producer])
                self.products.get(id_producer).append(product)
                break
        logging.info('Exit from remove_from_cart')

        pass

    def place_order(self, cart_id):
        
        logging.info('Entered in place_order')

        products_list = []
        for prod, id_prod in self.carts.get(cart_id):
            products_list.append(prod)

        logging.info('Returned from place_order')

        return products_list
        pass


class TestMarketplace(unittest.TestCase):

    def setUp(self):
        self.marketplace = Marketplace(2)

        self.coffee1 = Coffee(name="Indonesia", acidity="5.05", roast_level="MEDIUM", price=1)
        self.coffee2 = Coffee(name="Brasil", acidity="5.09", roast_level="MEDIUM", price=7)
        self.tea1 = Tea(name="Linden", type="Herbal", price=7)
        self.tea2 = Tea(name="Cactus fig", type="Green", price=5)

    def test_register_producer(self):
        self.setUp()
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.register_producer(), 3)
        self.assertEqual(self.marketplace.register_producer(), 4)

    def test_publish(self):
        self.setUp()
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(1, self.coffee1))
        self.assertTrue(self.marketplace.publish(1, self.coffee1))
        self.assertFalse(self.marketplace.publish(1, self.coffee1))
        self.assertFalse(self.marketplace.publish(1, self.coffee1))
        self.assertFalse(self.marketplace.publish(1, self.coffee1))

    def test_new_cart(self):
        self.setUp()
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.new_cart(), 3)
        self.assertEqual(self.marketplace.new_cart(), 4)
        self.assertEqual(self.marketplace.new_cart(), 5)

    def test_add_to_cart(self):
        self.setUp()
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.coffee1)
        self.marketplace.publish(1, self.coffee1)
        self.marketplace.publish(1, self.coffee1)

        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))
        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))
        self.assertFalse(self.marketplace.add_to_cart(1, self.coffee1))
        self.assertFalse(self.marketplace.add_to_cart(1, self.coffee1))

    def test_remove_from_cart(self):
        self.setUp()
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.coffee1)
        self.assertTrue(self.marketplace.add_to_cart(1, self.coffee1))
        self.assertFalse(self.marketplace.add_to_cart(1, self.coffee1))
        self.marketplace.remove_from_cart(1, self.coffee1)

    def test_place_order(self):
        self.setUp()
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.coffee1)
        self.marketplace.add_to_cart(1, self.coffee1)
        self.marketplace.publish(1, self.tea1)
        self.marketplace.add_to_cart(1, self.tea1)
        self.marketplace.remove_from_cart(1, self.coffee1)
        self.assertEqual(self.marketplace.place_order(1), '[Tea(name=\'Linden\', price=7, type=\'Herbal\')]')


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        pass

    def run(self):
        while True:
            
            producer_id = self.marketplace.register_producer()

            
            for prod in self.products:
                
                
                product = prod[0]
                quantity = prod[1]
                wait_time = prod[2]

                while quantity > 0:
                    ret = self.marketplace.publish(producer_id, product)
                    if ret:
                        quantity -= 1
                        time.sleep(wait_time)
                    else:
                        time.sleep(self.republish_wait_time)
        pass


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
