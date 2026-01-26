

from time import sleep
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        i = 0
        
        for i in range(len(self.carts)):
            listAddRem = self.carts[i]
            j = 0
            new_cart = self.marketplace.new_cart()
            
            for j in range(len(listAddRem)):
                
                command = listAddRem[j]
                AddRem = command["type"]
                prod = command["product"]
                qty = command["quantity"]
                k = 0
                
                while k < qty:
                    if AddRem == "add":
                        res = self.marketplace.add_to_cart(new_cart, prod)
                        if res:
                            k += 1
                        else:
                            sleep(self.retry_wait_time)
                    elif AddRem == "remove":
                        self.marketplace.remove_from_cart(new_cart, prod)
                        k += 1
            
            self.marketplace.place_order(new_cart)

import sys
import logging
sys.path.append('/tema/product')
import unittest
from threading import Lock, currentThread


class TestLogging:
	_myLogg = None

	def __init__(cls):
		if cls._myLogg is None:
			cls._myLogg = logging.getLogger("logg")
			file = logging.handlers.RotatingFileHandler('marketplace.log',
														mode='a', maxBytes=4096, backupCount=0,
														encoding=None, delay=False, errors=None)

			formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
			file.setFormatter(formatter)
			logger.addHandler(file)

class TestMarketplace(unittest.TestCase):

    @classmethod
    def setUp(self):


        self.Marketplace = Marketplace(3)
        self.prod1 = self.Marketplace.register_producer()
        self.prod2 = self.Marketplace.register_producer()
        self.cart1 = self.Marketplace.new_cart()
        self.cart2 = self.Marketplace.new_cart()
        obj = product.Coffee("Indonezia", 5.05, 1, "Medium")
        obj2 = product.Tea("Linden", "Herbal", 9)
        obj3 = product.Coffee("ElDuMari", 6.05, 2, "Large")
        self.produs1 = obj
        self.produs2 = obj2
        self.produs3 = obj3

    def test_register(self):
        self.assertEqual(self.prod1, 0)
        self.assertEqual(self.prod2, 1)

    def test_publish(self):
        i = 0
        for i in range(5):
            if i < 3:
                self.assertTrue(self.Marketplace.publish(
                    self.prod1, self.produs1))
                self.assertTrue(self.Marketplace.publish(
                    self.prod2, self.produs2))
            else:
                self.assertFalse(self.Marketplace.publish(
                    self.prod1, self.produs1))
                self.assertFalse(self.Marketplace.publish(
                    self.prod2, self.produs2))

    def test_new_cart(self):
        self.assertEqual(self.cart1, 0)
        self.assertEqual(self.cart2, 1)

    def test_add_to_cart(self):
        i = 0
        j = 0
        for j in range(3):
            self.Marketplace.publish(self.prod1, self.produs1)
            self.Marketplace.publish(self.prod2, self.produs2)
        for i in range(6):
            if i < 3:
                self.assertTrue(self.Marketplace.add_to_cart(
                    self.cart1, self.produs1))
                self.assertTrue(self.Marketplace.add_to_cart(
                    self.cart2, self.produs2))
            else:
                self.assertFalse(self.Marketplace.add_to_cart(
                    self.cart1, self.produs1))
                self.assertFalse(self.Marketplace.add_to_cart(
                    self.cart2, self.produs2))

    def test_remove_from_cart(self):
        for j in range(3):


            self.Marketplace.publish(self.prod1, self.produs1)
            self.Marketplace.publish(self.prod2, self.produs2)
            self.Marketplace.add_to_cart(self.cart1, self.produs1)



        self.Marketplace.add_to_cart(self.cart1, self.produs2)

        self.Marketplace.remove_from_cart(self.cart1, self.produs1)
        self.assertEqual(self.Marketplace.cartList[self.cart1],
                         [self.produs1, self.produs1, self.produs2])

        self.Marketplace.remove_from_cart(self.cart1, self.produs2)
        self.assertEqual(self.Marketplace.cartList[self.cart1],
                         [self.produs1, self.produs1])

        self.Marketplace.remove_from_cart(self.cart1, self.produs1)
        self.Marketplace.remove_from_cart(self.cart1, self.produs1)
        self.assertEqual(self.Marketplace.cartList[self.cart1], [])

    def test_place_order(self):


        self.Marketplace.publish(self.prod1, self.produs1)
        self.Marketplace.publish(self.prod2, self.produs2)
        self.Marketplace.publish(self.prod1, self.produs2)

        self.Marketplace.add_to_cart(self.cart1, self.produs1)
        self.Marketplace.add_to_cart(self.cart1, self.produs2)
        self.Marketplace.add_to_cart(self.cart1, self.produs2)

        self.Marketplace.publish(self.prod1, self.produs3)
        self.Marketplace.add_to_cart(self.cart1, self.produs3)

        self.Marketplace.publish(self.prod1, self.produs3)
        self.Marketplace.add_to_cart(self.cart1, self.produs3)

        self.Marketplace.remove_from_cart(self.cart1, self.produs1)


        self.Marketplace.remove_from_cart(self.cart1, self.produs2)
        self.Marketplace.remove_from_cart(self.cart1, self.produs3)

        self.assertEqual(self.Marketplace.place_order(self.cart1),
                         [self.produs2, self.produs3])


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = -1
        self.cart_id = -1
        self.listofproducts = [] 
        self.productsgotid = {} 
        self.producersNrQueueSize = [] 
        self.cartList = {} 
        self.lock_operations = Lock()
        self.CartsLocks = Lock()
        self.LockAddRemCart = Lock()
        self.ListSubmit = Lock()

    def register_producer(self):
        
        with self.lock_operations:
            self.id_producer += 1
            
            self.producersNrQueueSize.insert(self.id_producer, self.queue_size_per_producer)
        return self.id_producer

    def publish(self, producer_id, product):
        
        with self.lock_operations:
            
            if self.producersNrQueueSize[producer_id] - 1 > 0:
                
            	self.producersNrQueueSize[producer_id] -= 1
            	
            	self.productsgotid[product] = producer_id
            	self.listofproducts.append(product)
            	return True
            return False

    def new_cart(self):
        
        with self.CartsLocks:
            self.cart_id += 1
            
            
            self.cartList[self.cart_id] = []
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        with self.LockAddRemCart:
            
            if product in self.listofproducts:
            	
                self.cartList[cart_id].append(product)
                
                self.producersNrQueueSize[self.productsgotid[product]] += 1
                
                self.listofproducts.remove(product)
                return True
            return False

    def remove_from_cart(self, cart_id, product):
        
        with self.LockAddRemCart:
        	
            self.cartList[cart_id].remove(product)
            self.producersNrQueueSize[self.productsgotid[product]] -= 1
            self.listofproducts.append(product)

    def place_order(self, cart_id):
        
        with self.ListSubmit:
            
            for i in self.cartList[cart_id]:
                
                print(currentThread().getName() + " bought " + str(i))
        return self.cartList[cart_id]

from time import sleep
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        id_producer = self.marketplace.register_producer()
        while True:
            
            for type_prod in self.products:
                
                id_produs = type_prod[0]
                
                qty = type_prod[1]
                
                time_to_wait = type_prod[2]
                i = 0
                
                while i < qty:
                    ret = self.marketplace.publish(id_producer, id_produs)
                    if ret:
                        
                        i += 1
                        sleep(time_to_wait)
                    else:
                        
                        sleep(self.republish_wait_time)
