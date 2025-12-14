


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        

        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                if operation['type'] == 'add':
                    for _ in range(operation['quantity']):
                        success = self.marketplace.add_to_cart(cart_id, operation['product'])
                        while not success:
                            time.sleep(self.retry_wait_time)


                            success = self.marketplace.add_to_cart(cart_id, operation['product'])
                else:
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            
            orders = self.marketplace.place_order(cart_id)
            for order in orders:
                print(self.kwargs['name'], 'bought', order)

import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
import unittest
from tema.product import Tea
from tema.product import Coffee

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producers_slots = {}
        self.producers_queues = {}
        self.carts = {}
        self.next_producer_id = 0
        self.next_cart_id = 0

        
        self.slots_locks = {}
        self.queues_locks = {}
        self.producer_id_lock = Lock()
        self.cart_id_lock = Lock()

        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=20000, backupCount=5)
        self.logger.addHandler(handler)

    def register_producer(self):
        

        
        self.producer_id_lock.acquire()
        producer_id = str(self.next_producer_id)
        self.next_producer_id += 1
        self.producer_id_lock.release()

        
        self.producers_slots[producer_id] = self.queue_size_per_producer
        self.producers_queues[producer_id] = []
        self.slots_locks[producer_id] = Lock()
        self.queues_locks[producer_id] = Lock()

        self.logger.info('Register producer: producer_id = %s', producer_id)
        return producer_id

    def publish(self, producer_id, product):
        

        
        self.slots_locks[producer_id].acquire()
        if self.producers_slots[producer_id] == 0:
            self.logger.info('Publish product: producer_id = %s, product = %s, \
                            return = False', producer_id, product)
            self.slots_locks[producer_id].release()
            return False
        self.slots_locks[producer_id].release()

        
        self.queues_locks[producer_id].acquire()
        self.producers_queues[producer_id].append(product)
        self.queues_locks[producer_id].release()

        
        self.slots_locks[producer_id].acquire()
        self.producers_slots[producer_id] -= 1
        self.slots_locks[producer_id].release()

        self.logger.info('Publish product: producer_id = %s, product = %s, \
                        return = True ', producer_id, product)
        return True

    def new_cart(self):
        

        
        self.cart_id_lock.acquire()
        cart_id = self.next_cart_id
        self.next_cart_id = self.next_cart_id + 1
        self.cart_id_lock.release()

        self.carts[cart_id] = {}

        self.logger.info('New cart: cart_id = %d', cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        

        


        for producer in self.producers_queues:
            
            self.queues_locks[producer].acquire()
            if product in self.producers_queues[producer]:
                self.producers_queues[producer].remove(product)
                self.queues_locks[producer].release()

                
                if not producer in self.carts[cart_id].keys():
                    self.carts[cart_id][producer] = []
                self.carts[cart_id][producer].append(product)

                self.logger.info('Add to cart: cart_id = %d, product = %s, \
                                return = True', cart_id, product)
                return True
            self.queues_locks[producer].release()

        
        self.logger.info('Add to cart: cart_id = %d, product = %s, \
                        return = False', cart_id, product)
        return False

    def remove_from_cart(self, cart_id, product):
        
        for producer in self.carts[cart_id]:
            if product in self.carts[cart_id][producer]:
                
                self.carts[cart_id][producer].remove(product)
                
                self.queues_locks[producer].acquire()
                self.producers_queues[producer].append(product)
                self.queues_locks[producer].release()

                self.logger.info('Remove from cart: cart_id = %d, product = %s', cart_id, product)
                break

    def place_order(self, cart_id):
        
        returned_list = []

        
        for producer in self.carts[cart_id]:
            for product in self.carts[cart_id][producer]:
                returned_list.append(product)
                
                
                self.slots_locks[producer].acquire()
                self.producers_slots[producer] += 1
                self.slots_locks[producer].release()

        
        del self.carts[cart_id]

        self.logger.info('Place order: cart_id = %d,returned list = %s', cart_id, list)
        return returned_list


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(3)
        self.products = []
        self.products.append(Tea('Musetel', 1, 'Herbal'))
        self.products.append(Tea('Coada Soricelului', 3, 'Herbal'))
        self.products.append(Coffee('Espresso', 2, '10.0', 'HIGH'))
        self.products.append(Tea('Urechea boului', 4, 'Non-herbal'))

    def test_register_producer(self):
        
        for i in range(0, 10):
            self.assertEqual(self.marketplace.register_producer(), str(i))

    def test_new_cart(self):
        
        for i in range(0, 10):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_publish(self):
        
        for _ in range(0, 4):
            self.marketplace.register_producer()

        self.marketplace.publish('0', self.products[0])
        self.marketplace.publish('0', self.products[1])
        self.marketplace.publish('2', self.products[0])
        self.marketplace.publish('0', self.products[0])
        self.marketplace.publish('1', self.products[3])

        self.assertEqual(self.marketplace.producers_queues['0'],
                         [self.products[0], self.products[1], self.products[0]])
        self.assertEqual(self.marketplace.producers_queues['1'], [self.products[3]])
        self.assertEqual(self.marketplace.producers_queues['2'], [self.products[0]])
        self.assertEqual(self.marketplace.producers_queues['3'], [])

    def test_publish_fails(self):
        
        self.marketplace.register_producer()
        for i in range(0, 3):
            self.assertTrue(self.marketplace.publish(str(0), self.products[i]))
        self.assertFalse(self.marketplace.publish(str(0), self.products[0]))

    def test_add_to_cart(self):
        

        
        for i in range(0, 3):
            self.marketplace.new_cart()
            self.marketplace.register_producer()
        for i in range(0, 3):
            for _ in range(0, 3):
                self.marketplace.publish(str(i), self.products[i])

        
        for i in range(0, 3):
            for _ in range(0, 3):
                self.assertTrue(self.marketplace.add_to_cart(i, self.products[i]))

        
        for i in range(0, 3):
            self.assertFalse(self.marketplace.add_to_cart(i, self.products[i]))
        
        for i in range(0, 3):
            self.assertEqual(self.marketplace.carts[i][str(i)],
                             [self.products[i], self.products[i], self.products[i]])

    def test_remove_from_cart(self):
        
        id_cart = self.marketplace.new_cart()
        id_producer = self.marketplace.register_producer()
        for i in range(0, 3):
            self.marketplace.publish('0', self.products[i])
            self.marketplace.add_to_cart(0, self.products[i])

        self.marketplace.remove_from_cart(id_cart, self.products[1])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer],
                         [self.products[0], self.products[2]])
        self.marketplace.remove_from_cart(id_cart, self.products[0])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer], [self.products[2]])
        self.marketplace.remove_from_cart(id_cart, self.products[2])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer], [])

    def test_place_order(self):
        

        
        cart_id = self.marketplace.new_cart()
        producer_id = self.marketplace.register_producer()
        for i in range(0, 3):
            self.marketplace.publish(producer_id, self.products[i])
            self.marketplace.add_to_cart(cart_id, self.products[i])

        returned_list = self.marketplace.place_order(cart_id)
        
        self.assertEqual(returned_list, [self.products[0], self.products[1], self.products[2]])
        
        self.assertEqual(self.marketplace.producers_slots[producer_id], 3)


import time
from threading import Thread



class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        

        producer_id = self.marketplace.register_producer()
        
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    time.sleep(product[2])
                    success = self.marketplace.publish(producer_id, product[0])
                    while not success:
                        time.sleep(self.republish_wait_time)
                        success = self.marketplace.publish(producer_id, product[0])


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
