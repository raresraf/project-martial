


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        
        for i in range(len(self.carts)):
            for j in range(len(self.carts[i])):
                for k in range(self.carts[i][j]["quantity"]):
                    if self.carts[i][j]['type'] == 'add':
                        product = self.carts[i][j]['product']
                        while not self.marketplace.add_to_cart(self.cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif self.carts[i][j]['type'] == 'remove':


                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
        self.marketplace.place_order(self.cart_id)

from threading import Semaphore
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Tea

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0

        self.carts = {}
        self.producers = {}

        self.sem_carts = Semaphore(value=1)
        self.sem_producers = Semaphore(value=1)

        handler = RotatingFileHandler("marketplace.log", maxBytes=500000, backupCount=5)
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def register_producer(self):
        
        self.logger.info("register_producer started")

        self.sem_producers.acquire()
        self.producer_id += 1
        self.sem_producers.release()



        self.producers[self.producer_id] = []

        self.logger.info("register_producer ended")
        return self.producer_id

    def publish(self, producer_id, product):
        
        self.logger.info("publish started")
        self.logger.info("publish parameters: {}, {}".format(producer_id, product))

        if len(self.producers[producer_id]) < self.queue_size_per_producer:


            self.producers[producer_id].append(product)

            self.logger.info("publish ended")
            return True

        self.logger.info("publish ended")
        return False

    def new_cart(self):
        
        self.logger.info("new_cart started")

        self.sem_carts.acquire()
        self.cart_id += 1
        self.sem_carts.release()



        self.carts[self.cart_id] = []

        self.logger.info("register_producer ended")
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        self.logger.info("add_to_cart started")
        self.logger.info("add_to_cart parameters: {}, {}".format(cart_id, product))

        for key, value in self.producers.items():
            self.sem_producers.acquire()
            if product in value:
                value.remove(product)
                self.sem_producers.release()
                self.logger.info("add_to_cart ended")
                self.carts[cart_id].append((product, key))
                return True

            self.sem_producers.release()

        self.logger.info("add_to_cart ended")
        return False

    def remove_from_cart(self, cart_id, product):
        

        self.logger.info("remove_from_cart started")
        self.logger.info("remove_from_cart parameters: {}, {}".format(cart_id, product))

        for i in range(len(self.carts[cart_id])):
            self.sem_carts.acquire()
            if product == self.carts[cart_id][i][0]:
                self.producers[self.carts[cart_id][i][1]].append(product)
                self.carts[cart_id].remove((product, self.carts[cart_id][i][1]))
                self.sem_carts.release()

                self.logger.info("remove_from_cart ended")
                return

            self.sem_carts.release()
        self.logger.info("remove_from_cart ended")

    def place_order(self, cart_id):
        
        self.logger.info("place_order started")
        self.logger.info("place_order parameters: {}".format(cart_id))

        for i in range(len(self.carts[cart_id])):
            self.sem_carts.acquire()
            print("cons{} bought {}".format(cart_id, self.carts[cart_id][i][0]))
            self.sem_carts.release()

        self.logger.info("place_order ended")
        return [elem[0] for elem in self.carts[cart_id]]

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(3)
        self.product = Tea('Green', 2, 'Tea')

    def test_register_producer(self):
        
        iterations = 10

        while iterations > 0:
            producer_id = self.marketplace.register_producer()
            iterations -= 1

        self.assertEqual(producer_id, 10)

    def test_new_cart(self):
        
        iterations = 5

        while iterations > 0:
            cart_id = self.marketplace.register_producer()
            iterations -= 1

        self.assertEqual(cart_id, 5)

    def test_publish(self):
        
        producer = self.marketplace.register_producer()

        iterations = 3
        while iterations > 0:
            ret = self.marketplace.publish(producer, self.product)
            self.assertEqual(ret, True)
            iterations -= 1

        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, False)

    def test_add_to_cart(self):
        
        producer = self.marketplace.register_producer()
        cons = self.marketplace.new_cart()

        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, False)



        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, True)

        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, True)

    def test_place_order(self):
        
        producer = self.marketplace.register_producer()
        cons = self.marketplace.new_cart()

        iterations = 3
        while iterations > 0:
            ret = self.marketplace.publish(producer, self.product)
            self.assertEqual(ret, True)
            iterations -= 1

        iterations = 3
        while iterations > 0:
            ret = self.marketplace.add_to_cart(cons, self.product)
            self.assertEqual(ret, True)
            iterations -= 1

        ret = self.marketplace.place_order(cons)
        expected_ret = [Tea(name='Green', price=2, type='Tea')] * 3
        self.assertEqual(ret, expected_ret)

    def test_remove_from_cart(self):
        
        producer = self.marketplace.register_producer()
        cons = self.marketplace.new_cart()

        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, True)

        ret = self.marketplace.publish(producer, self.product)
        self.assertEqual(ret, True)

        ret = self.marketplace.add_to_cart(cons, self.product)


        self.assertEqual(ret, True)

        ret = self.marketplace.add_to_cart(cons, self.product)
        self.assertEqual(ret, True)

        self.marketplace.remove_from_cart(cons, self.product)

        ret = self.marketplace.place_order(cons)
        expected_ret = [Tea(name='Green', price=2, type='Tea')]
        self.assertEqual(ret, expected_ret)


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.cart_id = self.marketplace.register_producer()

    def run(self):
        
        while True:
            for i in range(len(self.products)):
                for j in range(self.products[i][1]):
                    while not self.marketplace.publish(self.cart_id, self.products[i][0]):
                        time.sleep(self.republish_wait_time)
                    time.sleep(self.products[i][2])
