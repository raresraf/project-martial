


from threading import Thread
from time import sleep


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
            for ops in cart:
                if ops['type'] == "add":
                    for _ in range(0, ops['quantity']):
                        
                        while self.marketplace.add_to_cart(cart_id, ops['product']) is not True:
                            sleep(self.retry_wait_time)
                else:
                    for _ in range(0, ops['quantity']):
                        self.marketplace.remove_from_cart(cart_id, ops['product'])
            products = self.marketplace.place_order(cart_id)

            lock = self.marketplace.get_consumer_lock()

            lock.acquire()
            for product in products:
                print(self.kwargs['name'] + " bought " + str(product))
            lock.release()


import logging
from logging.handlers import RotatingFileHandler
import time
from threading import Lock
import unittest

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO,
                    handlers=[RotatingFileHandler('marketplace.log',
                                                  maxBytes=20000, backupCount=10)])
logging.Formatter.converter = time.gmtime

class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.producer_id = -1
        self.cart_id = -1
        self.size_per_producer = {}
        self.carts = {}
        self.products_dict = {}

    def register_producer(self):


        
        self.producer_lock.acquire()
        logging.info("New producer entered register_producer method")
        self.producer_id += 1
        self.size_per_producer[self.producer_id] = 0


        self.producer_lock.release()
        logging.info("New producer registered with id %d", self.producer_id)
        return self.producer_id

    def publish(self, producer_id, product):
        
        logging.info("Producer with id %d entered publish method", producer_id)

        self.producer_lock.acquire()
        
        if self.size_per_producer[producer_id] == self.queue_size_per_producer:
            logging.info(f"Producer with id {producer_id} failed to publish product {product}")
            self.producer_lock.release()
            return False
        if product not in self.products_dict:
            self.products_dict[product] = [producer_id]
        else:
            self.products_dict[product].append(producer_id)

        self.size_per_producer[producer_id] += 1
        logging.info(f"Producer with id {producer_id} published product {product}")
        self.producer_lock.release()
        return True

    def new_cart(self):


        

        self.consumer_lock.acquire()
        logging.info("Consumer entered new_cart method")
        self.cart_id += 1
        self.carts[self.cart_id] = {}
        logging.info("Consumer registered new cart with id %d", self.cart_id)
        self.consumer_lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered add_to_cart method", cart_id)
        if product in self.products_dict:
            
            producer_id = self.products_dict[product].pop(0)
            if product in self.carts[cart_id]:
                self.carts[cart_id][product].append(producer_id)
            else:
                self.carts[cart_id][product] = [producer_id]
            
            
            if len(self.products_dict[product]) == 0:
                del self.products_dict[product]

            logging.info(f"Consumer with card id {cart_id} added product {product} to cart")
            self.consumer_lock.release()
            return True


        logging.info(f"Consumer with card id {cart_id} failed to add product {product} to cart")
        self.consumer_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered remove_from_cart method", cart_id)

        
        
        given_id = self.carts[cart_id][product].pop(0)
        if len(self.carts[cart_id][product]) == 0:
            del self.carts[cart_id][product]

        if product not in self.products_dict:
            self.products_dict[product] = [given_id]
        else:
            self.products_dict[product].append(given_id)
        logging.info(f"Consumer with card id {cart_id} removed product {product} from cart")
        self.consumer_lock.release()

    def place_order(self, cart_id):
        
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered place_order method", cart_id)
        
        
        
        products = []
        for product in self.carts[cart_id]:
            for given_id in self.carts[cart_id][product]:
                self.size_per_producer[given_id] -= 1
                products.append(product)
        logging.info("Consumer with card id %d placed order", cart_id)
        self.consumer_lock.release()
        return products

    def get_consumer_lock(self):
        
        logging.info("A consumer entered get_consumer_lock method")
        return self.consumer_lock


class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 0)

    def test_true_publish(self):
        
        producer_id = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(producer_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 1)
        self.assertEqual(len(self.marketplace.products_dict["id1"]), 1)

    def test_false_publish(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.assertFalse(self.marketplace.publish(producer_id, "id1"))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)

    def test_true_add_to_cart(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        cart_id = self.marketplace.new_cart()

        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 0)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)
        self.assertEqual(len(self.marketplace.carts[cart_id]["id1"]), 1)

        self.marketplace.publish(producer_id, "id1")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))
        self.assertEqual(len(self.marketplace.carts[cart_id]["id1"]), 2)

    def test_false_add_to_cart(self):
        
        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, "id1"))

    def test_remove_from_cart(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id2")

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")
        self.marketplace.add_to_cart(cart_id, "id2")
        self.assertEqual(len(self.marketplace.products_dict), 1)

        self.marketplace.remove_from_cart(cart_id, "id1")
        self.assertEqual(len(self.marketplace.products_dict), 2)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)
        self.assertFalse("id1" in self.marketplace.carts[cart_id])

        self.marketplace.remove_from_cart(cart_id, "id2")
        self.assertEqual(len(self.marketplace.products_dict), 2)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 0)
        self.assertFalse("id2" in self.marketplace.carts[cart_id])

    def test_place_order(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.assertEqual(self.marketplace.size_per_producer[producer_id], 2)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")

        expected_products = ["id1"]
        products = self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.size_per_producer[producer_id], 1)
        self.assertEqual(expected_products, products)

    def test_get_consumer_lock(self):
        


        self.assertEqual(self.marketplace.consumer_lock, self.marketplace.get_consumer_lock())


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
            for product in self.products:
                for _ in range(0, product[1]):
                    
                    while self.marketplace.publish(producer_id, product[0]) is not True:
                        sleep(self.republish_wait_time)
                    sleep(product[2])
