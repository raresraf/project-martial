


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            
            for action in cart:
                curr_quantity = 0

                
                while curr_quantity < action["quantity"]:
                    if action["type"] == "add":
                        
                        if self.marketplace.add_to_cart(cart_id, action["product"]):
                            curr_quantity += 1
                        else:
                            
                            sleep(self.retry_wait_time)
                    elif action["type"] == "remove":
                        
                        self.marketplace.remove_from_cart(cart_id, action["product"])
                        curr_quantity += 1

            
            self.marketplace.place_order(cart_id)

import time
import unittest
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock, currentThread


from tema.consumer import Consumer
from tema.producer import Producer
from tema.product import Coffee, Tea


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        


        self.nr_producers = -1

        
        self.nr_carts = -1

        
        self.queues = []

        
        self.products = []

        
        self.carts = []

        self.producer_lock = Lock()
        self.cart_lock = Lock()
        self.cart_add_lock = Lock()
        self.place_order_lock = Lock()

        
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        
        logging.Formatter.converter = time.gmtime

        self.logger = logging.getLogger("marketplace_logger")

        
        handler = RotatingFileHandler("file.log", maxBytes=5000, backupCount=15)
        self.logger.addHandler(handler)

        
        self.logger.propagate = False

    def register_producer(self):
        
        
        
        with self.producer_lock:
            self.nr_producers += 1

            
            self.queues.append([])

            self.logger.info(f'register_producer output: producer_id={self.nr_producers}')

        
        return self.nr_producers

    def publish(self, producer_id, product):
        
        self.logger.info(f'publish input: producer_id={producer_id}, product={product}')

        p_id = int(producer_id)

        
        if len(self.queues[p_id]) == self.queue_size_per_producer:
            self.logger.info("publish output: FALSE")
            return False

        
        self.products.append(product)
        self.queues[p_id].append(product)

        self.logger.info("publish output: TRUE")
        return True

    def new_cart(self):
        
        
        
        with self.cart_lock:
            self.nr_carts += 1

            
            self.carts.append([])

            self.logger.info(f'new_cart output: cart_id={self.nr_carts}')

        
        return self.nr_carts

    def add_to_cart(self, cart_id, product):
        
        self.logger.info(f'add_to_cart input: cart_id={cart_id}, product={product}')

        
        
        with self.cart_add_lock:
            
            if product not in self.products:
                self.logger.info("add_to_cart output: FALSE")
                
                return False

            
            self.products.remove(product)

            
            self.carts[cart_id].append(product)

            self.logger.info("add_to_cart output: TRUE")

        
        return True

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info(f'remove_from_cart input: cart_id={cart_id}, product={product}')

        
        self.products.append(product)

        
        self.carts[cart_id].remove(product)

    def remove_from_queue(self, product):
        
        self.logger.info(f'remove_from_queue input: product={product}')

        
        for producer_queue in self.queues:
            if product in producer_queue:
                producer_queue.remove(product)
                break

    def place_order(self, cart_id):
        
        
        self.logger.info(f'place_order input: cart_id={cart_id}')

        with self.place_order_lock:
            
            
            for product in self.carts[cart_id]:
                self.remove_from_queue(product)
                print(currentThread().name, "bought", product)

        self.logger.info(f'place_order output: cart_list={self.carts[cart_id]}')

        
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(15)
        self.product1 = Coffee("Indonezia", 1, "5.05", "MEDIUM")
        self.product2 = Tea("Linden", 9, "Herbal")


        self.producer = Producer([[self.product1, 2, 0.18],
                                  [self.product2, 1, 0.23]],
                                 self.marketplace,
                                 0.15)
        self.consumer = Consumer([[{"type": "add", "product": self.product2, "quantity": 2},
                                   {"type": "add", "product": self.product1, "quantity": 2},
                                   {"type": "remove", "product": self.product1, "quantity": 1}
                                   ]],
                                 self.marketplace,
                                 0.31)

        self.cart_id = self.marketplace.new_cart()

    def test_register_producer(self):
        
        self.assertEqual(self.producer.producer_id, "0")

    def test_publish(self):
        
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.assertEqual(self.marketplace.products,
                         self.marketplace.queues[int(self.producer.producer_id)])

    def test_new_cart(self):
        
        self.assertEqual(self.cart_id, 0)

    def test_add_to_cart(self):
        
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product2)

        self.assertEqual(self.marketplace.carts[self.cart_id], [self.product1, self.product2])

    def test_remove_from_cart(self):
        
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product2)

        self.marketplace.remove_from_cart(self.cart_id, self.product1)
        self.assertEqual(self.marketplace.carts[self.cart_id], [self.product1, self.product2])

    def test_place_order(self):
        
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.consumer.run()

        self.assertEqual(self.marketplace.queues[int(self.producer.producer_id)],
                         [self.product1])


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products


        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = str(self.marketplace.register_producer())

    def run(self):
        
        
        while True:
            
            for (product, max_products, success_wait_time) in self.products:
                curr_products = 0

                
                while curr_products < max_products:
                    
                    if self.marketplace.publish(self.producer_id, product):
                        
                        curr_products += 1
                        sleep(success_wait_time)
                    else:
                        
                        sleep(self.republish_wait_time)
