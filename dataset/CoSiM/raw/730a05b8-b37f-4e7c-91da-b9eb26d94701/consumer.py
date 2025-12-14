


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)


    def run(self):
        
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for cart_item in cart:
                count = 0
                quantity = cart_item["quantity"]
                action = cart_item["type"]
                product = cart_item["product"]

                while count < quantity:
                    
                    if action == "add":
                        
                        add = self.marketplace.add_to_cart(cart_id, product)
                        if add is True:
                            count += 1
                        else:
                            time.sleep(self.retry_wait_time)
                    
                    else:
                        self.marketplace.remove_from_cart(cart_id, product)
                        count += 1

            
            for order_product in self.marketplace.place_order(cart_id):
                print(self.name + " bought " + str(order_product))


import unittest
from threading import Lock
import logging
import time
from logging.handlers import RotatingFileHandler
import os
from .product import Tea, Coffee

logging.basicConfig(filename="marketplace.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logging.Formatter.converter = time.gmtime
logger = logging.getLogger()

handler = RotatingFileHandler("marketplace.log", mode='w', backupCount=10)



if os.path.isfile("marketplace.log"):
    handler.doRollover()

logger.setLevel(logging.DEBUG)


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = []
        self.products = {}
        self.carts = []

        self.producers_lock = Lock()


        self.carts_lock = Lock()

    def register_producer(self):
        
        logger.info("Entered register_producer")
        with self.producers_lock:
            producer_id = len(self.producers)
            self.producers.append([])
            self.products[str(producer_id)] = []
            logger.info("Exited register_producer with id %d", producer_id)
            return producer_id

    def publish(self, producer_id, product):
        
        logger.info("Entered publish with producer_id %s and product %s",
                    producer_id, str(product))
        if len(self.producers[int(producer_id)]) < self.queue_size_per_producer:
            
            self.producers[int(producer_id)].append(product)
            
            self.products[producer_id].append(product)
            logger.info("Exited publish")
            return True

        logger.info("Exited publish")
        return False

    def new_cart(self):
        

        logger.info("Entered new_cart")
        with self.carts_lock:
            
            cart_id = len(self.carts)
            self.carts.append({})
            logger.info("Exited new_cart with id %d", cart_id)
            return cart_id

    def add_to_cart(self, cart_id, product):
        
        logger.info("Entered add_to_cart with cart_id %d and product %s", cart_id, product)
        for producer_id, prods in self.products.items():
            if product in prods:
                
                prods.remove(product)
                
                if producer_id in self.carts[cart_id]:
                    self.carts[cart_id][producer_id].append(product)
                else:
                    self.carts[cart_id][producer_id] = []


                    self.carts[cart_id][producer_id].append(product)
                logger.info("Exited add_to_cart")
                return True
        logger.info("Exited add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        
        logger.info("Entered remove_from_cart with cart_id %d and product %s", cart_id, product)
        for producer_id in self.carts[cart_id]:
            if product in self.carts[cart_id][producer_id]:
                
                self.carts[cart_id][producer_id].remove(product)
                self.products[producer_id].append(product)
                logger.info("Exited remove_from_cart")
                break

    def place_order(self, cart_id):
        
        logger.info("Entered place_order with cart_id %d", cart_id)
        order = []
        for producer_id in self.carts[cart_id]:
            
            order = order + self.carts[cart_id][producer_id]
            for product in self.carts[cart_id][producer_id]:
                
                self.producers[int(producer_id)].remove(product)
        logger.info("Exited place_order")
        return order


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        
        self.marketplace = Marketplace(20)
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 2,
                         'Error: wrong producer id')
        self.assertEqual(self.marketplace.producers[0], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(self.marketplace.producers[1], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(self.marketplace.producers[2], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(len(self.marketplace.producers[0]), 0,
                         'Error: wrong number of products for the producer')
        self.assertEqual(len(self.marketplace.producers[1]), 0,
                         'Error: wrong number of products for the producer')
        self.assertEqual(len(self.marketplace.producers[2]), 0,
                         'Error: wrong number of products for the producer')

    def publish(self):
        
        self.assertEqual(self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea")), True,
                         'Error: wrong published item')
        self.assertEqual(self.marketplace.publish("1", Coffee("Colombia", 7, "5.05", "HIGH")),
                         True, 'Error: wrong published item')

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 2,
                         'Error: wrong cart id')

    def test_add_to_cart(self):
        
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("Colombia", 7, "5.05", "HIGH")),
                         True, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW")),
                         True, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("Colombia", 7, "5.05", "HIGH")),
                         False, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(1, Tea("Twinings", 7, "Black Tea")), True,
                         'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(1, Tea("Twinings", 7, "Black Tea")), False,
                         'Error: wrong product added to cart')

    def test_remove_from_cart(self):
        
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.marketplace.add_to_cart(0, Tea("Twinings", 7, "Black Tea"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("Colombia", 7, "5.05", "HIGH"))

        self.assertEqual(self.marketplace.remove_from_cart(0, Tea("Twinings", 7, "Black Tea")),
                         None, 'Error: wrong product removed from cart')
        self.assertEqual(self.marketplace.remove_from_cart(1, Coffee("India", 7, "5.05", "LOW")),
                         None, 'Error: wrong product removed from cart')
        self.assertEqual(self.marketplace.remove_from_cart(1, Coffee("Colombia", 7, "5.05",
                                                                     "HIGH")),
                         None, 'Error: wrong product removed from cart')

    def test_place_order(self):
        
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("1", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("1", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.marketplace.add_to_cart(0, Tea("Twinings", 7, "Black Tea"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("Colombia", 7, "5.05", "HIGH"))

        order0 = [Tea("Twinings", 7, "Black Tea"), Coffee("India", 7, "5.05", "LOW"),
                  Coffee("India", 7, "5.05", "LOW")]

        order1 = [Coffee("India", 7, "5.05", "LOW"), Coffee("Colombia", 7, "5.05", "HIGH")]

        self.assertEqual(self.marketplace.place_order(0), order0,
                         'Error: wrong order')

        self.assertEqual(self.marketplace.place_order(1), order1,
                         'Error: wrong order')

    if __name__ == '__main__':
        unittest.main()


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)
        self.producer_id = self.marketplace.register_producer()


    def run(self):
        while True:
            for (prod, quant, w_time) in self.products:
                count = 0
                while count < quant:
                    if self.marketplace.publish(str(self.producer_id), prod):
                        count += 1
                        time.sleep(w_time)
                    else:
                        time.sleep(self.republish_wait_time)
