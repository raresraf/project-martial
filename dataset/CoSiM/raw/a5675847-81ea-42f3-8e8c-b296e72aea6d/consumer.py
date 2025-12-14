


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_to_cart(self, quantity, cart_id, product):
        
        i = 0
        while i < quantity:
            added_ok = self.marketplace.add_to_cart(cart_id, product)
            if added_ok:
                i = i + 1
            else:
                time.sleep(self.retry_wait_time)

    def remove_from_cart(self, quantity, cart_id, product):
        
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        
        cart_id = self.marketplace.new_cart()
        for cart_list in self.carts:
            for cart_event in cart_list:
                if cart_event["type"] == "add":
                    self.add_to_cart(cart_event["quantity"], cart_id, cart_event["product"])
                else:
                    self.remove_from_cart(cart_event["quantity"], cart_id, cart_event["product"])
        for product in self.marketplace.place_order(cart_id):
            print(self.name, "bought", product)


from threading import Lock
import unittest
import time
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_peroducer = queue_size_per_producer
        self.products = []
        self.carts = []
        self.product_in_cart = {}
        self.lock_cart = Lock()
        self.lock_producer = Lock()
        self.logger = logging.getLogger('marketplace')
        handler = RotatingFileHandler('marketplace.log', maxBytes=4096, backupCount=10)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        logging.Formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel("INFO")

    def register_producer(self):
        
        self.logger.info("Method register_producer started")
        self.lock_producer.acquire()
        self.products.append([])
        ret = len(self.products) - 1
        self.lock_producer.release()
        self.logger.info("Method register_producer returned " + str(ret))
        return ret

    def publish(self, producer_id, product):
        

        self.logger.info("Method publish started")
        self.logger.info("producer_id = " + str(producer_id))
        self.logger.info("product = " + str(product))
        self.lock_producer.acquire()
        if len(self.products[producer_id]) < self.queue_size_per_peroducer:
            self.products[producer_id].append(product)
            self.product_in_cart[product] = False
            self.lock_producer.release()
            self.logger.info("New product published to marketplace")
            return True

        self.lock_producer.release()
        self.logger.info("Method publish returned False")
        return False

    def new_cart(self):
        

        self.logger.info("Method new_cart started")
        self.lock_cart.acquire()
        self.carts.append([])
        ret = len(self.carts) - 1
        self.lock_cart.release()
        self.logger.info("Method new_cart returned " + str(ret))
        return ret

    def add_to_cart(self, cart_id, product):
        

        self.logger.info("Method add_to_cart started")
        self.logger.info("cart_id = " + str(cart_id))
        self.logger.info("product = " + str(product))
        if product in self.product_in_cart.keys() and not self.product_in_cart[product]:
            self.carts[cart_id].append(product)
            self.product_in_cart[product] = True
            self.logger.info("New product added to cart " + str(cart_id))
            return True

        self.logger.info("Method add_to_cart returned False")
        return False

    def remove_from_cart(self, cart_id, product):
        

        self.logger.info("Method remove_from_cart started")
        self.logger.info("cart_id = " + str(cart_id))
        self.logger.info("product = " + str(product))
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)


            self.product_in_cart[product] = False
            self.logger.info("Product removed from cart")

    def place_order(self, cart_id):
        

        self.logger.info("Method place_order started")
        self.logger.info("cart_id = " + str(cart_id))
        for cart_product in self.carts[cart_id]:
            for prod_products in self.products:
                if cart_product in prod_products:
                    prod_products.remove(cart_product)
        self.logger.info("Method place_order returned " + str(self.carts[cart_id]))
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        self.marketplace = Marketplace(15)
        self.products = [Coffee("Espresso", 7, 4.00, "MEDIUM"), \
                        Coffee("Irish", 10, 5.00, "MEDIUM"), \
                        Tea("Black", 10, "Green")]

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_publish(self):
        
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(0, self.products[0]))
        self.assertTrue(self.marketplace.publish(0, self.products[1]))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[0]))
        self.assertEqual(len(self.marketplace.carts[0]), 1)
        self.assertTrue(self.marketplace.add_to_cart(0, self.products[1]))
        self.assertEqual(len(self.marketplace.carts[0]), 2)

    def test_remove_from_cart(self):
        
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[1])
        self.marketplace.remove_from_cart(0, self.products[2])
        self.assertEqual(len(self.marketplace.carts[0]), 2)
        self.marketplace.remove_from_cart(0, self.products[0])
        self.assertEqual(len(self.marketplace.carts[0]), 1)

    def test_place_order(self):
        
        self.marketplace.new_cart()
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.products[0])
        self.marketplace.publish(0, self.products[1])
        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.add_to_cart(0, self.products[1])
        self.assertEqual(self.marketplace.place_order(0), [self.products[0], self.products[1]])


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        
        while True:
            producer_id = self.marketplace.register_producer()
            for product in self.products:
                i = 0
                num_of_products = product[1]
                curr_product = product[0]
                curr_product_wait_time = product[2]
                while i < num_of_products:
                    published_ok = self.marketplace.publish(producer_id, curr_product)
                    if published_ok:
                        i += 1
                        time.sleep(curr_product_wait_time)
                    else:
                        time.sleep(self.republish_wait_time)
