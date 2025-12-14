


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        
        products_list = []
        
        cart_ids = []

        
        for list_cart in self.carts:
            cart_id = self.marketplace.new_cart()
            cart_ids.append(cart_id)

            
            for dict_command in list_cart:
                command = dict_command['type']
                prod = dict_command['product']
                quantity = dict_command['quantity']

                
                counter_add = 0
                counter_remove = 0

                if command == 'add':
                    while counter_add < quantity:
                        if not self.marketplace.add_to_cart(cart_id, prod):
                            
                            time.sleep(self.retry_wait_time)
                        else:
                            counter_add += 1
                else:
                    while counter_remove < quantity:
                        self.marketplace.remove_from_cart(cart_id, prod)
                        counter_remove += 1

        
        for cart in cart_ids:
            products_list.extend(self.marketplace.place_order(cart))

        
        self.marketplace.print_lock.acquire()
        for product in products_list:
            print(f'{self.name} bought', product)
        self.marketplace.print_lock.release()

from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea


logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('marketplace.log', maxBytes=3000, backupCount=5)
logger.addHandler(handler)

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        
        self.producer_list = []
        
        self.consumer_carts = []
        
        self.lock_prod = Lock()
        
        self.lock_cons = Lock()
        
        self.producer_id = 0
        
        self.cart_id = 0
        
        self.taken_products = []
        
        self.print_lock = Lock()

    def register_producer(self):


        

        self.lock_prod.acquire()
        logger.info("Producer wants to register")

        
        self.producer_id += 1

        
        self.producer_list.insert(self.producer_id - 1, [])



        logger.info("Producer %d got registered", self.producer_id)
        self.lock_prod.release()

        return self.producer_id

    def publish(self, producer_id, product):
        
        logger.info("Producer %d wants to publish %s", producer_id, product)

        
        if len(self.producer_list[producer_id - 1]) == self.queue_size_per_producer:
            return False

        
        self.producer_list[producer_id - 1].append(product)

        logger.info("Producer %d published %s", producer_id, product)

        return True

    def new_cart(self):


        
        self.lock_cons.acquire()
        logger.info("Consumer requested a cart")

        
        self.cart_id += 1

        
        self.consumer_carts.insert(self.cart_id - 1, [])

        logger.info("Consumer got cart %d", self.cart_id)
        self.lock_cons.release()

        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        self.lock_cons.acquire()
        logger.info("Cart %d wants to add %s", cart_id, product)

        
        copy_lst = list(self.producer_list)

        
        for lst in copy_lst:
            
            if product in lst:
                
                indx = copy_lst.index(lst)

                
                self.consumer_carts[cart_id - 1].append(product)

                
                self.taken_products.append((indx + 1, product))

                
                self.producer_list[indx].remove(product)

                self.lock_cons.release()

                logger.info("Cart %d added %s", cart_id, product)

                return True

        self.lock_cons.release()
        logger.info("Cart %d added %s", cart_id, product)
        return False


    def remove_from_cart(self, cart_id, product):
        
        logger.info("Cart %d wants to remove %s", cart_id, product)

        
        self.consumer_carts[cart_id - 1].remove(product)

        
        for prod in self.taken_products:
            if prod[1] == product:
                self.producer_list[prod[0] - 1].append(product)
                break

        logger.info("Cart %d removed %s", cart_id, product)

    def place_order(self, cart_id):
        
        logger.info("Cart %d placed order", cart_id)

        
        return self.consumer_carts[cart_id - 1]

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        
        self.assertNotEqual(self.marketplace.register_producer(), 0, "Wrong id for producer!")

    def test_publish(self):
        
        actual_product = self.marketplace.publish(self.marketplace.register_producer(), \
                                                    Tea("Linden", 13, "Floral"))
        self.assertTrue(actual_product, "The product should be published!")

    def test_new_cart(self):
        
        self.assertNotEqual(self.marketplace.new_cart(), 0, "Wrong id for cart!")

    def test_add_to_cart(self):
        
        actual_cart_id = self.marketplace.new_cart()
        wanted_product = self.marketplace.add_to_cart(actual_cart_id,\
                                                    Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.assertFalse(wanted_product, "This product should not be added to cart now!")

    def test_remove_from_cart(self):
        
        actual_cart_id = self.marketplace.new_cart()

        self.marketplace.publish(self.marketplace.register_producer(),\
                                    Coffee("Ethiopia", 25, 6.5, "MEDIUM"))

        self.marketplace.add_to_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.remove_from_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))

        self.assertListEqual(self.marketplace.consumer_carts[actual_cart_id - 1], [],\
                                        "Product was not removed!")

    def test_place_order(self):
        
        prod_id = self.marketplace.register_producer()
        self.marketplace.publish(prod_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.publish(prod_id, Tea("Brewstar", 17, "Green"))

        actual_cart_id = self.marketplace.new_cart()

        self.marketplace.add_to_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))
        self.marketplace.add_to_cart(actual_cart_id, Tea("Brewstar", 17, "Green"))
        self.marketplace.remove_from_cart(actual_cart_id, Coffee("Ethiopia", 25, 6.5, "MEDIUM"))

        remainder_list = []
        remainder_list.append(Tea("Brewstar", 17, "Green"))

        self.assertCountEqual(self.marketplace.place_order(actual_cart_id),\
                                    remainder_list, "Wrong order!")


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        
        producer_id = self.marketplace.register_producer()

        
        while True:
            for prod in self.products:
                i = 0

                
                while i < prod[1]:
                    if not self.marketplace.publish(producer_id, prod[0]):
                        
                        time.sleep(self.time)
                    else:
                        i += 1
                        
                        time.sleep(prod[2])
