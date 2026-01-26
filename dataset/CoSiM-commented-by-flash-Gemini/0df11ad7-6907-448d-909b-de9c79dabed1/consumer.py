


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, kwargs=kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']


    def run(self):

        for curr_cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            for curr_op in curr_cart:
                quantity_count = 0

                
                if curr_op["type"] == "add":
                    while curr_op["quantity"] > quantity_count:
                        if self.marketplace.add_to_cart(cart_id, curr_op["product"]):
                            quantity_count += 1
                        else:
                            sleep(self.retry_wait_time)

                
                if curr_op["type"] == "remove":
                    while curr_op["quantity"] > quantity_count:
                        self.marketplace.remove_from_cart(cart_id, curr_op["product"])
                        quantity_count += 1

            
            product_list = self.marketplace.place_order(cart_id)

            with self.marketplace.lock:
                for curr_prod in product_list:
                    print(self.name, "bought", curr_prod)

from threading import Lock
import unittest
import logging
import logging.handlers
import time

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.lock = Lock()

        self.queue_size_per_producer = queue_size_per_producer
        self.no_products = 0
        self.no_carts = 0

        self.product_list = {}
        self.shopping_cart = {}
        self.prodcuer_ids = {}

        self.logger = logging.getLogger("marketLogger")
        self.logger.setLevel(logging.INFO)

        handler = logging.handlers.RotatingFileHandler(filename="marketplace.log", backupCount=10, maxBytes=6000000)

        handler.setLevel(logging.INFO)



        formatter = logging.Formatter('%(asctime)s %(message)s')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def register_producer(self):
        
        self.logger.info("Start of register_producer.")

        
        with self.lock:
            self.no_products += 1
            self.product_list[self.no_products] = []
            self.logger.info("End of register_producer. Value returned: %d.", self.no_products)

            return self.no_products

    def publish(self, producer_id, product):
        

        self.logger.info("Start of publish. Arguments of function: %d, %s.", producer_id, str(product))

        no_products = len(self.product_list[producer_id])

        
        if self.queue_size_per_producer >= no_products:
            self.product_list[producer_id].append(product)
            self.logger.info("End of publish. Value returned: True.")
            return True
        else:
            self.logger.info("End of publish. Value returned: False.")
            return False

    def new_cart(self):
        
        self.logger.info("Start of new_cart.")

        with self.lock:
            self.no_carts += 1
            self.shopping_cart[self.no_carts] = []
            self.logger.info("End of new_cart. Value returned: %d.", self.no_carts)
            return self.no_carts

    def add_to_cart(self, cart_id, product):
        
        self.logger.info("Start of add_to_cart. Arguments of function: %d, %s.", cart_id, str(product))

        is_product = False
        with self.lock:
            for curr_product in self.product_list:
                for aux in self.product_list[curr_product]:
                    
                    if aux[0] == product:
                        is_product = True
                        final_product = (curr_product, product)
                        break

        
        if is_product:
            self.shopping_cart[cart_id].append(final_product)
            self.product_list.pop(final_product[1], None)
            self.logger.info("End of add_to_cart. Value returned: True.")
            return True
        else:
            self.logger.info("End of add_to_cart. Value returned: False.")
            return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("Start of remove_from_cart. Arguments of function: %d, %s.", cart_id, str(product))

        
        for curr_product in self.shopping_cart[cart_id]:
            if curr_product[1] == product:
                self.shopping_cart[cart_id].remove(curr_product)
                self.logger.info("End of remove_from_cart.")
                return

    def place_order(self, cart_id):
        
        self.logger.info("Start of remove_from_cart. Argument of function: %d.", cart_id)

        cart_products = []

        
        for curr_product in self.shopping_cart[cart_id]:
            cart_products.append(curr_product[1])

        self.shopping_cart.pop(cart_id)
        self.logger.info("End of place_order. Value returned: %s.", str(cart_products))

        return cart_products

class TestMarketplace(unittest.TestCase):
    def setUp(self):
        queue_size_per_producer = 5
        republish_wait_time = 5
        retry_wait_time = 5

        products = {
            "id1": {
                "product_type": "Coffee",
                "name": "Indonezia",
                "acidity": 5.05,
                "roast_level": "MEDIUM",
                "price": 1
            },
            "id2": {
                "product_type": "Tea",
                "name": "Linden",
                "type": "Herbal",
                "price": 9
            }
        }



        self.marketplace = Marketplace(queue_size_per_producer)

    def test_register_producer(self):
        recv_id = self.marketplace.register_producer()
        self.assertNotEqual(recv_id, 0)

    def test_publish(self):
        product1 = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        recv_id = self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(recv_id, product1), True)


    def test_new_cart(self):
        recv_id = self.marketplace.new_cart()
        self.assertNotEqual(recv_id, 0)

    def test_add_to_cart(self):
        is_in_cart = 0
        product1 = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product1)
        if product1 in self.marketplace.shopping_cart[cart_id]:
            is_in_cart = 1
        self.assertNotEqual(is_in_cart, 1)

    def test_remove_from_cart(self):
        is_in_cart = 0
        product1 = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product1)
        self.marketplace.remove_from_cart(cart_id, product1)
        if product1 in self.marketplace.shopping_cart[cart_id]:
            is_in_cart = 1
        self.assertEqual(is_in_cart, 0)

    def test_place_order(self):
        is_empty_list = 1
        product1 = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, product1)

        ret_list = self.marketplace.place_order(cart_id)
        if ret_list == []:
            is_empty_list = 0

        self.assertEqual(is_empty_list, 0)


from time import sleep
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, kwargs=kwargs, daemon=True)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        
        id = self.marketplace.register_producer()

        for curr_product in self.products:
            
            
            while self.marketplace.publish(id, curr_product) == False:
                sleep(self.republish_wait_time)
            
            sleep(self.republish_wait_time)


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
