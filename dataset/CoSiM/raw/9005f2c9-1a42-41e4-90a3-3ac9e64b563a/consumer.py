

from threading import Thread
import time


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

                
                for _ in range(action['quantity']):
                    if action['type'] == "add":
                        return_value = self.marketplace.add_to_cart(cart_id, action['product'])
                    else:
                        return_value = self.marketplace\
                            .remove_from_cart(cart_id, action['product'])
                    time.sleep(self.retry_wait_time)

                    while return_value == False:
                        time.sleep(self.retry_wait_time)

                        if action['type'] == "add":
                            return_value = self.marketplace\
                                .add_to_cart(cart_id, action['product'])
                        else:
                            return_value = self.marketplace\
                                .remove_from_cart(cart_id, action['product'])

            
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock


import unittest
import time
import logging
from logging.handlers import RotatingFileHandler
logging.basicConfig(
        handlers=[RotatingFileHandler('./marketplace.log', maxBytes=150000, backupCount=15)],
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = time.gmtime

class Producer:
    
    def __init__(self, producer_id, nr_items):
        
        self.producer_id = producer_id 
        self.nr_items = nr_items 

class Product:
    
    def __init__(self, details, producer_id, quantity):
        
        self.details = details 
        self.producer_id = producer_id 
        self.quantity = quantity 

class Cart:
    
    def __init__(self, cart_id, products):
        
        self.cart_id = cart_id 
        self.products = products 

def get_index_of_product(product, list_of_products):
    
    idx = 0
    for element in list_of_products:
        if product == element.details:
            return idx
        idx += 1
    return -1

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.limit_per_producer = queue_size_per_producer  
        self.producers = [] 
        self.products = [] 
        self.carts = [] 
        self.new_producer_lock = Lock()
        self.new_cart_lock = Lock()
        self.add_to_cart_lock = Lock()
        self.remove_from_cart_lock = Lock()
        self.publish_lock = Lock()
        self.pint_lock = Lock()



    def register_producer(self):
        
        with self.new_producer_lock:
            
            self.producers.append(Producer(len(self.producers), 0))
            logging.info(f'FROM "register_producer" ->'
                         f' output: producer_id = {len(self.producers) - 1}')
            return len(self.producers) - 1

    def publish(self, producer_id, product):
        
        logging.info(f'FROM "publish" ->'
                     f' input: producer_id = {producer_id}, product = {product}')

        
        if self.producers[producer_id].nr_items >= self.limit_per_producer:
            logging.info(f'FROM "publish" ->'
                         f' output: False')
            return False

        with self.publish_lock:
            
            self.producers[producer_id].nr_items += 1

            
            idx_product = get_index_of_product(product, self.products)
            if idx_product == -1:
                self.products.append(Product(product, producer_id, 1))
            else:
                self.products[idx_product].quantity += 1
                self.products[idx_product].producer_id = producer_id

        logging.info(f'FROM "publish" ->'
                     f' output: True')
        return True


    def new_cart(self):
        
        with self.new_cart_lock:
            
            self.carts.append(Cart(len(self.carts), []))
            logging.info(f'FROM "new_cart" ->'
                         f' output: cart_id: {len(self.carts) - 1}')
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        
        logging.info(f'FROM "add_to_cart" ->'
                     f' input: cart_id = {cart_id}, product = {product}')

        idx_product = get_index_of_product(product, self.products)
        
        if idx_product == -1:
            logging.info(f'FROM "add_to_cart" ->'
                         f' output: False')
            return False
        
        if self.products[idx_product].quantity == 0:
            logging.info(f'FROM "add_to_cart" ->'
                         f' output: False')
            return False

        with self.add_to_cart_lock:
            
            idx_producer = self.products[idx_product].producer_id
            self.producers[idx_producer].nr_items -= 1

            
            self.products[idx_product].quantity -= 1

        
        self.carts[cart_id].products.append(product)

        logging.info(f'FROM "add_to_cart" ->'
                     f' output: True')
        return True

    def remove_from_cart(self, cart_id, product):
        
        logging.info(f'FROM "remove_from_cart" ->'
                     f' input: cart_id = {cart_id}, product = {product}')

        
        self.carts[cart_id].products.remove(product)
        with self.remove_from_cart_lock:
            
            idx_product = get_index_of_product(product, self.products)
            self.products[idx_product].quantity += 1

            
            idx_producer = self.products[idx_product].producer_id
            self.producers[idx_producer].nr_items += 1

    def place_order(self, cart_id):
        
        logging.info(f'FROM "place_order" ->'
                     f' input: cart_id = {cart_id}')

        self.pint_lock.acquire()
        for product in self.carts[cart_id].products:
            print(f'{currentThread().name} bought {product}')
        self.pint_lock.release()

        logging.info(f'FROM "place_order" ->'
                     f' input: cart_id = {self.carts[cart_id].products}')
        return self.carts[cart_id].products

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), len(self.marketplace.producers) - 1)
        
        self.assertEqual(len(self.marketplace.producers), 1)

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), len(self.marketplace.carts) - 1)
        
        self.assertEqual(len(self.marketplace.carts), 1)

    def test_publish_success(self):
        
        self.marketplace.register_producer()
        product = "product1"
        self.assertEqual(self.marketplace.publish(0, product), True)
        
        self.assertEqual(len(self.marketplace.products), 1)

    def test_publish_fail(self):
        
        self.marketplace.register_producer()
        self.marketplace.producers[0].nr_items = 5
        product = "product1"
        self.assertEqual(self.marketplace.publish(0, product), False)

    def test_add_to_cart_success(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.assertEqual(self.marketplace.add_to_cart(0, product), True)

    def test_add_to_cart_fail_case1(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        nonexistent_product = "nonexistent_product"
        self.assertEqual(self.marketplace.add_to_cart(0, nonexistent_product), False)

    def test_add_to_cart_fail_case2(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.marketplace.products[0].quantity = 0
        self.assertEqual(self.marketplace.add_to_cart(0, product), False)

    def test_remove_from_cart(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.marketplace.add_to_cart(0, product)
        self.marketplace.remove_from_cart(0, product)
        self.assertEqual(self.marketplace.carts[0].products, [])

    def test_place_order(self):
        
        self.marketplace.new_cart()
        products_sample = ["prod1", "prod2", "prod3"]
        self.marketplace.carts[0].products = products_sample
        self.assertEqual(self.marketplace.place_order(0), products_sample)


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):

        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):

        id_producer = self.marketplace.register_producer()
        while True:
            for i in range(len(self.products)):
                product_id = self.products[i][0]
                how_many = self.products[i][1]
                wait_time = self.products[i][2]
                for _ in range(how_many):
                    while True:
                        return_value = self.marketplace.publish(id_producer, product_id)
                        if return_value:
                            time.sleep(wait_time)
                            break
                        time.sleep(self.republish_wait_time)
