


import threading
import time

class Consumer(threading.Thread):
    
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts

        
        self.marketplace = marketplace


        self.consumert_cart_id = self.marketplace.new_cart()

        
        self.retry_wait_time = retry_wait_time

        
        threading.Thread.__init__(self, **kwargs)


    def run(self):
        if self.marketplace is None:
            return

        for cart_entry in self.carts:
            for elem in cart_entry:
                while elem['quantity'] > 0:
                    if elem['type'] == 'add':
                        valid_op = self.marketplace.add_to_cart(
                                                        self.consumert_cart_id,
                                                        elem['product'])
                    else:
                        valid_op = self.marketplace.remove_from_cart(
                                                        self.consumert_cart_id,
                                                        elem['product'])

                    if not valid_op:
                        time.sleep(self.retry_wait_time)
                    else:
                        elem['quantity'] = elem['quantity'] - 1

            products = self.marketplace.place_order(self.consumert_cart_id)
            for product_types in products:
                for product in product_types:
                    print(f'{str(threading.currentThread().getName())} bought {str(product)}')


import collections
import json
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import unittest

class Marketplace:
    
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        handlers=[RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=5)],
        format='%(asctime)s - %(message)s',
        level=logging.INFO)

    def __init__(self, queue_size_per_producer):
        
        self.max_products_allowed = queue_size_per_producer

        self.ticket_nr = 0
        self.products_nr = []

        self.carts_nr = 0
        self.products = collections.defaultdict(list)
        self.cart_products = {}

        self.register_producer_lock = threading.Lock()
        self.new_cart_lock = threading.Lock()
        self.add_to_cart_lock = threading.Lock()

        logging.info('Started Marketplace process.')

    def register_producer(self):
        
        self.register_producer_lock.acquire()

        self.products_nr.append(0)
        self.ticket_nr = self.ticket_nr + 1

        self.register_producer_lock.release()

        logging.info('Registered producer with ID %s.', self.ticket_nr - 1)
        return self.ticket_nr - 1

    def publish(self, producer_id, product):
        

        if self.products_nr[producer_id] >= self.max_products_allowed:
            logging.info('Producer %s published too many products.', producer_id)
            return False

        self.products[product].append(producer_id)
        self.products_nr[producer_id] += 1
        logging.info('Producer %s published product %s.', producer_id, product)
        return True

    def new_cart(self):
        
        
        self.new_cart_lock.acquire()

        self.carts_nr = self.carts_nr + 1

        self.new_cart_lock.release()

        logging.info('Created new cart with ID %s.', self.carts_nr - 1)
        return self.carts_nr - 1

    def add_to_cart(self, cart_id, product):
        
        
        if product not in self.products:
            logging.info('Product %s does not exist on marketplace.', product)
            return False
        if len(self.products[product]) <= 0:
            logging.info('Product %s does not exist on marketplace.', product)
            return False

        
        self.add_to_cart_lock.acquire()

        producer_picked = self.products[product][0]
        self.products_nr[producer_picked] -= 1

        if cart_id not in self.cart_products:
            
            self.cart_products[cart_id] = {}

        
        
        if product in self.cart_products[cart_id]:
            
            self.cart_products[cart_id][product].append(producer_picked)
        else:
            self.cart_products[cart_id][product] = [producer_picked]
        self.products[product].pop(0)

        self.add_to_cart_lock.release()

        logging.info('Product %s was added to cart %s.', product, cart_id)
        return True

    def remove_from_cart(self, cart_id, product):
        

        if cart_id > self.carts_nr:
            logging.info('Cart %s does not exist.', cart_id)
            return False
        if cart_id not in self.cart_products:
            logging.info('Cart %s does not exist.', cart_id)
            return False
        if product not in self.cart_products[cart_id]:
            logging.info('Product %s does not exist in cart %s.', product, cart_id)
            return False
        if len(self.cart_products[cart_id][product]) <= 0:
            logging.info('Product %s does not exist in cart %s.', product, cart_id)
            return False

        
        self.add_to_cart_lock.acquire()

        removed_product = self.cart_products[cart_id][product][0]
        self.products[product].append(removed_product)
        self.products_nr[removed_product] += 1

        
        self.cart_products[cart_id][product].pop(0)

        self.add_to_cart_lock.release()

        return True

    def place_order(self, cart_id):
        
        if cart_id not in self.cart_products:
            return []

        ans = []
        for product in self.cart_products[cart_id]:
            product_nr = len(self.cart_products[cart_id][product])
            products_repeat = []
            for _ in range(product_nr):
                products_repeat.append(product)
            ans.append(products_repeat)

        self.cart_products[cart_id] = {}
        return ans

class TestMarketplace(unittest.TestCase):
    

    max_queue = 3
    def setUp(self):
        self.marketplace = Marketplace(self.max_queue)

    def test_register_producer(self):
        
        producer_id = self.marketplace.register_producer()
        self.assertEqual(producer_id, 0)

        producer_id = self.marketplace.register_producer()
        self.assertEqual(producer_id, 1)

        producer_id = self.marketplace.register_producer()
        self.assertEqual(producer_id, 2)

    def test_publish(self):
        
        product_sample = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        producer_id = self.marketplace.register_producer()

        
        self.assertTrue(self.marketplace.publish(producer_id,
                                                 json.dumps(product_sample)))
        self.assertTrue(self.marketplace.publish(producer_id,
                                                 json.dumps(product_sample)))
        self.assertTrue(self.marketplace.publish(producer_id,
                                                 json.dumps(product_sample)))

        
        self.assertFalse(self.marketplace.publish(producer_id, json.dumps(product_sample)))

    def test_new_cart(self):
        
        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, 0)

        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, 1)

        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, 2)

    def test_add_to_cart(self):
        
        product_sample = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        
        self.assertFalse(self.marketplace.add_to_cart(cart_id, json.dumps(product_sample)))

        
        self.marketplace.publish(producer_id, json.dumps(product_sample))
        self.assertTrue(self.marketplace.add_to_cart(cart_id, json.dumps(product_sample)))

        
        self.assertFalse(self.marketplace.add_to_cart(cart_id, json.dumps(product_sample)))

    def test_remove_from_cart(self):
        
        product_sample = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }

        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, json.dumps(product_sample))

        
        self.assertFalse(self.marketplace.remove_from_cart(cart_id, json.dumps(product_sample)))

        
        self.marketplace.add_to_cart(cart_id, json.dumps(product_sample))
        self.assertTrue(self.marketplace.remove_from_cart(cart_id, json.dumps(product_sample)))

        
        self.assertFalse(self.marketplace.remove_from_cart(cart_id, json.dumps(product_sample)))

    def test_place_order(self):
        
        product_sample = {
            "product_type": "Coffee",
            "name": "Indonezia",
            "acidity": 5.05,
            "roast_level": "MEDIUM",
            "price": 1
        }
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(producer_id, json.dumps(product_sample))
        self.marketplace.publish(producer_id, json.dumps(product_sample))
        self.marketplace.publish(producer_id, json.dumps(product_sample))

        
        self.assertEqual(self.marketplace.place_order(cart_id), [])

        
        self.marketplace.add_to_cart(cart_id, json.dumps(product_sample))
        self.assertEqual(self.marketplace.place_order(cart_id), [json.dumps(product_sample)])

        
        self.marketplace.add_to_cart(cart_id, json.dumps(product_sample))
        self.marketplace.add_to_cart(cart_id, json.dumps(product_sample))

        arr_product = self.marketplace.place_order(cart_id)[0]
        self.assertEqual(arr_product[0], json.dumps(product_sample))
        self.assertEqual(arr_product[1], json.dumps(product_sample))


import threading
import time

class Producer(threading.Thread):
    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        self.products = products

        
        self.marketplace = marketplace

        
        self.republish_wait_time = republish_wait_time

        
        threading.Thread.__init__(self, **kwargs)
        
        self.producer_id = marketplace.register_producer()

    def run(self):
        while True:
            for (typep, remaning_quantity, timep) in self.products:
                while remaning_quantity > 0:
                    if not self.marketplace.publish(self.producer_id, typep):
                        time.sleep(self.republish_wait_time)
                    else:
                        time.sleep(timep)
                        remaning_quantity = remaning_quantity - 1
