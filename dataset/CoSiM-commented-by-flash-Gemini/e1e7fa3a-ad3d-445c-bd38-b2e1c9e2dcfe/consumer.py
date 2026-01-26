


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry = retry_wait_time

    def run(self):
        cart_id = self.marketplace.new_cart()
        for cart in self.carts:
            for cart_action in cart:
                if cart_action['type'] == 'add':
                    for _ in range(cart_action['quantity']):
                        done = self.marketplace.add_to_cart(cart_id, cart_action['product'])
                        
                        while not done:
                            sleep(self.retry)
                            done = self.marketplace.add_to_cart(cart_id, cart_action['product'])
                else:
                    for _ in range(cart_action['quantity']):
                        self.marketplace.remove_from_cart(cart_id, cart_action['product'])
            self.marketplace.place_order(cart_id)

from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
from time import gmtime

class Logger:
    
    def __init__(self):
        formatter = logging.Formatter('%(asctime)s-%(message)s')
        formatter.converter = gmtime

        handler = RotatingFileHandler('marketplace.log', maxBytes=2000, backupCount=3)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def info(self, msg):
        
        self.logger.info(msg)

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.logger = Logger()
        self.logger.info(f'Initialized marketplace with max queue:{queue_size_per_producer}')
        self.max_queue_size = queue_size_per_producer
        self.producer_index = 0
        self.cart_index = 0
        self.lock_producer = Lock()
        self.lock_cart = Lock()
        self.producer_lists = []
        self.cart_lists = []

    def register_producer(self):
        
        with self.lock_producer:
            producer_id = self.producer_index
            self.producer_index += 1
            self.producer_lists.append({'lock': Lock(), 'products': []})



        self.logger.info(f'Marketplace returned producer id: {producer_id}')

        return producer_id

    def publish(self, producer_id, product):
        
        self.logger.info(f'Producer {producer_id} tried publishing {product}:')
        producer_dic = self.producer_lists[producer_id]

        producer_dic['lock'].acquire()
        if len(producer_dic['products']) < self.max_queue_size:
            producer_dic['products'].append({'product': product, 'available': True})
            product_published = True
        else:
            product_published = False

        producer_dic['lock'].release()


        self.logger.info(f'Product:{product} published status:{product_published}')

        return product_published

    def new_cart(self):
        
        with self.lock_cart:
            cart_index = self.cart_index
            self.cart_index += 1
            self.cart_lists.append({'lock': Lock(), 'products': []})



        self.logger.info(f'Marketplace returned cart id:{cart_index}')

        return cart_index

    def add_to_cart(self, cart_id, product):
        
        self.logger.info(f'Consumer {cart_id} tried tried to add {product}:')
        added_product = False
        for index in range(len(self.producer_lists)):
            producer = self.producer_lists[index]
            producer['lock'].acquire()

            for prod in producer['products']:
                if (prod['product'] == product) and prod['available']:
                    prod['available'] = False
                    added_product = True
                    break

            producer['lock'].release()
            if added_product:
                self.cart_lists[cart_id]['products'].append((product, index))
                break
        self.logger.info(f'Product:{product} added to cart status:{added_product}')

        return added_product

    def remove_from_cart(self, cart_id, product):
        
        found = False

        with self.cart_lists[cart_id]['lock']:
            for item in self.cart_lists[cart_id]['products']:
                if item[0] == product:
                    self.cart_lists[cart_id]['products'].remove(item)
                    producer_id = item[1]
                    found = True
                    break

        if found:
            
            producer = self.producer_lists[producer_id]
            with producer['lock']:
                for prod in producer['products']:
                    if prod['product'] == product:
                        prod['available'] = True
                        break

            self.logger.info(f'Consumer removed {product}')
        else:


            self.logger.info(f'Product :{product} was not found in cart:{cart_id}')

    def place_order(self, cart_id):
        
        final_list = []
        cart = self.cart_lists[cart_id]
        with cart['lock']:
            for prod in cart['products']:
                
                final_list.append(prod[0])
                
                producer = self.producer_lists[prod[1]]
                producer['products'].remove({'product': prod[0], 'available': False})

            
            cart['products'].clear()

        for item in final_list:
            cons_number = cart_id + 1
            print(f'cons{cons_number} bought {item}')
        self.logger.info(f'Consumer checked out cart:{cart_id}\
            with the following products:{final_list}')

        return final_list

class TestMarketplaceMethods(unittest.TestCase):
    
    product_1 = {
        "product_type": "Tea",
        "name": "Linden",
        "type": "Herbal",
        "price": 9
    }

    product_2 = {
        "product_type": "Coffee",
        "name": "Indonezia",
        "acidity": 5.05,
        "roast_level": "MEDIUM",
        "price": 1
    }

    def setUp(self):
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)

    def test_publish(self):
        
        producer_id = self.marketplace.register_producer()

        self.assertTrue(self.marketplace.publish(producer_id, self.product_1))
        self.assertTrue(self.marketplace.publish(producer_id, self.product_1))
        self.assertFalse(self.marketplace.publish(producer_id, self.product_1))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)

    def test_add_to_cart(self):
        
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        
        self.marketplace.publish(producer_id, self.product_2)
        
        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.product_1))
        
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.product_2))

    def test_remove_from_cart(self):
        
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        
        self.marketplace.publish(producer_id, self.product_2)
        self.marketplace.add_to_cart(cart_id, self.product_2)

        
        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.product_2))

        
        self.marketplace.remove_from_cart(cart_id, self.product_2)

        
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.product_2))

    def test_place_order(self):
        
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        
        self.marketplace.publish(producer_id, self.product_2)
        self.marketplace.publish(producer_id, self.product_1)

        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.product_1))
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.product_2))

        expected_list = [self.product_1, self.product_2]
        
        self.assertEqual(self.marketplace.place_order(cart_id), expected_list)


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.retry = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    
                    sleep(product[2])
                    done = self.marketplace.publish(self.producer_id, product[0])
                    
                    while not done:
                        sleep(self.retry)
                        done = self.marketplace.publish(self.producer_id, product[0])
            sleep(self.retry)


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
