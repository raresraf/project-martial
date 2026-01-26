


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):


        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

        

    def run(self):
        id_cart = self.marketplace.new_cart()

        for commands in self.carts:
            for command in commands:
                type_command = command["type"]
                product = command["product"]
                prod_quantity = command["quantity"]

                if type_command == "add":
                    for _ in range(prod_quantity):
                        while not self.marketplace.add_to_cart(id_cart, product):
                            time.sleep(self.retry_wait_time)

                if type_command == "remove":
                    for _ in range(prod_quantity):
                        self.marketplace.remove_from_cart(id_cart, product)

            for product in self.marketplace.place_order(id_cart):
                print(f'{self.name} bought {product}', flush=True)



from logging.handlers import RotatingFileHandler
from time import gmtime
import unittest
import logging
from .product import Product

logging.basicConfig(
    handlers =[RotatingFileHandler('marketplace.log', maxBytes = 500000, backupCount = 20)],
    format = "%(asctime)s %(levelname)s %(funcName)s %(message)s",
    level = logging.INFO)

logging.Formatter.converter = gmtime

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.nr_producers = 0
        self.nr_carts = 0
        self.producers_dict = {}
        self.consumers_dict = {}
        

    def register_producer(self):
        

        
        self.producers_dict[self.nr_producers] = []
        self.nr_producers += 1

        return self.nr_producers - 1

    def publish(self, producer_id, product):
        
        logging.info('%d %s', producer_id, product)

        products_list = self.producers_dict[producer_id]
        if len(products_list) < self.queue_size_per_producer:
            products_list.append({product:'a'})
            logging.info('True')
            return True
        logging.info('False')
        return False

    def new_cart(self):
        

        
        
        self.consumers_dict[self.nr_carts] = []
        self.nr_carts += 1

        return self.nr_carts - 1

    def add_to_cart(self, cart_id, product):
        

        logging.info('%d %s', cart_id, product)

        
        

        cart_list = self.consumers_dict[cart_id]

        for key in self.producers_dict:
            products_map = self.producers_dict[key]
            for dict_item in products_map:
                if product in dict_item:
                    if dict_item[product] == 'a':
                        cart_list.append({product:key})
                        dict_item[product] = 'u'
                        logging.info('True')
                        return True
        logging.info('False')
        return False

    def remove_from_cart(self, cart_id, product):
        
        logging.info('%d %s', cart_id, product)

        
        
        cart_list = self.consumers_dict[cart_id]
        for prod_dict in cart_list:
            if product in prod_dict:
                product_list = self.producers_dict[prod_dict[product]]
                for prodd in product_list:
                    if product in prodd:
                        prodd[product] = 'a'
                cart_list.remove(prod_dict)
                return

    def place_order(self, cart_id):
        
        logging.info('%d', cart_id)

        
        
        products_ordered_list = []
        for item in self.consumers_dict[cart_id]:
            for key in item:
                products_ordered_list.append(key)
                for prod in self.producers_dict[item[key]]:
                    if key in prod:
                        self.producers_dict[item[key]].remove(prod)
                        break

        self.consumers_dict[cart_id].clear()
        logging.info(products_ordered_list)
        return products_ordered_list

class TestMarketPlace(unittest.TestCase):
    
    
    def setUp(self) -> None:
        self.market_place_object = Marketplace(10)

        self.products_list = []
        for i in range(10):
            new_product = Product('product' + str(i), i * 10)
            self.products_list.append(new_product)

        self.cart = {}
        self.producer = None

    def test_register_producer(self):
        
        for i in range(10):
            self.assertEqual(self.market_place_object.register_producer(), i)

    def test_publish(self):
        
        self.producer = self.market_place_object.register_producer()
        for product in self.products_list:
            self.assertTrue(self.market_place_object.publish(self.producer, product))

        
        new_product = Product('product10', 0)
        self.assertFalse(self.market_place_object.publish(self.producer, new_product))

    def test_new_cart(self):
        
        for i in range(10):
            self.assertEqual(self.market_place_object.new_cart(), i)

    def test_add_to_cart(self):
        
        self.producer = self.market_place_object.register_producer()
        for product in self.products_list:
            self.market_place_object.publish(self.producer, product)
        self.cart = self.market_place_object.new_cart()
        for product in self.products_list:
            self.assertTrue(self.market_place_object.add_to_cart(self.cart, product))
    def test_remove_from_cart(self):
        
        product_to_be_removed = self.products_list[0]

        self.producer = self.market_place_object.register_producer()
        self.market_place_object.publish(self.producer, product_to_be_removed)

        self.cart = self.market_place_object.new_cart()
        self.market_place_object.add_to_cart(self.cart, product_to_be_removed)

        producer_list = self.market_place_object.producers_dict[self.producer]

        self.assertTrue({product_to_be_removed:'u'} in producer_list)
        self.market_place_object.remove_from_cart(self.cart, product_to_be_removed)

        self.assertTrue({product_to_be_removed:'a'} in producer_list)

    def test_place_order(self):
        
        self.producer = self.market_place_object.register_producer()

        for product in self.products_list:
            self.market_place_object.publish(self.producer, product)

        self.cart = self.market_place_object.new_cart()

        for product in self.products_list:
            self.market_place_object.add_to_cart(self.cart, product)

        products_ordered = self.market_place_object.place_order(self.cart)

        self.assertEqual(self.products_list, products_ordered)


from threading import Thread
import time


class Producer(Thread):
    



    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, daemon=kwargs['daemon'])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs['name']

        
        

    def run(self):
        id_prod = self.marketplace.register_producer()
        while True:
            for (prod, prod_quantity, waiting_time) in self.products:
                time.sleep(waiting_time)
                for _ in range(prod_quantity):
                    while not self.marketplace.publish(id_prod, prod):
                        time.sleep(self.republish_wait_time)


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
