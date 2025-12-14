

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        for i in range(len(self.carts)):
            cart_id = self.marketplace.new_cart()


            for cart in self.carts[i]:
                if cart['type'] == 'add':
                    for _ in range(cart['quantity']):
                        while not self.marketplace.add_to_cart(cart_id, cart['product']):


                            time.sleep(self.retry_wait_time)
                elif cart['type'] == 'remove':
                    for _ in range(cart['quantity']):
                        self.marketplace.remove_from_cart(cart_id, cart['product'])
            products_to_buy = self.marketplace.place_order(cart_id)
            for item in products_to_buy:
                print(str(self.kwargs['name']) + " bought " + str(item))
            self.marketplace.clean_cart(cart_id)

from threading import RLock


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.lock = RLock()
        self.producers = []
        self.carts = []
        self.counter = 0
        self.consumer_id = 0

    def register_producer(self):
        
        self.lock.acquire()
        producer_dict = dict()
        queue = list()
        self.counter += 1
        producer_dict['id'] = self.counter
        producer_dict['published_products'] = queue
        self.producers.append(producer_dict)
        self.lock.release()
        return producer_dict['id']

    def publish(self, producer_id, product):
        
        for producer in self.producers:
            if producer['id'] == producer_id:
                if len(producer['published_products']) < self.queue_size_per_producer:
                    producer['published_products'].append([product, True])
                    return True
                return False
        return False

    def new_cart(self):
        
        self.lock.acquire()
        cart_dict = dict()
        products_in_cart = list()
        self.consumer_id += 1
        cart_dict['id'] = self.consumer_id
        cart_dict['products_in_cart'] = products_in_cart
        self.carts.append(cart_dict)
        self.lock.release()
        return cart_dict['id']

    def add_to_cart(self, cart_id, product):
        
        for cart in self.carts:
            if cart['id'] == cart_id:
                self.lock.acquire()
                for producer in self.producers:
                    for published_product in producer['published_products']:
                        if published_product[0][0] == product and published_product[1]:
                            cart['products_in_cart'].append(product)
                            published_product[1] = False
                            self.lock.release()
                            return True
                self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        for cart in self.carts:
            if cart['id'] == cart_id:
                self.lock.acquire()
                for product_in_cart in cart['products_in_cart']:
                    if product_in_cart == product:
                        cart['products_in_cart'].remove(product_in_cart)
                        for producer in self.producers:
                            for published_product in producer['published_products']:
                                if published_product[0][0] == product and not published_product[1]:
                                    published_product[1] = True
                                    self.lock.release()
                                    return None
                self.lock.release()
        return None

    def place_order(self, cart_id):
        
        for cart in self.carts:
            if cart['id'] == cart_id:
                self.lock.acquire()
                index = 0
                while index < len(cart['products_in_cart']):
                    for producer in self.producers:
                        for product in producer['published_products']:
                            if product[0][0] == cart['products_in_cart'][index] and not product[1]:
                                producer['published_products'].remove(product)
                                index += 1
                                break
                        if index == len(cart['products_in_cart']):
                            self.lock.release()
                            return cart['products_in_cart']
                self.lock.release()
                return cart['products_in_cart']
        return []

    def clean_cart(self, cart_id):
        
        for cart in self.carts:
            if cart['id'] == cart_id:
                cart['products_in_cart'].clear()

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.daemon = True
        self.is_published = False
        self.kwargs = kwargs

    def run(self):
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    self.is_published = self.marketplace.publish(producer_id, product)
                    while not self.is_published:
                        self.is_published = self.marketplace.publish(producer_id, product)
                        time.sleep(self.republish_wait_time)
                    time.sleep(product[2])


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
