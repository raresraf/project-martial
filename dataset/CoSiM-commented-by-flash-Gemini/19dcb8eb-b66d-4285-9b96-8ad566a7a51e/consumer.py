

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.activities = {}

    def extract_activities(self, activity, consumer_id):
        
        self.activities[consumer_id] = []

        activity_type = activity.get("type")
        activity_product = activity.get("product")
        activity_quantity = activity.get("quantity")

        self.activities[consumer_id] = (activity_type, activity_product, activity_quantity)

        return self.activities[consumer_id]

    def get_info(self, consumer_id, cart):
        
        for activity in cart:

            self.activities[consumer_id] = self.extract_activities(activity, consumer_id)

            if self.activities[consumer_id][0] == "add":

                product_counter = 0

                while product_counter < self.activities[consumer_id][2]:

                    add_ok = self.marketplace.\
                        add_to_cart(consumer_id, self.activities[consumer_id][1])

                    if add_ok:
                        product_counter = product_counter + 1
                    elif not add_ok:
                        time.sleep(self.retry_wait_time)

            elif self.activities[consumer_id][0] == "remove":

                product_counter = 0

                while product_counter < self.activities[consumer_id][2]:

                    self.marketplace.remove_from_cart(consumer_id, self.activities[consumer_id][1])
                    product_counter = product_counter + 1

    def run(self):

        for cart in self.carts:

            consumer_id = self.marketplace.new_cart()
            self.get_info(consumer_id, cart)

            for product in self.marketplace.place_order(consumer_id):
                print(self.name, "bought", product)

from threading import Lock
import unittest



from tema.consumer import Consumer
from tema.producer import Producer
from tema.product import Coffee, Tea


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer



        self.carts_counter = -1
        self.producers_counter = -1

        self.carts = []
        self.products = []

        self.in_stock_products = []
        self.in_stock_products_producers = []

        self.carts_lock = Lock()
        self.lock_publish = Lock()
        self.producers_lock = Lock()
        self.in_stock_lock = Lock()

    def register_producer(self):
        
        self.producers_lock.acquire()

        self.producers_counter = self.producers_counter + 1

        self.products.append(0)

        self.producers_lock.release()

        return self.producers_counter

    def publish(self, producer_id, product):
        

        number_of_products = self.products[producer_id]

        if number_of_products < self.queue_size_per_producer:

            self.in_stock_products.append(product)
            self.products[producer_id] = self.products[producer_id] + 1
            self.in_stock_products_producers.append((producer_id, product))

            return True

        elif number_of_products >= self.queue_size_per_producer:
            return False

    def new_cart(self):
        
        self.carts_lock.acquire()

        self.carts_counter = self.carts_counter + 1
        self.carts.append([])

        self.carts_lock.release()

        return self.carts_counter

    def add_to_cart(self, cart_id, product):
        
        self.in_stock_lock.acquire()
        if product in self.in_stock_products:

            self.carts[cart_id].append(product)
            self.in_stock_products.remove(product)

            self.in_stock_lock.release()

            return True

        else:
            self.in_stock_lock.release()
            return False

    def remove_from_cart(self, cart_id, product):
        

        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)

            self.in_stock_products.append(product)

    def place_order(self, cart_id):
        

        for prod in self.carts[cart_id]:

            if prod in self.in_stock_products_producers:
                self.producers_lock.acquire()

                element = self.in_stock_products_producers[prod]
                self.products[element] = self.products[element] - 1

                self.producers_lock.release()

        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(24)
        self.tea1 = Tea("Mint", 15, "Green")
        self.tea2 = Tea("Earl grey", 30, "Black")
        self.coffee = Coffee("Lavazza", 14, "2.23", "MEDIUM")
        self.producer = Producer([[self.tea1, 8, 0.11],


                                  [self.tea2, 5, 0.7],
                                  [self.coffee, 1, 0.13]],
                                 self.marketplace,
                                 0.35)
        self.consumer = Consumer([[{"type": "add", "product": self.coffee, "quantity": 1},
                                   {"type": "add", "product": self.tea1, "quantity": 4},
                                   {"type": "add", "product": self.tea2, "quantity": 2},
                                   {"type": "remove", "product": self.tea2, "quantity": 1}
                                   ]],
                                 self.marketplace,
                                 0.31)
        self.cart_id = self.marketplace.new_cart()

    def test_register_function(self):
        
        self.assertEqual(str(self.marketplace.register_producer()), "0")



import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)



        self.products = products

        self.marketplace = marketplace

        self.republish_wait_time = republish_wait_time

        self.kwargs = kwargs

    def run(self):

        while True:

            for prod in self.products:

                product_counter = 0

                product = prod[0]
                quantity = prod[1]
                wait_time = prod[2]

                while product_counter < quantity:

                    if self.marketplace.publish(self.marketplace.register_producer(), product):
                        product_counter = product_counter + 1
                        time.sleep(wait_time)

                    elif not self.marketplace.\
                            publish(self.marketplace.register_producer(), product):
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
