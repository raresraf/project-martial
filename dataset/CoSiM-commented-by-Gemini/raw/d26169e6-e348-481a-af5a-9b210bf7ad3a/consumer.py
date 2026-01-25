


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        
        for cart in self.carts:
            id_cart = self.marketplace.new_cart()
            for purchase in cart:
                if purchase["type"] == 'add':
                    for _ in range(purchase["quantity"]):
                        cart_new_product = self.marketplace.add_to_cart(id_cart,
                                                                        purchase["product"])


                        while not cart_new_product:
                            sleep(self.retry_wait_time)
                            cart_new_product = self.marketplace.add_to_cart(id_cart,
                                                                            purchase["product"])
                else:
                    for _ in range(purchase["quantity"]):
                        self.marketplace.remove_from_cart(id_cart, purchase["product"])
            order = self.marketplace.place_order(id_cart)
            for buy in order:
                print(self.name + ' bought ' + str(buy))


import threading


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.contor_producer = -1
        self.contor_consumer = -1
        self.product_queue = [[]]
        self.cart_queue = [[]]
        self.producer_cart = [[]]
        self.lock = threading.Lock()
        self.producer_locks = []

    def register_producer(self):
        
        with self.lock:
            self.contor_producer += 1
            tmp = self.contor_producer
            self.product_queue.append([])
            self.producer_cart.append([])
            self.producer_locks.append(threading.Lock())
        return tmp

    def publish(self, producer_id, product):
        
        self.producer_locks[producer_id].acquire()
        if self.queue_size_per_producer > len(self.product_queue[producer_id]):
            self.product_queue[producer_id].append(product)
            self.producer_locks[producer_id].release()
            return True
        self.producer_locks[producer_id].release()
        return False

    def new_cart(self):
        
        self.lock.acquire()
        self.contor_consumer += 1
        self.cart_queue.append([])
        tmp = self.contor_consumer
        self.lock.release()
        return tmp

    def add_to_cart(self, cart_id, product):
        
        if any(product in list_products for list_products in self.product_queue):
            for products in self.product_queue:
                for prod in products:
                    if prod == product:
                        self.lock.acquire()
                        tmp = self.product_queue.index(products)
                        self.producer_cart[tmp].append((product, cart_id))
                        self.cart_queue[cart_id].append(product)
                        self.product_queue[tmp].remove(product)
                        self.lock.release()
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.cart_queue[cart_id].remove(product)
        for producer in self.producer_cart:
            if (cart_id, product) in producer:
                tmp = self.producer_cart.index(producer)
                self.producer_cart.remove((cart_id, product))
                self.producer_locks[tmp].acquire()
                self.product_queue[tmp].append(product)
                self.producer_locks[tmp].release()

    def place_order(self, cart_id):
        
        return self.cart_queue[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        
        id_producer = self.marketplace.register_producer()
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    sleep(product[2])
                    market_confirm = self.marketplace.publish(id_producer, product[0])
                    while not market_confirm:


                        sleep(self.republish_wait_time)
                        market_confirm = self.marketplace.publish(id_producer, product[0])


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
