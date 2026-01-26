


from __future__ import print_function
from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        if 'name' in kwargs:
            self.name = kwargs['name']
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_to_cart(self, product, cart_id):
        
        if not self.marketplace.add_to_cart(cart_id, product):
            while True:
                sleep(self.retry_wait_time)
                if self.marketplace.add_to_cart(cart_id, product):
                    break

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for action in cart:
                if action['type'] == 'add':
                    for _ in range(action['quantity']):
                        self.add_to_cart(action['product'], cart_id)
                else:
                    for _ in range(action['quantity']):
                        self.marketplace.remove_from_cart(cart_id, action['product'])

            for item in self.marketplace.place_order(cart_id):
                print("{} bought {}".format(self.name, item))


from threading import Lock


class Cart:
    
    def __init__(self):
        
        self.products = []
        self.producer_ids = []

    def add_product(self, product, producer_id):
        
        self.products.append(product)
        self.producer_ids.append(producer_id)

    def remove_product(self, product):
        
        idx = self.products.index(product)
        producer_id = self.producer_ids[idx]
        del self.products[idx]
        del self.producer_ids[idx]
        return producer_id


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.carts = []
        self.carts_lock = Lock()

        self.product_queues = []
        self.products_locks = []
        self.queues_lock = Lock()

    def register_producer(self):
        
        with self.queues_lock:
            producer_id = len(self.product_queues)
            self.product_queues.append([])
            self.products_locks.append(Lock())
        return producer_id

    def publish(self, producer_id, product):
        
        producer_id = int(producer_id)
        with self.products_locks[producer_id]:
            if len(self.product_queues[producer_id]) < self.queue_size_per_producer:
                self.product_queues[producer_id].append(product)
                return True
        return False

    def new_cart(self):
        
        with self.carts_lock:
            cart_id = len(self.carts)
            self.carts.append(Cart())
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        added = False
        with self.queues_lock:
            q_num = len(self.product_queues)

        for i in range(q_num):
            with self.products_locks[i]:
                if product not in self.product_queues[i]:
                    continue
                added = True
                self.carts[cart_id].add_product(product, i)
                self.product_queues[i].remove(product)
                break
        return added

    def remove_from_cart(self, cart_id, product):
        
        producer_id = self.carts[cart_id].remove_product(product)
        with self.products_locks[producer_id]:
            self.product_queues[producer_id].append(product)

    def place_order(self, cart_id):
        
        return self.carts[cart_id].products


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self)
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'daemon' in kwargs:
            self.daemon = kwargs['daemon']

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for prod in self.products:
                for _ in range(prod[1]):  
                    if not self.marketplace.publish(self.producer_id, prod[0]):
                        while True:
                            sleep(self.republish_wait_time)
                            if self.marketplace.publish(self.producer_id, prod):
                                break
                    sleep(prod[2])  
