


from threading import Thread, current_thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:


            cart_id = self.marketplace.new_cart()
            for operation in cart:
                for _ in range(operation['quantity']):
                    if operation['type'] == 'add':
                        while not self.marketplace.add_to_cart(cart_id, operation['product']):
                            sleep(self.retry_wait_time)
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(cart_id, operation['product'])
            products = self.marketplace.place_order(cart_id)
            for product in products:
                print(f"{current_thread().name} bought {product}")

from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.current_producer_id = -1
        self.producer_queues = {}
        self.producers_lock = Lock()
        self.current_cart_id = -1
        self.carts = {}
        self.consumers_lock = Lock()

    def register_producer(self):
        
        with self.producers_lock:
            self.current_producer_id += 1
            self.producer_queues[self.current_producer_id] = []
            aux = self.current_producer_id
            return aux

    def publish(self, producer_id, product):
        
        if len(self.producer_queues[producer_id]) >= self.queue_size_per_producer:
            return False

        self.producer_queues[producer_id].append(product)
        return True

    def new_cart(self):
        
        with self.consumers_lock:
            self.current_cart_id += 1
            self.carts[self.current_cart_id] = []
            return self.current_cart_id

    def add_to_cart(self, cart_id, product):
        



        for producer_id, producer_queue in self.producer_queues.items():
            if product in producer_queue:
                producer_queue.remove(product)
                self.carts[cart_id].append((producer_id, product))
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        p_id = -1
        for producer_id, cart_product in self.carts[cart_id]:
            if cart_product == product:
                self.producer_queues[producer_id].append(product)
                p_id = producer_id
        self.carts[cart_id].remove((p_id, product))

    def place_order(self, cart_id):
        
        return [cart_product for producer_id, cart_product in self.carts[cart_id]]

from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        while True:
            for product, num, time in self.products:
                for _ in range(num):
                    while not self.marketplace.publish(self.producer_id, product):
                        sleep(self.republish_wait_time)
                sleep(time)


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
