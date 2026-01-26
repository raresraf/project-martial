


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                if operation.get("type") == "add":
                    for quantity in range(operation.get("quantity")):
                        status = self.marketplace.add_to_cart(cart_id, operation.get("product"))
                        while not status:
                            sleep(self.retry_wait_time)
                            status = self.marketplace.add_to_cart(cart_id, operation.get("product"))
                elif operation.get("type") == "remove":
                    for quantity in range(operation.get("quantity")):
                        self.marketplace.remove_from_cart(cart_id, operation.get("product"))
            for product in self.marketplace.place_order(cart_id):
                print(self.kwargs.get("name") + " bought " + product.__str__())>>>> file: marketplace.py

from threading import Lock, Thread

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_index = -1
        self.consumer_cart_index = -1
        self.producer_queue_size_list = []
        self.products_list = []
        self.producers_to_products_map = {}
        self.carts = []
        self.lock_register = Lock()
        self.lock_publish = Lock()
        self.lock_new_cart = Lock()
        self.lock_add_cart = Lock()
        self.lock_remove_cart = Lock()

    def register_producer(self):
        
        self.lock_register.acquire()
        self.producer_index += 1
        self.lock_register.release()
        self.producer_queue_size_list.append(0)
        return self.producer_index


    def publish(self, producer_id, product):
        
        if self.producer_queue_size_list[producer_id] <= self.queue_size_per_producer:
            self.producer_queue_size_list[producer_id] += 1
            self.products_list.append(product)
            self.lock_publish.acquire()
            self.producers_to_products_map[product] = producer_id
            self.lock_publish.release()
            return True
        else:
            return False

    def new_cart(self):
        
        self.lock_new_cart.acquire()
        self.consumer_cart_index += 1
        self.lock_new_cart.release()
        self.carts.append([])
        return self.consumer_cart_index

    def add_to_cart(self, cart_id, product):
        
        if product in self.products_list:
            self.products_list.remove(product)


            self.lock_add_cart.acquire()
            self.producer_queue_size_list[self.producers_to_products_map[product]] -= 1
            self.lock_add_cart.release()
            self.carts[cart_id].append(product)
            return True
        else:
            return False

    def remove_from_cart(self, cart_id, product):
        
        self.products_list.append(product)


        self.carts[cart_id].remove(product)
        self.lock_remove_cart.acquire()
        self.producer_queue_size_list[self.producers_to_products_map[product]] += 1
        self.lock_remove_cart.release()

    def place_order(self, cart_id):
        
        return self.carts[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        producer_id = self.marketplace.register_producer()
        while True:
            for (product, quantity, wait_time) in self.products:
                while quantity > 0:
                    status = self.marketplace.publish(producer_id, product)
                    while not status:
                        sleep(self.republish_wait_time)
                        status = self.marketplace.publish(producer_id, product)
                    if status:
                        quantity -= 1
                        sleep(wait_time)



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
