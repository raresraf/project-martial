


from threading import Thread
import time

class Consumer(Thread):
    


    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs


    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for product in cart:
                for _ in range(product["quantity"]):
                    if product["type"] == "add":
                        
                        while not self.marketplace.add_to_cart(cart_id, product["product"]):
                            time.sleep(self.retry_wait_time)

                    elif product["type"] == "remove":
                        
                        
                        self.marketplace.remove_from_cart(cart_id, product["product"])

            
            bought = self.marketplace.place_order(cart_id)
            for item in bought:
                print(self.kwargs['name'], "bought", item)

from threading import Lock
import collections

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.cart_ids = 0
        self.producers_buffers = collections.defaultdict(list)
        self.carts = collections.defaultdict(list)
        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()
        self.add_to_cart_lock = Lock()
        self.remove_from_cart_lock = Lock()


    def register_producer(self):
        
        
        
        with self.register_producer_lock:
            producer_id = self.id_producer
            self.id_producer += 1

        
        return str(producer_id)


    def publish(self, producer_id, product):
        
        
        if producer_id in self.producers_buffers:
            
            
            if len(self.producers_buffers[producer_id]) >= self.queue_size_per_producer:
                return False

        
        self.producers_buffers[producer_id].append(product)
        return True


    def new_cart(self):
        
        
        
        with self.new_cart_lock:
            cart_id = self.cart_ids
            self.cart_ids += 1

        
        return cart_id


    def add_to_cart(self, cart_id, product):
        
        for producer_id, products in self.producers_buffers.items():
            for prod in products:
                if product == prod:
                    
                    
                    
                    self.carts[cart_id].append((product, producer_id))
                    products.remove(product)
                    return True

        
        
        return False


    def remove_from_cart(self, cart_id, product):
        
        for item in self.carts[cart_id]:
            
            
            if product == item[0]:
                self.carts[cart_id].remove(item)
                self.producers_buffers[item[1]].append(product)
                return


    def place_order(self, cart_id):
        
        
        
        return [i[0] for i in self.carts[cart_id]]


from threading import Thread
import time

class Producer(Thread):
    


    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.id_producer = self.marketplace.register_producer()


    def run(self):
        
        
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    
                    while not self.marketplace.publish(self.id_producer, product[0]):
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
