


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def consumer_add_to_cart(self, quantity, cart_id, product_id):
        
        counter = 0
        while counter < quantity:
            if not self.marketplace.add_to_cart(cart_id, product_id):
                
                time.sleep(self.retry_wait_time)
            else:
                
                counter = counter + 1

    def run(self):
        
        cart_id = self.marketplace.new_cart()
        
        for cart in self.carts:
            for entry in cart:
                
                if entry.get("type") == "remove":
                    
                    for _ in range(entry.get("quantity")):
                        self.marketplace.remove_from_cart(cart_id, entry.get("product"))
                else:
                    
                    self.consumer_add_to_cart(entry.get("quantity"), cart_id, entry.get("product"))
        
        for product in self.marketplace.place_order(cart_id):
            print(self.name, "bought", product)


from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        self.producer_queue = {}
        self.producer_lock = Lock()
        self.consumer_id = -1
        self.consumer_queue = {}
        self.consumer_lock = Lock()

    def register_producer(self):
        
        self.producer_id = self.producer_id + 1
        
        self.producer_queue[self.producer_id] = []
        return self.producer_id

    def publish(self, producer_id, product):
        
        
        if len(self.producer_queue.get(producer_id)) < self.queue_size_per_producer:
            
            self.producer_lock.acquire()
            self.producer_queue.get(producer_id).append(product)
            self.producer_lock.release()
            return True
        return False

    def new_cart(self):
        
        self.consumer_id = self.consumer_id + 1
        
        self.consumer_queue[self.consumer_id] = []
        return self.consumer_id

    def add_to_cart(self, cart_id, product):
        
        
        for producer in self.producer_queue:
            for item in self.producer_queue.get(producer):
                if item == product:
                    
                    self.consumer_lock.acquire()
                    self.consumer_queue.get(cart_id).append([product, producer])
                    self.producer_queue.get(producer).remove(product)
                    self.consumer_lock.release()
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        
        for item, producer in self.consumer_queue.get(cart_id):
            if item == product:
                self.consumer_queue.get(cart_id).remove([product, producer])
                
                self.producer_lock.acquire()
                self.producer_queue.get(producer).append(product)
                self.producer_lock.release()
                break

    def place_order(self, cart_id):
        
        products = []
        
        for product, _ in self.consumer_queue.get(cart_id):
            products.append(product)
        return products


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def produce(self, product, quantity, produce_time, producer_id):
        
        counter = 0
        while counter < quantity:
            if not self.marketplace.publish(producer_id, product):
                
                time.sleep(self.republish_wait_time)
            else:
                
                time.sleep(produce_time)
                counter = counter + 1

    def run(self):
        while True:
            
            producer_id = self.marketplace.register_producer()
            for product in self.products:
                
                self.produce(product[0], product[1], product[2], producer_id)


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
