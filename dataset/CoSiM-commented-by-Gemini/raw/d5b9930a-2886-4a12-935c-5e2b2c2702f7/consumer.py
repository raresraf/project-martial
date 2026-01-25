


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__(name=kwargs["name"])
        self.carts: list = carts
        self.marketplace = marketplace
        self.retry_time = retry_wait_time
        self.output_str = "%s bought %s"

    def run(self):
        while len(self.carts) != 0:
            
            order = self.carts.pop(0)
            
            cart_id = self.marketplace.new_cart()

            while len(order) != 0:
                
                request = order.pop(0)

                
                if request["type"] == "add":
                    added_products = 0                           
                    while added_products < request["quantity"]:  
                        
                        if self.marketplace.add_to_cart(cart_id, request["product"]):
                            added_products += 1
                        else:
                            sleep(self.retry_time)               

                
                if request["type"] == "remove":
                    for _ in range(0, request["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, request["product"])

            
            cart_items = self.marketplace.place_order(cart_id)
            for product in cart_items:
                print(self.output_str % (self.name, product))    


from threading import Lock
from queue import Queue, Full, Empty
from typing import Dict


class Marketplace:
    

    def __init__(self, queue_size_per_producer: int):
        

        
        self.register_lock = Lock()                     
        self.producers_no = 0                           
        self.queue_size = queue_size_per_producer       
        self.producer_queues: Dict[int, Queue] = {}     

        
        self.cart_lock = Lock()                         
        self.consumers_no = 0                           
        self.consumer_carts: Dict[int, list] = {}       

        
        self.register_producer(ignore_limit=True)

    def register_producer(self, ignore_limit: bool = False) -> int:
        
        self.register_lock.acquire()                            
        producer_id = self.producers_no                         
        if ignore_limit:
            
            self.producer_queues[producer_id] = Queue()
        else:
            
            self.producer_queues[producer_id] = Queue(self.queue_size)
        self.producers_no += 1                                  
        self.register_lock.release()                            
        return producer_id



    def publish(self, producer_id: int, product) -> bool:
        
        try:
            self.producer_queues[producer_id].put_nowait(product)
        except Full:
            return False
        return True

    def new_cart(self) -> int:
        
        self.cart_lock.acquire()                    
        cart_id = self.consumers_no
        self.consumer_carts[cart_id] = []           
        self.consumers_no += 1                      
        self.cart_lock.release()                    
        return cart_id

    def add_to_cart(self, cart_id: int, product) -> bool:


        
        
        cart = self.consumer_carts[cart_id]
        for producer_id in range(0, self.producers_no):
            try:
                
                queue_head = self.producer_queues[producer_id].get_nowait()

                if queue_head == product:
                    
                    cart.append(queue_head)
                    return True

                
                while True:
                    
                    try:
                        self.producer_queues[producer_id].put_nowait(queue_head)
                        break
                    except Full:
                        
                        continue

            except Empty:
                
                continue

        return False

    def remove_from_cart(self, cart_id: int, product) -> None:
        
        try:
            
            self.consumer_carts[cart_id].remove(product)
            
            self.publish(0, product)
        except ValueError:
            
            pass

    def place_order(self, cart_id: int) -> list:
        
        return self.consumer_carts[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__(name=kwargs["name"], daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        while True:
            for product in self.products:                   
                produced = 0                                
                waited = False                              

                while produced < product[1]:                
                    if not waited:
                        sleep(product[2])                   

                    
                    if self.marketplace.publish(self.producer_id, product[0]):
                        produced += 1
                        waited = False
                    else:
                        sleep(self.republish_time)          
                        waited = True


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True, eq=True)
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
