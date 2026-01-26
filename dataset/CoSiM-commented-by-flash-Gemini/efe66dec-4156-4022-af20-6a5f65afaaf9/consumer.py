


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs["name"]

    def run(self):
        
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()


            for event in cart:
                count = 0
                while count < event["quantity"]:
                    if event["type"] == "add":
                        if self.marketplace.add_to_cart(cart_id, event["product"]):
                            count += 1
                        else:
                            time.sleep(self.retry_wait_time)


                    if event["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, event["product"])
                        count += 1
            products_list = self.marketplace.place_order(cart_id)

            
            
            for product in products_list:
                print(self.consumer_name + " bought " + str(product))

from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        
        


        self.register_lock = Lock()
        self.add_remove_lock = Lock()
        self.cart_lock = Lock()

        
        
        self.queue_capacity = queue_size_per_producer

        
        
        self.nr_producers = 0

        
        self.nr_carts = 0

        
        
        
        self.producer_queues = {}

        
        
        
        self.carts = []

    def register_producer(self):
        

        
        
        
        
        
        with self.register_lock:
            self.nr_producers += 1
            producer_id = "prod" + str(self.nr_producers)
            
            
            self.producer_queues[producer_id] = []

        return producer_id

    def publish(self, producer_id, product):
        

        
        
        
        if len(self.producer_queues[producer_id]) < self.queue_capacity:
            self.producer_queues[producer_id].append(product)
            return True
        return False

    def new_cart(self):
        

        
        
        
        
        with self.cart_lock:
            cart_id = self.nr_carts
            self.carts.append([])
            self.nr_carts += 1
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        for (producer_id, products_queue) in self.producer_queues.items():
            if product in products_queue:
                products_queue.remove(product)
                self.carts[cart_id].append((product, producer_id))
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        

        
        
        
        
        
        
        with self.add_remove_lock:
            index = 0
            for (cart_product, producer_id) in self.carts[cart_id]:
                if cart_product == product:
                    self.producer_queues[producer_id].append(product)
                    break
                index += 1
        self.carts[cart_id].pop(index)

    def place_order(self, cart_id):
        

        
        
        
        
        return [elem[0] for elem in self.carts[cart_id]]


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        
        while True:
            for product in self.products:
                count = 0
                
                
                
                


                while count < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        time.sleep(product[2])
                        count += 1
                    else:
                        
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
