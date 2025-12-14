


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.market = marketplace
        self.wait_time = retry_wait_time
        self.carts = carts
        

    def run(self):
        cart = self.market.new_cart()
        
        for cart_list in self.carts:
            for act in cart_list:
                
                
                if act["type"] == "add":
                    cantitate = act["quantity"]
                    while cantitate > 0 :
                        ok = self.market.add_to_cart(cart, act["product"])
                        if ok:
                            cantitate -= 1
                        else :
                            
                            time.sleep(self.wait_time)
                else:
                    
                    
                    cantitate = act["quantity"]
                    for i in range(cantitate):
                        self.market.remove_from_cart(cart, act["product"])
        comanda = self.market.place_order(cart)
        
        for product in comanda :
            print(self.name + " bought " + str(product))



from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.consumer_lock = Lock()
        self.buffer = []
        self.carts = []
        self.producer_lock = Lock()
        self.producer_id = -1


        self.cart_id = -1
        self.queue_size = queue_size_per_producer


    def register_producer(self):
        
        self.producer_lock.acquire()
        self.producer_id += 1
        self.buffer.append([])
        
        self.producer_lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        
        self.producer_lock.acquire()
        
        if len(self.buffer[producer_id]) >= self.queue_size :
            self.producer_lock.release()
            return False
        to_add = {
            'product' : product,
            'producer_id' : producer_id
        }
        self.buffer[producer_id].append(to_add)
        self.producer_lock.release()
        return True

    def new_cart(self):
        
        self.consumer_lock.acquire()
        self.cart_id += 1
        self.carts.append([])
        self.consumer_lock.release()
        return self.cart_id


    def add_to_cart(self, cart_id, product):
        
        self.consumer_lock.acquire()
        
        for i in range(len(self.buffer)):
            for product_aux in self.buffer[i]:
                if product == product_aux['product']:
                    
                    self.buffer[i].remove(product_aux)
                    self.carts[cart_id].append(product_aux)
                    self.consumer_lock.release()
                    return True
        self.consumer_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        
        self.consumer_lock.acquire()
        for product_aux in self.carts[cart_id]:
            if product_aux['product'] == product: 
                
                self.buffer[product_aux['producer_id']].append(product_aux)
                self.carts[cart_id].remove(product_aux)
                break
        self.consumer_lock.release()
       



    def place_order(self, cart_id):
        
        order = []
        for prod in self.carts[cart_id] :
            order.append(prod['product'])
        return order
        


from threading import Thread
import time

from httplib2 import ProxiesUnavailableError


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.market = marketplace
        self.wait_time = republish_wait_time
        self.products = products
        
        self.id = self.market.register_producer()

    def run(self):
        contor = 0
        
        time.sleep(self.products[contor][2])
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    while not self.market.publish(self.id, product[0]):
                        time.sleep(self.wait_time)
                    
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
