


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get('name')

    def run(self):
        if self.marketplace.nr_of_consumers == -1:
            self.marketplace.nr_of_consumers = 1
        else:
            self.marketplace.nr_of_consumers += 1

        for cart in self.carts:
            new_cart_id = self.marketplace.new_cart()
            for task in cart:
                i=0
                while i < task.get('quantity'):
                    if task.get('type') == "add":   
                        check = self.marketplace.add_to_cart(new_cart_id, task.get('product'))
                    elif task.get('type') == "remove":
                        check = self.marketplace.remove_from_cart(new_cart_id, task.get('product'))
                    if check == False:
                        sleep(self.retry_wait_time)
                    else:
                        i += 1

            for prod in self.marketplace.place_order(new_cart_id):
                print("%s bought %s" % (self.name, prod))
        
        self.marketplace.nr_of_consumers -= 1
        

from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.last_producer_id = -1
        self.last_cart_id = -1
        self.prod_queue = [] 
        self.all_carts = []
        self.producerAndProduct = []
        self.addToCart_lock = Lock()
        self.removeFromCart_lock = Lock()
        self.lastProdId_lock = Lock()
        self.publish_lock = Lock()
        self.new_cart_lock = Lock()


        self.nr_of_consumers = -1
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        
        self.lastProdId_lock.acquire()
        self.last_producer_id+=1
        self.prod_queue.append([])
        self.lastProdId_lock.release()
        return self.last_producer_id


    def publish(self, producer_id, product):
        
        self.publish_lock.acquire()
        if(len(self.prod_queue[producer_id]) < self.queue_size_per_producer):
            self.prod_queue[producer_id].append(product)
            self.publish_lock.release()
            return True
        else:
            self.publish_lock.release()
            return False

    def new_cart(self):
        
        self.new_cart_lock.acquire()
        self.last_cart_id+=1
        
        self.all_carts.append([])
        self.new_cart_lock.release()
        return self.last_cart_id


    def add_to_cart(self, cart_id, product):
        
        self.addToCart_lock.acquire()
        for i in range(len(self.prod_queue)):
            for j in range(len(self.prod_queue[i])):
                
                if self.prod_queue[i][j] == product:
                    self.all_carts[cart_id].append(product)
                    
                    self.producerAndProduct.append((i, product))
                    
                    self.prod_queue[i].remove(product)
                    self.addToCart_lock.release()
                    return True
        self.addToCart_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.removeFromCart_lock.acquire()
        for i in range(len(self.all_carts[cart_id])):
            if self.all_carts[cart_id][i] == product:
                self.all_carts[cart_id].remove(product)
                
                for j in range(len(self.producerAndProduct)):
                    (index, searchProduct) = self.producerAndProduct[j]
                    if(searchProduct == product): 
                        self.prod_queue[index].append(product)
                        
                        self.producerAndProduct.pop(j)
                        break
                break 
        self.removeFromCart_lock.release()
                    

    def place_order(self, cart_id):
        
        return self.all_carts[cart_id]



from threading import Thread
import threading
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs.get('name')

    def run(self):
        id = self.marketplace.register_producer()

        while 1:
            for prod in self.products:
                nr_of_prod = 0
                while nr_of_prod < prod[1]:
                    check = self.marketplace.publish(id, prod[0])
                    if check:
                        
                        sleep(prod[2])
                        nr_of_prod += 1
                    else:
                        
                        sleep(self.republish_wait_time)
            
            if self.marketplace.nr_of_consumers == 0:
                break
                        >>>> file: product.py


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
