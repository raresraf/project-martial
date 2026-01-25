


from threading import Thread
import time



class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):


        Thread.__init__(self, name=kwargs['name'])
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        

    def run(self):
        for cart in self.carts:
            my_cart_id = self.marketplace.new_cart()
            i = 0
            while i < len(cart):
                prod = cart[i]['product']
                quantity = cart[i]['quantity']
                if cart[i]['type'] == 'add':
                    while quantity != 0:
                        while not self.marketplace.add_to_cart(my_cart_id, prod):
                            time.sleep(self.retry_wait_time)
                        quantity = quantity - 1
                else:
                    while quantity != 0:
                        self.marketplace.remove_from_cart(my_cart_id, prod)
                        quantity = quantity - 1
                i = i + 1
            self.marketplace.place_order(my_cart_id)


from threading import RLock
import threading

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.queues = []
        self.nr_producers = 0
        self.nr_produse_producator = {}
        self.lock_per_queue = []
        self.carts = {}
        self.nr_carts = 0

    def register_producer(self):
        
        self.lock_per_queue.append(RLock())
        self.queues.append([])
        self.nr_produse_producator[self.nr_producers] = 0
        self.nr_producers = self.nr_producers + 1
        return self.nr_producers - 1

    def publish(self, producer_id, product):
        
        if self.nr_produse_producator[producer_id] > self.queue_size_per_producer:
            return False


        self.lock_per_queue[producer_id].acquire()
        self.queues[producer_id].append(product)
        self.nr_produse_producator[producer_id] = self.nr_produse_producator[producer_id] + 1
        self.lock_per_queue[producer_id].release()
        return True

    def new_cart(self):
        
        self.carts[self.nr_carts] = []
        self.nr_carts = self.nr_carts + 1
        return self.nr_carts - 1

    def add_to_cart(self, cart_id, product):
        
        found = False
        for i in range(0, self.nr_producers):
            self.lock_per_queue[i].acquire()
            for j in range(0, len(self.queues[i])):
                if self.queues[i][j] == product:
                    self.queues[i] = self.queues[:i] + self.queues[i+1:]
                    self.nr_produse_producator[i] = self.nr_produse_producator[i] - 1
                    self.carts[cart_id].append((i, product))
                    found = True
                    break
            self.lock_per_queue[i].release()
            if found:
                break
        return found

    def remove_from_cart(self, cart_id, product):
        
        for i in range(0, len(self.carts[cart_id])):
            (nr_prod, prod) = self.carts[cart_id][i]
            if prod == product:
                self.lock_per_queue[nr_prod].acquire()
                self.queues[nr_prod].append(product)
                self.nr_produse_producator[nr_prod] = self.nr_produse_producator[nr_prod] + 1
                self.lock_per_queue[nr_prod].release()
                self.carts[cart_id] = self.carts[cart_id][:i] + self.carts[cart_id][i+1:]
                break

    def place_order(self, cart_id):
        
        for item in self.carts[cart_id]:
            (nr_prod, prod) = item
            self.nr_produse_producator[nr_prod] = self.nr_produse_producator[nr_prod] - 1
            print(threading.currentThread().getName(), "bought", prod)


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, name=kwargs['name'], daemon=kwargs['daemon'])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        

    def run(self):
        my_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                (prod, quantity, duration) = product
                while quantity != 0:
                    time.sleep(duration)
                    while not self.marketplace.publish(my_id, prod):
                        time.sleep(self.republish_wait_time)
                    quantity = quantity - 1


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
