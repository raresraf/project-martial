


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        for cart in self.carts:
            new_cart = self.marketplace.new_cart()
            for instruction in cart:
                i = 0
                product = instruction['product']
                action = instruction['type']
                while i < instruction['quantity']:
                    if action == "add":
                        if self.marketplace.add_to_cart(new_cart, product):
                            i += 1
                        else:
                            sleep(self.retry_wait_time)
                    elif action == "remove":
                        self.marketplace.remove_from_cart(new_cart, product)
                        i += 1


            new_list = self.marketplace.place_order(new_cart)
            for _, instruction in new_list:
                print(self.name, "bought", instruction)

from threading import Lock


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.producer_queue_size = queue_size_per_producer
        self.producers = {}
        self.carts = {}
        self.current_producer = 1
        self.current_cart = 1
        self.register_lock = Lock()
        self.cart_lock = Lock()

    def register_producer(self):
        
        self.register_lock.acquire()
        products = []
        producer_id = self.current_producer
        self.producers[producer_id] = products
        self.current_producer += 1
        self.register_lock.release()
        return producer_id

    def publish(self, producer_id, product):
        



        if len(self.producers[producer_id]) >= self.producer_queue_size:
            return False

        self.producers[producer_id].append(product)
        return True

    def new_cart(self):
        
        self.cart_lock.acquire()
        cart = []
        new_cart = self.current_cart
        self.carts[new_cart] = cart
        self.current_cart += 1
        self.cart_lock.release()
        return new_cart

    def add_to_cart(self, cart_id, product):
        

        if cart_id not in self.carts.keys():
            return False

        for producer in self.producers:
            if product in self.producers[producer]:


                self.carts[cart_id].append([producer, product])
                self.producers[producer].remove(product)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        if cart_id not in self.carts.keys():
            return False



        for producer_id, prod in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove([producer_id, product])
                self.producers[producer_id].append(product)
                break
        return None

    def place_order(self, cart_id):
        
        if cart_id not in self.carts.keys():
            return None
        return self.carts[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.daemon = kwargs['daemon']
        self.name = kwargs['name']
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for product in self.products:
                i = 0
                while i < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        sleep(product[2])
                        i += 1
                    else:
                        sleep(self.republish_wait_time)


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
