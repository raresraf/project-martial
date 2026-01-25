


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]
    def run(self):
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                
                for _ in range(operation["quantity"]):
                    if operation["type"] == "add":
                        while not self.marketplace.add_to_cart(cart_id, operation["product"]):
                            sleep(self.retry_wait_time) 
                    elif operation["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, operation["product"])

            products = self.marketplace.place_order(cart_id)

            
            for product in products:
                print("{0} bought {1}".format(self.name, product))


from threading import Lock
from queue import Full

class SafeList:
    
    def __init__(self, maxsize=0):
        
        self.mutex = Lock()
        self.list = []
        self.maxsize = maxsize

    def put(self, item):
        
        with self.mutex:
            if self.maxsize != 0 and self.maxsize == len(self.list):
                raise Full

            self.list.append(item)

    def put_anyway(self, item):
        
        with self.mutex:
            self.list.append(item)

    def remove(self, item):
        
        with self.mutex:
            if item not in self.list:
                return False

            self.list.remove(item)
            return True

class Cart:
    

    def __init__(self):
        
        self.products = []

    def add_product(self, product, producer_id):
        
        self.products.append({"product": product, "producer_id": producer_id})

    def remove_product(self, product):
        
        for item in self.products:
            if item["product"] == product:
                self.products.remove(item)
                return item["producer_id"]

        return None

    def get_products(self):
        
        return map(lambda item: item["product"], self.products)

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer


        self.producer_queues = {}
        self.producer_id_generator = 0
        self.producer_id_generator_lock = Lock()

        self.carts = {}
        self.cart_id_generator = 0
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        
        with self.producer_id_generator_lock:
            current_prod_id = self.producer_id_generator
            self.producer_queues[current_prod_id] = SafeList(maxsize=self.queue_size_per_producer)

            self.producer_id_generator += 1
            return current_prod_id

    def publish(self, producer_id, product):
        
        try:
            self.producer_queues[producer_id].put(product)
            return True
        except Full:
            return False

    def new_cart(self):
        
        with self.cart_id_generator_lock:
            current_cart_id = self.cart_id_generator
            self.carts[current_cart_id] = Cart()

            self.cart_id_generator += 1
            return current_cart_id

    def add_to_cart(self, cart_id, product):
        
        producers_num = 0
        with self.producer_id_generator_lock:
            producers_num = self.producer_id_generator

        for producer_id in range(producers_num):
            
            if self.producer_queues[producer_id].remove(product):
                self.carts[cart_id].add_product(product, producer_id)
                return True

        return False

    def remove_from_cart(self, cart_id, product):
        
        producer_id = self.carts[cart_id].remove_product(product)

        
        self.producer_queues[producer_id].put_anyway(product)

    def place_order(self, cart_id):
        
        return self.carts[cart_id].get_products()


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        
        Thread.__init__(self, daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        producer_id = self.marketplace.register_producer()

        while True:
            for (product, quantity, production_time) in self.products:
                
                sleep(production_time)

                for _ in range(quantity):
                    
                    while not self.marketplace.publish(producer_id, product):
                        
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
