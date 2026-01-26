


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, name=kwargs["name"], kwargs=kwargs)
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
    def run(self):
        for lista_cumparaturi in self.carts:
            
            cart_id = self.marketplace.new_cart()
            
            failed = True
            while failed:
                failed = False
                for operatie in lista_cumparaturi:
                    
                    
                    cantitate = operatie["quantity"]
                    for _ in range(cantitate):
                        if operatie["type"] == "add":
                            if self.marketplace.add_to_cart(cart_id, operatie["product"]):
                                
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                
                                
                                failed = True
                                break
                        elif operatie["type"] == "remove":
                            if self.marketplace.remove_from_cart(cart_id, operatie["product"]):
                                
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                
                                
                                failed = True
                                break
            if failed:
                
                sleep(self.retry_wait_time)
            else:
                
                self.marketplace.place_order(cart_id)
                
                
                


from threading import Lock
from threading import current_thread


class PublishedProduct:
    
    def __init__(self, product):
        self.product = product
        self.reserved = False

    
    def __eq__(self, obj):
        ret = isinstance(obj, PublishedProduct) and self.reserved == obj.reserved
        return ret and obj.product == self.product

class ProductsList:
    
    def __init__(self, maxsize):
        
        self.lock = Lock()
        self.list = []
        self.maxsize = maxsize

    def put(self, item):
        
        with self.lock:
            
            if self.maxsize == len(self.list):
                return False
            self.list.append(item)
        return True

    def rezerva(self, item):
        
        
        item = PublishedProduct(item)
        with self.lock:
            
            if item in self.list:
                self.list[self.list.index(item)].reserved = True
                return True
        return False

    def anuleaza_rezervarea(self, item):
        
        
        item = PublishedProduct(item)
        item.reserved = True
        with self.lock:
            
            self.list[self.list.index(item)].reserved = False

    def remove(self, item):
        
        
        product = PublishedProduct(item)
        product.reserved = True
        with self.lock:
            self.list.remove(product)
            return item

class Cart:
    

    def __init__(self):
        
        self.products = []

    def add_product(self, product, producer_id):
        
        self.products.append((product, producer_id))

    def remove_product(self, product):
        
        for item in self.products:
            if item[0] == product:
                self.products.remove(item)
                return item[1]

        return None

    def get_products(self):
        
        return self.products

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.print_lock = Lock()
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_queues = {}


        self.generator_id_producator = 0
        self.generator_id_producator_lock = Lock()

        self.carts = {}
        self.cart_id_generator = 0
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        
        id_producator = None
        with self.generator_id_producator_lock:
            
            id_producator = self.generator_id_producator
            
            self.generator_id_producator += 1
            
            self.producer_queues[id_producator] = ProductsList(self.queue_size_per_producer)

        return id_producator

    def publish(self, producer_id, product):
        
        
        
        return self.producer_queues[producer_id].put(PublishedProduct(product))

    def new_cart(self):
        
        with self.cart_id_generator_lock:
            
            current_cart_id = self.cart_id_generator
            
            self.cart_id_generator += 1
            
            
            self.carts[current_cart_id] = Cart()

            return current_cart_id

    def add_to_cart(self, cart_id, product):
        
        producers_num = 0
        with self.generator_id_producator_lock:
            producers_num = self.generator_id_producator

        for producer_id in range(producers_num):
            
            if self.producer_queues[producer_id].rezerva(product):
                self.carts[cart_id].add_product(product, producer_id)
                return True

        return False

    def remove_from_cart(self, cart_id, product):
        
        
        producer_id = self.carts[cart_id].remove_product(product)
        if producer_id is None:
            return False
        self.producer_queues[producer_id].anuleaza_rezervarea(product)
        return True

    def place_order(self, cart_id):
        
        
        
        
        lista = list()
        for (produs, producer_id) in self.carts[cart_id].get_products():
            lista.append(self.producer_queues[producer_id].remove(produs))
            with self.print_lock:
                print(f"{current_thread().getName()} bought {produs}")
        return lista


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        
        Thread.__init__(self, name=kwargs["name"], daemon=kwargs["daemon"], kwargs=kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        
        producer_id = self.marketplace.register_producer()
        
        while True:
            for (product, cantitate, production_time) in self.products:
                
                sleep(production_time)
                
                for _ in range(cantitate):
                    
                    while not self.marketplace.publish(producer_id, product):
                        
                        sleep(self.republish_wait_time)
