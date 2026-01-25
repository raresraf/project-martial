


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()
            
            for command in cart:
                command_type = command["type"]
                product = command["product"]
                quantity = command["quantity"]
                
                if command_type == "add":


                    for _ in range(quantity):
                        while True:
                            added = self.marketplace.add_to_cart(cart_id, product)
                            if added:
                                break
                            sleep(self.retry_wait_time)
                elif command_type == "remove":


                    for _ in range(quantity):
                        while True:
                            removed = self.marketplace.remove_from_cart(cart_id, product)
                            if removed:
                                break
                            sleep(self.retry_wait_time)
            products_bought = self.marketplace.place_order(cart_id)
            for product in products_bought:
                print(f"{self.getName()} bought {product}")


from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0 


        self.cart_id = 0 
        self.queues = [] 
        self.carts = [] 
        self.mutex = Lock() 
        self.products_dict = {} 

    def register_producer(self):
        

        self.mutex.acquire() 
        producer_id = self.producer_id
        self.producer_id += 1 
        self.queues.append([]) 
        self.mutex.release()
        return str(producer_id)

    def publish(self, producer_id, product):
        

        producer_index = int(producer_id) 
        if len(self.queues[producer_index]) == self.queue_size_per_producer: 
            return False
        self.queues[producer_index].append(product) 
        self.products_dict[product] = producer_index 
        return True

    def new_cart(self):
        

        self.mutex.acquire()


        cart_id = self.cart_id
        self.cart_id += 1
        self.mutex.release()
        self.carts.append([]) 
        return cart_id

    def add_to_cart(self, cart_id, product):
        

        
        is_product = False
        for queue in self.queues:
            if product in queue:
                is_product = True
                queue.remove(product) 
                break
        if not is_product:
            return False
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        



        if product not in self.carts[cart_id]:
            return False
        producer_idx = self.products_dict[product] 
        if len(self.queues[producer_idx]) == self.queue_size_per_producer: 
            return False
        self.carts[cart_id].remove(product) 
        self.queues[producer_idx].append(product) 
        return True

    def place_order(self, cart_id):
        

        cart_content = self.carts[cart_id] 
        self.carts[cart_id] = [] 
        return cart_content 


from threading import Thread
from time import sleep

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
                quantity = product[1]
                production_time = product[2]
                
                for _ in range(0, quantity):
                    
                    while True:
                        published = self.marketplace.publish(self.producer_id, product[0])
                        if published:
                            sleep(production_time)
                            break
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
