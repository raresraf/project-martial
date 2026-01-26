


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for opp in cart:
                for i in range(0, opp["quantity"]):
                    if opp["type"] == "add":
                        while self.marketplace.add_to_cart(cart_id, opp["product"]) == False:
                            sleep(self.retry_wait_time)
                    elif opp["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, opp["product"])

            prod_list = self.marketplace.place_order(cart_id)

            for product in prod_list:
                print(str(self.name) + " bought " + str(product))




from threading import Lock

class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0


        self.consumer_id = 0
        self.prod_dict = {}
        self.cart_dict = {}
        self.lock_add_cart = Lock()
        self.lock_publish = Lock()

        pass

    
    def register_producer(self):
        

        self.producer_id += 1
        self.prod_dict[self.producer_id] = []


        return self.producer_id
        pass

    def publish(self, producer_id, product):
        

        self.lock_publish.acquire()
        if len(self.prod_dict[producer_id]) < self.queue_size_per_producer:
            self.prod_dict[producer_id].append(product)
            self.lock_publish.release()
            return True
        self.lock_publish.release()
        return False
        pass

    def new_cart(self):
        
        self.consumer_id += 1
        self.cart_dict[self.consumer_id] = []


        return self.consumer_id
        pass

    def add_to_cart(self, cart_id, product):
        

        self.lock_add_cart.acquire()

        for prod_id in self.prod_dict.keys():
            for p in self.prod_dict[prod_id]:
                if p == product:
                    self.prod_dict[prod_id].remove(product)
                    self.cart_dict[cart_id].append([product, prod_id])
                    self.lock_add_cart.release()
                    return True
        self.lock_add_cart.release()


        return False
        pass

    def remove_from_cart(self, cart_id, product):
        
        

        for prod in self.cart_dict[cart_id]:
            if prod[0] == product:
                self.cart_dict[cart_id].remove(prod)
                self.prod_dict[prod[1]].append(prod[0])
                break

    def place_order(self, cart_id):
        
        prod_list = []
        for prod in self.cart_dict[cart_id]:
           prod_list.append(prod[0])
        return prod_list>>>> file: producer.py


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs

    def run(self):
        producer_id = self.marketplace.register_producer()
        while True:

            for product in self.products:
                sleep(product[2])
                for i in range(0, product[1]):
                    while self.marketplace.publish(producer_id, product[0]) == False:
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
