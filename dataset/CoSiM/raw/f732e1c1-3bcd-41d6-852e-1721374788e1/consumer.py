


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        id_cart = self.marketplace.new_cart()
        for cart in self.carts:
            for operation in cart:


                qty = operation['quantity']
                type_op = operation['type']
                product = operation['product']
                while qty > 0:
                    if type_op == 'add':
                        ret_value = self.marketplace.add_to_cart(id_cart, product)
                        if ret_value:
                            qty = qty - 1
                        else:
                            sleep(self.retry_wait_time)


                    if type_op == 'remove':
                        self.marketplace.remove_from_cart(id_cart, product)
                        qty = qty - 1
        cart_list = self.marketplace.place_order(id_cart)
        for product in cart_list:
            print(self.kwargs['name'] + " bought " + str(product))

from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size = queue_size_per_producer
        self.producers = []
        self.carts = []
        self.lock_producers = Lock()
        self.lock_consumer = Lock()
        self.producers_locks = []

    def register_producer(self):
        
        with self.lock_producers:
            id_new_producer = len(self.producers)
            self.producers.append(list())
            self.producers_locks.append(Lock())

        return id_new_producer

    def publish(self, producer_id, product):
        
        if len(self.producers[producer_id]) == self.queue_size:
            return False

        self.producers[producer_id].append(product)
        return True

    def new_cart(self):
        
        with self.lock_consumer:
            id_new_cart = len(self.carts)
            self.carts.append(list())

        return id_new_cart

    def add_to_cart(self, cart_id, product):
        
        for id_producer in range(len(self.producers)):
            if product in self.producers[id_producer]:
                with self.producers_locks[id_producer]:
                    if product in self.producers[id_producer]:
                        self.producers[id_producer].remove(product)
                        self.carts[cart_id].append((product, id_producer))
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        for (current_product, id_producer) in self.carts[cart_id]:
            if current_product == product:
                self.producers[id_producer].append(product)
                self.carts[cart_id].remove((current_product, id_producer))
                break

    def place_order(self, cart_id):
        
        list_cart = list()
        for element in self.carts[cart_id]:
            list_cart.append(element[0])
        self.carts[cart_id].clear()
        return list_cart


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, group=None, target=None, name=kwargs['name'], daemon=kwargs['daemon'])
        self.marketplace = marketplace
        self.products = products
        self.republish_wait_time = republish_wait_time


        self.kwargs = kwargs

    def run(self):
        id_producer = self.marketplace.register_producer()
        while True:
            for (product, qty, time) in self.products:
                while qty > 0:
                    ret_value = self.marketplace.publish(id_producer, product)
                    if ret_value:
                        sleep(time)
                        qty = qty - 1
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
