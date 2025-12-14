


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        super(Consumer, self).__init__(**kwargs)

    def run(self):

        
        for cart in self.carts:

            
            cart_id = self.marketplace.new_cart()

            


            for command in cart:
                if command['type'] == 'add':
                    for i in range(command['quantity']):

                        
                        while not self.marketplace.add_to_cart(cart_id, command['product']):


                            sleep(self.retry_wait_time)

                elif command['type'] == 'remove':
                    for i in range(command['quantity']):
                        self.marketplace.remove_from_cart(cart_id, command['product'])

            
            order_list = self.marketplace.place_order(cart_id)

            
            for ol in order_list:
                print(self.name, end=" ")
                print("bought ", end="")
                print(ol)>>>> file: marketplace.py



class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.producer_count = 0
        self.producers_list = []
        self.carts_count = 0
        self.carts_list = []
        self.products_q = []
        self.products_count = 0
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        
        self.producer_count = self.producer_count + 1
        self.producers_list.append([self.producer_count, 0])
        return self.producer_count

    def publish(self, producer_id, product):
        

        
        for p in self.producers_list:
            if p[0] == producer_id:

                
                if p[1] == self.queue_size_per_producer:
                    return False

                self.products_count = self.products_count + 1

                
                self.products_q.append(
                    [self.products_count, product, producer_id, 0])
                p[1] = p[1] + 1
                return True

    def new_cart(self):
        
        self.carts_count = self.carts_count + 1
        self.carts_list.append([self.carts_count])
        return self.carts_count

    def add_to_cart(self, cart_id, product):
        

        
        for c in self.carts_list:
            if c[0] == cart_id:

                
                for i in self.products_q:

                    
                    if i[1] == product and i[3] == 0:

                        
                        c.append(i)

                        
                        i[3] = 1
                        return True
                return False
        return False

    def remove_from_cart(self, cart_id, product):
        
        
        for c in self.carts_list:
            if c[0] == cart_id:

                
                for i in self.products_q:
                    if i[1] == product:

                        
                        for x, y in enumerate(c[1:]):
                            if y[1] == product:

                                
                                c.pop(x+1)

                                
                                i[3] = 0
                                break
                        break
                break

    def place_order(self, cart_id):
        
        order_list = []

        
        for cart in self.carts_list:
            if cart[0] == cart_id:

                
                products = cart[1:]

                
                for pr in products:

                    
                    for x, y in enumerate(self.products_q):
                        if y[0] == pr[0]:

                            
                            self.products_q.pop(x)

                            
                            for producer in self.producers_list:
                                if y[2] == producer[0]:

                                    
                                    producer[1] = producer[1] - 1
                                    break
                            break

                    order_list.append(pr[1])
                return order_list>>>> file: producer.py


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super(Producer, self).__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):

        
        self.id = self.marketplace.register_producer()

        while True:
            for p in self.products:
                q = p[1]
                for i in range(q):

                    
                    published = self.marketplace.publish(self.id, p[0])

                    
                    if not published:
                        i = i - 1
                        sleep(self.republish_wait_time)
                    sleep(p[2])


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
