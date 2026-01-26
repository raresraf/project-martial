

from threading import Thread
from time import sleep


class Consumer(Thread):
    
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        
        for shopping in self.carts:
            num_cart = self.marketplace.new_cart()
            for product in shopping:
                number_action = int(product['quantity'])
                command = product['type']
                name_product = product['product']
                while number_action != 0:
                    if command == "add":
                        if self.marketplace.add_to_cart(num_cart, name_product):
                            number_action = number_action - 1
                        else:
                            sleep(self.wait_time)
                    if command == "remove":
                        self.marketplace.remove_from_cart(num_cart, name_product)
                        number_action = number_action - 1


            shopping = self.marketplace.place_order(num_cart)
            for _, product in shopping:
                print(self.name, "bought", product)

from threading import Lock


class Marketplace:
    
    producers = {}
    consumers = {}
    id_prod = 1
    id_cons = 1
    lock_producer = Lock()
    lock_cart = Lock()
    def __init__(self, queue_size_per_producer):
        
        self.size = queue_size_per_producer


    def register_producer(self):
        
        self.lock_producer.acquire()
        products = []
        self.producers[self.id_prod] = products
        self.id_prod = self.id_prod+1
        self.lock_producer.release()
        return self.id_prod-1


    def publish(self, producer_id, product):
        


        if len(self.producers[producer_id]) == self.size:
            return False
        self.producers[producer_id].append(product)
        return True


    def new_cart(self):
        
        self.lock_cart.acquire()
        cart = []
        self.consumers[self.id_cons] = cart
        self.id_cons = self.id_cons + 1
        self.lock_cart.release()
        return self.id_cons - 1

    def add_to_cart(self, cart_id, product):
        
        for producer in self.producers:
            for prod in self.producers[producer]:
                if product == prod:


                    self.consumers[cart_id].insert(0, [producer, product])
                    self.producers[producer].remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        for cart in self.consumers:
            if cart == cart_id:


                for index, prod in self.consumers[cart]:
                    if prod == product:
                        self.consumers[cart_id].remove([index, product])
                        self.producers[index].append(product)
                        return None
        return None

    def place_order(self, cart_id):
        
        return self.consumers[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time


    def run(self):
        
        id_producer = self.marketplace.register_producer()
        while True:
            for product in self.products:
                name_product = product[0]
                number_pieces = int(product[1])
                time_product = product[2]

                while number_pieces != 0:
                    if self.marketplace.publish(id_producer, name_product):
                        sleep(time_product)
                    else:
                        sleep(self.wait_time)
                    number_pieces = number_pieces - 1
