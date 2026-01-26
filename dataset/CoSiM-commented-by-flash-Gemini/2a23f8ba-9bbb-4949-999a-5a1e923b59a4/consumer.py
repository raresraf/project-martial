


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):

        for check in self.carts:
            
            id_cart = self.marketplace.new_cart()
            
            for demand in check:
                if demand["type"] == "add":


                    i = 0
                    
                    while i < (demand["quantity"]):
                        add_cart = self.marketplace.add_to_cart(id_cart, demand["product"])
                        while not add_cart:
                            time.sleep(self.retry_wait_time)
                            add_cart = self.marketplace.add_to_cart(id_cart, demand["product"])
                        i = i + 1


                elif demand["type"] == "remove":
                    i = 0
                    while i < (demand["quantity"]):
                        self.marketplace.remove_from_cart(id_cart, demand["product"])
                        i = i + 1
            place_order = self.marketplace.place_order(id_cart)

            for product in place_order:
                print(self.name + " bought " + str(product))

from threading import Lock
class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.id_producer = 0 
        self.id_cart = 0 

        self.producers_dictionary = {}  
        self.carts_dictionary = {}  

        self.nr_prod = []   
        self.mutex = Lock()

    def register_producer(self):
        
        
        with self.mutex:
            
            self.id_producer = len(self.nr_prod)
            self.producers_dictionary[self.id_producer] = []
        
        self.nr_prod.append(0)
        return self.id_producer

    def publish(self, producer_id, product):
        
        
        if self.nr_prod[producer_id] >= self.queue_size_per_producer:
            return False
        
        self.producers_dictionary[producer_id].append(product)
        self.nr_prod[producer_id] += 1
        return True

    def new_cart(self):
        
        
        with self.mutex:
            
            self.id_cart = self.id_cart + 1
            self.carts_dictionary[self.id_cart] = []
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        
        
        for prod_count in self.producers_dictionary:
            if product in self.producers_dictionary[prod_count]:
                self.carts_dictionary[cart_id].append((product, prod_count))
                self.producers_dictionary[prod_count].remove(product)
                
                if self.nr_prod[prod_count]:
                    self.nr_prod[prod_count] -= 1
                return True

        return False


    def remove_from_cart(self, cart_id, product):
        
        
        try:
            with self.mutex:
                for cart in self.carts_dictionary[cart_id]:
                    if product == cart[0]:
                        self.carts_dictionary[cart_id].remove(cart)
                        self.producers_dictionary[cart[1]].append(product)
                        self.nr_prod[cart[1]] += 1
                        return

        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt')

    def place_order(self, cart_id):
        
        cart_dic = self.carts_dictionary[cart_id]
        products_ordered = []
        
        for cart in cart_dic:
            products_ordered.append(cart[0])
        self.carts_dictionary.pop(cart_id)

        return products_ordered


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.producer_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for prod in self.products:
                i = 0
                


                while i < (prod[1]):
                    add_prod = self.marketplace.publish(self.producer_id, prod[0])

                    
                    while not add_prod:
                        time.sleep(self.republish_wait_time)
                        add_prod = self.marketplace.publish(self.producer_id, prod[0])

                    time.sleep(prod[2])
                    i = i + 1
