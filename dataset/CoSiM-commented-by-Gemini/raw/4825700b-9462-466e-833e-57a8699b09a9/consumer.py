




from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def add_product(self, cart_id, product):
        
        added = False
        while not added:
            added = self.marketplace.add_to_cart(cart_id, product)
            if not added:
                time.sleep(self.retry_wait_time)

    def run(self):
        
        carts_id = []
        
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            carts_id.append(cart_id)
            
            for command in cart:
                if command["type"] == "add":
                    for _ in range(command["quantity"]):


                        self.add_product(cart_id, command["product"])
                else:
                    for _ in range(command["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, command["product"])

        
        for cart_id in carts_id:
            products = self.marketplace.place_order(cart_id)
            for product in products:
                print(f'{self.name} bought {product}', flush=True)




from threading import Lock


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer

        
        self.register_lock = Lock()
        self.producer_id = 0

        
        self.cart_lock = Lock()
        self.cart_id = 0

        
        self.products = []
        
        self.carts = []

        
        self.sizes = []
        
        
        self.producers_lock = []

    def register_producer(self):
        

        
        self.register_lock.acquire()
        id_copy = self.producer_id
        self.producer_id = self.producer_id + 1
        self.register_lock.release()

        
        
        self.products.append({})
        self.sizes.append(0)
        self.producers_lock.append(Lock())

        return id_copy

    def publish(self, producer_id, product):
        

        
        
        
        self.producers_lock[producer_id].acquire()
        if self.sizes[producer_id] < self.queue_size_per_producer:

            
            
            if product in self.products[producer_id]:
                self.products[producer_id][product] += 1
            else:
                self.products[producer_id][product] = 0

            
            self.sizes[producer_id] += 1
            self.producers_lock[producer_id].release()
            return True

        self.producers_lock[producer_id].release()
        return False

    def new_cart(self):
        

        
        self.cart_lock.acquire()
        id_copy = self.cart_id
        self.cart_id = self.cart_id + 1
        self.cart_lock.release()

        
        self.carts.append({})

        return id_copy

    def add_to_cart(self, cart_id, product):
        

        


        for producer_id in range(len(self.products)):

            
            
            
            self.producers_lock[producer_id].acquire()
            if product in self.products[producer_id]:

                
                self.products[producer_id][product] -= 1
                self.sizes[producer_id] -= 1

                
                if self.products[producer_id][product] == 0:
                    self.products[producer_id].pop(product)
                self.producers_lock[producer_id].release()

                
                
                if (product, producer_id) in self.carts[cart_id]:
                    new_quantity = self.carts[cart_id].get((product, producer_id)) + 1
                    self.carts[cart_id].update({(product, producer_id): new_quantity})
                else:
                    self.carts[cart_id].update({(product, producer_id): 1})

                return True

            self.producers_lock[producer_id].release()
        return False

    def remove_from_cart(self, cart_id, product):
        

        
        producer_id = -1
        for product_tuple in self.carts[cart_id].keys():
            if product == product_tuple[0]:
                producer_id = product_tuple[1]
                break

        
        
        
        self.producers_lock[producer_id].acquire()
        if product in self.products[producer_id]:
            self.products[producer_id][product] += 1
        else:
            self.products[producer_id][product] = 0

        
        self.sizes[producer_id] += 1
        self.producers_lock[producer_id].release()

        
        
        new_quantity = self.carts[cart_id].get((product, producer_id)) - 1
        self.carts[cart_id].update({(product, producer_id): new_quantity})
        if self.carts[cart_id].get((product, producer_id)) == 0:
            self.carts[cart_id] = {key: val for key, val in self.carts[cart_id].items()
                                   if key != (product, producer_id)}

    def place_order(self, cart_id):
        

        simple_list = []

        
        for product_tuple in self.carts[cart_id]:
            for _ in range(self.carts[cart_id][product_tuple]):
                simple_list.append(product_tuple[0])

        return simple_list




from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        
        while True:
            
            for product in self.products:
                published = False
                while not published:
                    published = self.marketplace.publish(self.producer_id, product[0])
                    
                    
                    if not published:
                        time.sleep(self.republish_wait_time)
                    else:
                        time.sleep(product[2])
