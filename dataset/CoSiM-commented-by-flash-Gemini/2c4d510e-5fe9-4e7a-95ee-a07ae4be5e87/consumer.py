/**
 * @file consumer.py
 * @brief Semantic documentation for consumer.py. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */



from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = -1

    def run(self):
        
        for cart in self.carts:
            self.cart_id = self.marketplace.new_cart()
            for command in cart:
                
                if command["type"] == "add":
                    i = 0
                    while i < command["quantity"]:
                        result = self.marketplace.add_to_cart(self.cart_id, command["product"])
                        if result:
                            i = i + 1
                        
                        else:
                            sleep(self.retry_wait_time)
                
                elif command["type"] == "remove":
                    i = 0
                    while i < command["quantity"]:
                        self.marketplace.remove_from_cart(self.cart_id, command["product"])
                        i = i + 1
            
            self.marketplace.place_order(self.cart_id)


from threading import Lock, currentThread


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.queue_list = [[]]
        self.cart_list = [[]]
        self.place_order_lock = Lock()
        self.register_producer_lock = Lock()
        self.cart_lock = Lock()

    def register_producer(self):
        

        
        self.register_producer_lock.acquire()
        self.id_producer += 1
        self.queue_list.append([])
        self.register_producer_lock.release()
        return self.id_producer

    def publish(self, producer_id, product):
        

        
        if len(self.queue_list[int(producer_id)]) >= self.queue_size_per_producer:
            return False

        self.queue_list[int(producer_id)].append(product)
        return True

    def new_cart(self):
        

        
        self.cart_list.append([])
        return len(self.cart_list) - 1

    def add_to_cart(self, cart_id, product):
        

        self.cart_lock.acquire()
        i = 0
        
        for producer_list in self.queue_list:
            for prod in producer_list:
                
                
                
                if prod == product:
                    self.cart_list[cart_id].append([i, prod])
                    producer_list.remove(prod)
                    self.cart_lock.release()
                    return True
            i = i + 1
        self.cart_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        

        self.cart_lock.acquire()
        i = 0
        


        for tup in self.cart_list[cart_id]:
            
            
            if tup[1] == product:
                self.queue_list[tup[0]].append(tup[1])
                self.cart_list[cart_id].pop(i)
                break
            i = i + 1
        self.cart_lock.release()

    def place_order(self, cart_id):
        

        products = []
        
        for tup in self.cart_list[cart_id]:
            products.append(tup[1])
            self.place_order_lock.acquire()
            print(currentThread().getName() + " bought " + str(tup[1]))
            self.place_order_lock.release()

        return products


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
        while 1:
            
            for prod in self.products:
                i = 0
                
                while i < prod[1]:
                    result = self.marketplace.publish(str(self.producer_id), prod[0])
                    if result:
                        sleep(self.republish_wait_time)
                        i = i + 1
                    
                    else:
                        sleep(prod[2])
