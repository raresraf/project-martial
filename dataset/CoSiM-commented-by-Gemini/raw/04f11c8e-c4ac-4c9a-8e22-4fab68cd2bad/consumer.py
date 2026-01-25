

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__()
        self.name = kwargs["name"]
        self.retry_wait_time = retry_wait_time
        self.id_cart = -1  
        self.carts = carts
        self.marketplace = marketplace

    def run(self):
        for cart in self.carts:
            self.id_cart = self.marketplace.new_cart()
            for command in cart:
                comm_type = command["type"]


                product = command["product"]
                quantity = command["quantity"]

                if comm_type == "add":
                    for i in range(quantity):
                        add_result = self.marketplace.add_to_cart(self.id_cart, product)
                        while True:
                            
                            if not add_result:


                                time.sleep(self.retry_wait_time)
                                add_result = self.marketplace.add_to_cart(self.id_cart, product)
                            else:
                                
                                break
                elif comm_type == "remove":
                    for i in range(quantity):
                        
                        remove_result = self.marketplace.remove_from_cart(self.id_cart, product)
                        if not remove_result:  
                            print("INVALID OPERATION RESULT! REMOVED FAILED! EXITING THREAD")
                            return
                else:  
                    print("INVALID OUTPUT! EXITING THREAD")
                    return
            cart_list = self.marketplace.place_order(self.id_cart)
            for item in cart_list:
                print(f"{self.name} bought {item}")


import threading


class Marketplace(object):
    

    def __init__(self, queue_size_per_producer):
        
        self.queues = {}  
                                                                
        self.capacity = queue_size_per_producer
        self.id_producer = -1
        self.id_cart = -1
        self.general_semaphore = threading.Semaphore(1)  
                                                         
        self.carts = {}  

    def register_producer(self):
        
        self.general_semaphore.acquire()


        self.id_producer += 1
        self.queues[self.id_producer] = {}
        self.queues[self.id_producer]["products"] = []
        self.queues[self.id_producer]["semaphore"] = threading.Semaphore(1)
        self.general_semaphore.release()
        return self.id_producer

    def publish(self, producer_id, product):
        


        self.queues[producer_id]["semaphore"].acquire()
        if len(self.queues[producer_id]["products"]) < self.capacity:
            
            self.queues[producer_id]["products"].append(product)
            self.queues[producer_id]["semaphore"].release()
            return True
        self.queues[producer_id]["semaphore"].release()
        
        return False

    def new_cart(self):
        
        self.id_cart += 1
        self.carts[self.id_cart] = []  
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        

        for id_producer, queue_producer in self.queues.items():
            queue_producer["semaphore"].acquire()
            for queue_product in queue_producer["products"]:
                if product == queue_product:
                    
                    queue_producer["products"].remove(queue_product)
                    
                    self.carts[cart_id].append((id_producer, product))
                    queue_producer["semaphore"].release()
                    return True
            queue_producer["semaphore"].release()

        
        return False

    def remove_from_cart(self, cart_id, product):
        
        for id_producer, cart_product in self.carts[cart_id]:
            if product == cart_product:
                self.carts[cart_id].remove((id_producer, cart_product))
                self.queues[id_producer]["semaphore"].acquire()
                self.queues[id_producer]["products"].append(product)
                self.queues[id_producer]["semaphore"].release()
                return True
        return False

    def place_order(self, cart_id):
        
        result = []
        for _, product in self.carts[cart_id]:
            result.append(product)

        return result


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__(daemon=True)
        self.products = products
        self.marketplace = marketplace
        self.name = kwargs["name"]
        self.republish_wait_time = republish_wait_time
        self.id_producer = -1  
                               

    def run(self):
        self.id_producer = self.marketplace.register_producer()

        while True:
            for product_struct in self.products:
                product = product_struct[0]  
                quantity = product_struct[1]  
                sleep_time = product_struct[2]  

                for _ in range(quantity):
                    publish_result = self.marketplace.publish(self.id_producer, product)

                    if not publish_result:
                        while True:
                            
                            time.sleep(self.republish_wait_time)
                            publish_result = self.marketplace.publish(self.id_producer, product)
                            if publish_result:
                                
                                time.sleep(sleep_time)
                                break
                    else:
                        time.sleep(sleep_time)
