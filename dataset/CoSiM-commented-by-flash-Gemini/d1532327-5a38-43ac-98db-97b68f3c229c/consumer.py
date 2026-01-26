


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

        self.cart_id = -1


    def run(self):

        for c in self.carts:
            self.cart_id = self.marketplace.new_cart()
            for op in c:
                op_type = op['type']
                if op_type == "add":
                    i = 0
                    while i < op['quantity']:
                        ret = self.marketplace.add_to_cart(self.cart_id, op['product'])
                        if ret is True:
                            i = i + 1
                        else:
                            time.sleep(self.retry_wait_time)
                elif op_type == "remove":
                    i = 0
                    while i < op['quantity']:
                        ret = self.marketplace.remove_from_cart(self.cart_id, op['product'])
                        if ret is True:
                            i = i + 1
                        else:
                            time.sleep(self.retry_wait_time)
            my_cart = self.marketplace.place_order(self.cart_id)
            for p in my_cart:
                print(self.name + ' bought ' + str(p))


from threading import Lock
from logging import getLogger
import logging.handlers


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer

        self.producersList = []
        self.cartsList = []
        self.available_products_pairs = []

        self.add_remove_lock = Lock()
        self.producer_lock = Lock()

        self.info_logger = getLogger(__name__)
        self.info_logger.setLevel(logging.INFO)
        self.info_logger.addHandler(logging.handlers.RotatingFileHandler("marketplace.log"))

    def register_producer(self):
        
        
        self.info_logger.info("Function register_producer with parameters: %s", str(self))

        
        new_producer = []
        self.producersList.append(new_producer)

        
        self.info_logger.info("Function register_producer returns value: "
                              + str(len(self.producersList) - 1))

        return len(self.producersList) - 1


    def publish(self, producer_id, product):
        

        
        self.info_logger.info("Function publish with parameters: "
                              + str(self)
                              + str(producer_id)
                              + str(product))

        self.producer_lock.acquire()

        
        if len(self.producersList[producer_id]) == self.queue_size_per_producer:
            self.producer_lock.release()
            
            self.info_logger.info("Function publish returns value: False")
            return False

        
        self.producersList[producer_id].append(product)
        self.available_products_pairs.append((product, producer_id))

        self.producer_lock.release()

        
        self.info_logger.info("Function publish returns value: True")

        return True


    def new_cart(self):
        

        
        self.info_logger.info("Function new_cart with parameters:"
                              + str(self))

        self.add_remove_lock.acquire()

        
        new_c = []
        self.cartsList.append(new_c)

        
        self.info_logger.info("Function new_cart returns value: "
                              + str(len(self.cartsList) - 1))

        self.add_remove_lock.release()

        return len(self.cartsList) - 1


    def add_to_cart(self, cart_id, product):
        

        
        self.info_logger.info("Function add_to_cart with parameters:"
                              + str(cart_id)
                              + str(product))

        self.add_remove_lock.acquire()

        
        for pair in self.available_products_pairs:
            if pair[0] == product:
                
                self.cartsList[cart_id].append(pair)
                
                self.available_products_pairs.remove(pair)
                self.add_remove_lock.release()
                
                self.info_logger.info("Function add_to_cart returns value: True")
                return True

        self.add_remove_lock.release()

        
        self.info_logger.info("Function add_to_cart returns value: False")
        return False


    def remove_from_cart(self, cart_id, product):
        

        
        self.info_logger.info("Function remove_from_cart with parameters: "
                              + str(cart_id)
                              + str(product))

        self.add_remove_lock.acquire()

        
        for pair in self.cartsList[cart_id]:
            if pair[0] == product:
                
                self.available_products_pairs.append(pair)
                
                self.cartsList[cart_id].remove(pair)
                
                self.info_logger.info("Function remove_from_cart returns value: True")
                self.add_remove_lock.release()
                return True

        
        self.info_logger.info("Function remove_from_cart returns value: False")

        self.add_remove_lock.release()
        return False


    def place_order(self, cart_id):
        

        
        self.info_logger.info("Function place_order with parameters: "
                              + str(cart_id))

        prod_list = []

        self.producer_lock.acquire()

        
        
        for pair in self.cartsList[cart_id]:
            prod_list.append(pair[0])
            self.producersList[pair[1]].remove(pair[0])

        self.producer_lock.release()

        


        self.info_logger.info("Function place_order returns value: "
                              + str(prod_list))

        return prod_list


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

            for p in self.products:
                i = 0
                while i < p[1]:
                    ret = self.marketplace.publish(self.producer_id, p[0])
                    if ret is True:
                        i = i + 1
                        time.sleep(float(p[2]))
                    else:
                        time.sleep(float(self.republish_wait_time))
