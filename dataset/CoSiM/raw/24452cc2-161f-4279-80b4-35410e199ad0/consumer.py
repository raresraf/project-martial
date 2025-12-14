


from threading import Thread
from time import sleep

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
                command_type = command["type"]


                product = command["product"]
                quantity = command["quantity"]

                if command_type == "add":
                    for _ in range(quantity):
                        add_result = self.marketplace.add_to_cart(self.id_cart, product)
                        while True:
                            
                            
                            
                            if not add_result:


                                sleep(self.retry_wait_time)
                                add_result = self.marketplace.add_to_cart(self.id_cart, product)
                            else:
                                
                                break
                elif command_type == "remove":
                    for _ in range(quantity):
                        remove_result = self.marketplace.remove_from_cart(self.id_cart, product)
                        if not remove_result:
                            
                            
                            print("INVALID REMOVE RESULT; EXITING")
                            return
                else:
                    
                    print("INVALID OPERATION; EXITING")
                    return
            cart_list = self.marketplace.place_order(self.id_cart)
            
            
            with self.marketplace.print_semaphore:
                for item in cart_list:
                    if item is not None:
                        print(f"{self.name} bought {item}")


from threading import Semaphore
import logging
from logging.handlers import RotatingFileHandler
import time

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG) 
        formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
        formatter.converter = time.gmtime 
        
        handler = RotatingFileHandler('marketplace.log', maxBytes = 1000000, backupCount = 5)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.queues = {} 
                         
                         
                                                     
        self.capacity = queue_size_per_producer
        self.id_producer = -1
        self.id_cart = -1
        self.print_semaphore = Semaphore(1) 
        self.carts_semaphore = Semaphore(1) 
        self.general_semaphore = Semaphore(1) 
                                                        
        self.carts = {} 
                        
                        
        
        
        
        
        
        
        self.logger.info(, queue_size_per_producer)


    def register_producer(self):
        
        
        self.logger.info()
        with self.general_semaphore:


            self.id_producer += 1
            self.queues[self.id_producer] = {} 
            self.queues[self.id_producer]["products"] = [] 
                                                           
            self.queues[self.id_producer]["semaphore"] = Semaphore(1) 
                                                                      
                                                                      
                                                                      
        
        self.logger.info(, self.id_producer)
        return self.id_producer


    def publish(self, producer_id, product):
        
        
        self.logger.info(, producer_id, product)


        self.queues[producer_id]["semaphore"].acquire()
        if len(self.queues[producer_id]["products"]) < self.capacity:
            
            self.queues[producer_id]["products"].append((product, True))
            self.queues[producer_id]["semaphore"].release()
            
            self.logger.info()
            return True
        self.queues[producer_id]["semaphore"].release()
        
        
        self.logger.info()
        return False


    def new_cart(self):
        
        
        self.logger.info()
        with self.carts_semaphore:
            self.id_cart += 1
            self.carts[self.id_cart] = [] 
            
            self.logger.info(,
self.id_cart)
        return self.id_cart


    def add_to_cart(self, cart_id, product):
        
        
        self.logger.info(, cart_id, product)
        for id_producer, queue_producer in self.queues.items():
            queue_producer["semaphore"].acquire()
            for idx, queue_item in enumerate(queue_producer["products"]):
                
                if product == queue_item[0] and queue_item[1] is True:
                    
                    self.carts[cart_id].append((id_producer, product))
                    
                    queue_producer["products"][idx] = (queue_item[0], False)
                    queue_producer["semaphore"].release()
                    
                    self.logger.info(Returning from add_to_cart function with value False
        Removes a product from cart.

        :type cart_id: Int
        :param cart_id: id cart

        :type product: Product
        :param product: the product to remove from cart
        Entering remove_from_cart function with parameters \
cart_id = %s, product = %sReturning from add_to_cart function with value TrueReturning from add_to_cart function with value False
        Return a list with all the products in the cart.
        And removes those products from their producers' queue

        :type cart_id: Int
        :param cart_id: id cart
        Entering place_order function with parameters \
cart_id = %sReturning from place_order function with result = %s
This module represents the Producer.

Computer Systems Architecture Course
Assignment 1
March-April 2022
IONITA Dragos 341 C1

    Class that represents a producer.
    
        Constructor.

        @type products: List()
        @param products: a list of products that the producer will produce

        @type marketplace: Marketplace
        @param marketplace: a reference to the marketplace

        @type republish_wait_time: Time
        @param republish_wait_time: the number of seconds that a producer must
        wait until the marketplace becomes available

        @type kwargs:


        @param kwargs: other arguments that are passed to the Thread's __init__()
        
This module offers the available Products.

Computer Systems Architecture Course
Assignment 1
March-April 2022
IONITA Dragos 341 C1

    Class that represents a product.
    
    Tea products
    
    Coffee products
    """
    acidity: str
    roast_level: str
