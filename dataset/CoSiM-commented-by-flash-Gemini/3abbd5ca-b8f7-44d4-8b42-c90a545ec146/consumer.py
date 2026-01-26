


from threading import Thread
import time

ADD_COMMAND = "add"
REMOVE_COMMAND = "remove"
COMMAND_TYPE = "type"
ITEM_QUANTITY = "quantity"
PRODUCT = "product"
NAME = "name"

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs[NAME]

    
    def run(self):
        id_cart = self.marketplace.new_cart()

        for item in self.carts:
            for command in item:

                if command[COMMAND_TYPE] == ADD_COMMAND:

                    for _ in range(command[ITEM_QUANTITY]):
                        while self.marketplace.add_to_cart(id_cart, command[PRODUCT]) is False:
                            time.sleep(self.retry_wait_time)

                elif command[COMMAND_TYPE] == REMOVE_COMMAND:

                    for _ in range(command[ITEM_QUANTITY]):
                        self.marketplace.remove_from_cart(id_cart, command[PRODUCT])

        order_result = self.marketplace.place_order(id_cart)

        for item in order_result:

            self.marketplace.lock.acquire()
            print(self.consumer_name + " bought " + str(item[1]))
            self.marketplace.lock.release()




from logging.handlers import RotatingFileHandler
from threading import Lock
import logging

class Marketplace:
    

    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer

        self.producer_id = 0
        self.consumer_id = 0

        self.products = []
        self.producers = []
        self.carts = []

        self.lock = Lock()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        file_handler = RotatingFileHandler("marketplace.log")
        self.logger.addHandler(file_handler)

    
    def register_producer(self):
        

        self.logger.info("Entered method: register_producer")
        self.lock.acquire()

        self.producer_id += 1
        producer_id = self.producer_id

        self.lock.release()

        self.logger.info("Exited method: register_producer")
        return producer_id


    
    def publish(self, producer_id, product):
        

        self.logger.info("Entered method: publish")
        self.logger.info("Params: producer_id: " + str(producer_id)
        + ", product: " + str(product.name))
        self.lock.acquire()

        if self.producers[producer_id - 1].nr_products < self.queue_size_per_producer:

            self.products.append((producer_id, product))
            self.producers[producer_id - 1].nr_products += 1
            self.lock.release()
            self.logger.info("Exited method: publish")
            return True

        self.lock.release()
        self.logger.info("Exited method: publish")
        return False

    
    def new_cart(self):
        

        self.logger.info("Entered method: new_cart")
        self.lock.acquire()

        self.consumer_id += 1
        consumer_id = self.consumer_id

        self.carts.append([])

        self.lock.release()

        self.logger.info("Exited method: new_cart")
        return consumer_id

    
    def add_to_cart(self, cart_id, product):
        

        self.logger.info("Entered method: add_to_cart")
        self.logger.info("Params: cart_id: " + str(cart_id) + ", product: " + str(product.name))
        self.lock.acquire()
        for item in self.products:
            if product == item[1]:

                self.carts[cart_id - 1].append(item)
                self.products.remove(item)
                self.producers[item[0] - 1].nr_products -= 1
                self.lock.release()
                self.logger.info("Exited method: add_to_cart")
                return True

        self.lock.release()
        self.logger.info("Exited method: add_to_cart")
        return False


    
    def remove_from_cart(self, cart_id, product):
        

        self.logger.info("Entered method: remove_from_cart")
        self.logger.info("Params: cart_id: " + str(cart_id) + ", product: " + str(product.name))
        self.lock.acquire()
        for item in self.carts[cart_id - 1]:
            if product == item[1]:

                self.carts[cart_id - 1].remove(item)
                self.products.append(item)
                self.producers[item[0] - 1].nr_products += 1
                self.lock.release()
                self.logger.info("Exited method: remove_from_cart")
                return

        self.lock.release()

    
    def place_order(self, cart_id):
        
        self.logger.info("Entered method: place_order")
        self.logger.info("Exited method: place_order")
        return self.carts[cart_id - 1]


from threading import Thread
import time


class Producer(Thread):
    

    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)

        self.nr_products = 0
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.marketplace.producers.append(self)
        self.producer_id = self.marketplace.register_producer()

    
    def run(self):
        while True:
            for item in self.products:
                for _ in range(item[1]):

                    while self.marketplace.publish(self.producer_id, item[0]) is False:
                        time.sleep(self.republish_wait_time)

                    time.sleep(item[2])


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
