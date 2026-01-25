


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        id_ = self.marketplace.new_cart()
        for cart in self.carts:
            for entry in cart:
                action = entry["type"]
                product = entry["product"]
                quantity = entry["quantity"]
                counter = 0
                if action == "add":
                    while counter < quantity:
                        added = self.marketplace.add_to_cart(id_, product)
                        if added:
                            counter += 1
                        else:
                            sleep(self.retry_wait_time)
                else:
                    while counter < quantity:
                        self.marketplace.remove_from_cart(id_, product)
                        counter += 1
        new_cart = self.marketplace.place_order(id_)
        for product in new_cart:
            print(self.name, "bought", product, flush=True)


from threading import Lock
import logging
from logging.handlers import RotatingFileHandler

class Marketplace:
    

    logger = logging.getLogger('marketplace.log')
    logger.setLevel(logging.INFO)
    logger.addHandler(RotatingFileHandler('marketplace.log', maxBytes=2000, backupCount=10))

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        self.consumer_id = -1
        self.producers = {}
        self.carts = {}
        self.publish_lock = Lock()
        self.register_lock = Lock()
        self.producer_lock = Lock()
        self.cart_lock = Lock()



    def register_producer(self):
        
        self.register_lock.acquire()
        self.logger.info('Start - register_producer')
        self.producer_id += 1
        self.producers[self.producer_id] = []
        self.logger.info('End - register_producer')
        self.register_lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        
        
        self.publish_lock.acquire()
        self.logger.info('Start - publish')
        success = False
        if len(self.producers.get(producer_id)) < self.queue_size_per_producer:
            self.producers.get(producer_id).append([product, 1])
            success = True
        self.logger.info('End - publish')
        self.publish_lock.release()
        return success

    def new_cart(self):
        
        self.cart_lock.acquire()
        self.logger.info('Start - new_cart')
        self.consumer_id += 1
        self.carts[self.consumer_id] = []
        self.logger.info('End - new_cart')
        self.cart_lock.release()
        return self.consumer_id

    def add_to_cart(self, cart_id, product):
        
        found_flag = False
        self.producer_lock.acquire()
        self.logger.info('Start - add_to_cart, params = %d, %s', cart_id, product)
        for producer in list(self.producers):
            for i in range(len(self.producers[producer])):
                if (self.producers[producer][i][0] == product and
                        self.producers[producer][i][1] == 1):
                    self.producers[producer][i][1] = 0
                    self.carts[cart_id].append([producer, product, i])
                    found_flag = True
                    break
            if found_flag:
                break


        self.logger.info('End - add_to_cart')
        self.producer_lock.release()
        return found_flag

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info('Start - remove_from_cart, params = %d, %s', cart_id, product)
        for entry in self.carts[cart_id]:
            if entry[1] == product:
                self.carts[cart_id].remove(entry)
                self.producers[entry[0]][entry[2]][1] = 1
                break
        self.logger.info('End - remove_from_cart')


    def place_order(self, cart_id):
        
        self.logger.info('Start - place_order, params = %d', cart_id)
        return [entry[1] for entry in self.carts[cart_id]]

def unittesting():
    
    market = Marketplace(10)
    producer_id = market.register_producer()
    consumer_id = market.new_cart()

    market.publish(producer_id, "paine")
    market.publish(producer_id, "apa")
    market.publish(producer_id, "sare")

    market.add_to_cart(consumer_id, "piper")

    market.add_to_cart(consumer_id, "apa")
    market.add_to_cart(consumer_id, "sare")
    market.add_to_cart(consumer_id, "apa")

    market.remove_from_cart(consumer_id, "sare")

    answer = market.place_order(consumer_id)

    print(answer)

if __name__ == "__main__":
    unittesting()


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        while True:
            id_ = self.marketplace.register_producer()
            for product in self.products:
                counter = 0
                while counter < product[1]:
                    published = self.marketplace.publish(id_, product[0])
                    if published:
                        counter += 1
                        sleep(product[2])
                    else:
                        sleep(self.republish_wait_time)
