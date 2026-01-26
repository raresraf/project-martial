


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for action in cart:
                action_type = action["type"]
                product = action["product"]
                quantity = action["quantity"]

                i = 0
                while i < quantity:
                    if action_type == "add":
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    else:
                        self.marketplace.remove_from_cart(cart_id, product)
                    i += 1

            self.marketplace.place_order(cart_id)



from threading import Lock, currentThread
import logging
from logging.handlers import RotatingFileHandler
import time



handler = RotatingFileHandler('marketplace.log', maxBytes=100000, backupCount=10)


logging.Formatter.converter = time.gmtime
logging.basicConfig(
        handlers=[handler],
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')

class ProducerInfo:
    def __init__(self, producer_id):
        self.producer_id = producer_id
        self.products_queue = []

    def is_product_in_queue(self, product):
        if product in self.products_queue:
            return True
        return False


class ProducerQueues:
    def __init__(self):
        self.queues = []

    def add_product(self, producer_id, product):
        for producer_info in self.queues:
            if producer_info.producer_id == producer_id:
                producer_info.products_queue.append(product)

    def has_producer_queue(self, producer_id):
        for producer_info in self.queues:
            if producer_info.producer_id == producer_id:
                return True
        return False

    def add_producer(self, producer_info):
        self.queues.append(producer_info)

    def find_producer(self, producer_id):
        for producer_info in self.queues:
            if producer_info.producer_id == producer_id:
                return producer_info
        return None


class CartInfo:
    def __init__(self, cart_id):
        self.cart_id = cart_id
        self.cart_products_queue = []

    def add_product(self, producer_id, product):
        self.cart_products_queue.append((producer_id, product))

    def remove_product(self, producer_id, product):
        self.cart_products_queue.remove((producer_id, product))


class CartQueues:
    def __init__(self):
        self.queues = []

    def exists_cart(self, cart_id):
        for cart_info in self.queues:
            if cart_info.cart_id == cart_id:
                return True
        return False

    def add_cart(self, cart_info):
        self.queues.append(cart_info)

    def find_cart(self, cart_id):
        for cart_info in self.queues:
            if cart_info.cart_id == cart_id:
                return cart_info
        return None

    def remove_cart(self, cart_id):
        for cart_info in self.queues:
            if cart_info.cart_id == cart_id:
                return self.queues.remove(cart_info)
        return None


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.producer_ids = []
        self.register_producer_id_counter = 0
        self.register_producer_id_lock = Lock()
        self.producer_queues = ProducerQueues()  

        self.register_new_cart_lock = Lock()
        self.register_new_cart_id_counter = 0
        self.cart_queues = CartQueues()
        self.add_product_to_cart_lock = Lock()

    def register_producer(self):
        
        logging.info("register_producer() was called")
        producer_id = 0
        with self.register_producer_id_lock:
            producer_id = self.register_producer_id_counter
            self.register_producer_id_counter += 1

        self.producer_ids.append(producer_id)
        logging.info("register_producer() has finished its execution")
        return producer_id

    def publish(self, producer_id, product):
        

        logging.info("publish() was called with arguments {} {}".format(producer_id, product))

        id_producer = int(producer_id)

        if not self.producer_queues.has_producer_queue(id_producer):
            producer_info = ProducerInfo(id_producer)
            self.producer_queues.add_producer(producer_info)

        queue_length = len(self.producer_queues.find_producer(id_producer).products_queue)
        if queue_length >= self.queue_size_per_producer:
            return False

        self.producer_queues.add_product(id_producer, product)

        logging.info("publish() has finished its execution")
        return True

    def new_cart(self):
        

        logging.info("new_cart() was called")

        cart_id = 0
        with self.register_new_cart_lock:
            cart_id = self.register_new_cart_id_counter
            self.register_new_cart_id_counter += 1

        logging.info("new_cart() has finished its execution")
        return cart_id

    def add_to_cart(self, cart_id, product):
        

        logging.info("add_to_cart() was called with arguments {} {}".format(cart_id, product))

        for producer_id in self.producer_ids:
            if self.producer_queues.has_producer_queue(producer_id):
                producer_info = self.producer_queues.find_producer(producer_id)

                self.add_product_to_cart_lock.acquire()

                if producer_info.is_product_in_queue(product):
                    producer_info.products_queue.remove(product)

                    if not self.cart_queues.exists_cart(cart_id):
                        cart_info = CartInfo(cart_id)
                        self.cart_queues.add_cart(cart_info)

                    cart_info = self.cart_queues.find_cart(cart_id)
                    cart_info.add_product(producer_id, product)
                    self.add_product_to_cart_lock.release()
                    return True



                self.add_product_to_cart_lock.release()

        logging.info("add_to_cart() has finished its execution")
        return False

    def remove_from_cart(self, cart_id, product):
        

        logging.info("remove_from_cart() was called with arguments {} {}".format(cart_id, product))

        cart_info = self.cart_queues.find_cart(cart_id)
        for cart_info_tuple in cart_info.cart_products_queue:
            tuple_producer_id = cart_info_tuple[0]
            tuple_product = cart_info_tuple[1]

            if tuple_product == product:
                cart_info.remove_product(tuple_producer_id, tuple_product)

                self.producer_queues.add_product(tuple_producer_id, tuple_product)
                break
        logging.info("remove_from_cart() has finished its execution")


    def place_order(self, cart_id):
        
        logging.info("place_order() was called with arguments {}".format(cart_id))
        products = []
        cart_info = self.cart_queues.find_cart(cart_id)
        self.cart_queues.remove_cart(cart_id)

        cart_products = cart_info.cart_products_queue
        for cart_tuple in cart_products:
            product = cart_tuple[1]
            print("{} bought {}".format(currentThread().getName(), product))
            products.append(product)

        logging.info("place_order() has finished its execution")
        return products


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id = self.marketplace.register_producer()

    def run(self):
        while 1:
            for product_tuple in self.products:
                product = product_tuple[0]
                quantity = product_tuple[1]
                waiting_time = product_tuple[2]
                i = 0
                while i < quantity:
                    while not self.marketplace.publish(str(self.id), product):
                        time.sleep(self.republish_wait_time)
                    time.sleep(waiting_time)
                    i += 1
