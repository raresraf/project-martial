


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_command(self, cart_id, product, quantity):


        for _ in range(quantity):
            status = False
            while not status:
                status = self.marketplace.add_to_cart(cart_id, product)
                if not status:
                    sleep(self.retry_wait_time)

    def remove_command(self, cart_id, product, quantity):


        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for op in cart:
                command = op.get("type")
                if command == "add":
                    self.add_command(cart_id, op.get("product"), op.get("quantity"))
                if command == "remove":
                    self.remove_command(cart_id, op.get("product"), op.get("quantity"))
            item_list = self.marketplace.place_order(cart_id)
            for prod in item_list:
                print("%s bought %s" % (self.name, prod))

from threading import Lock


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        self.cart_id = -1
        self.producer_list = []
        self.cart_list = []
        self.lock = Lock()

    def register_producer(self):
        
        self.lock.acquire()
        self.producer_id += 1
        self.producer_list.append([])
        

        self.lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        
        self.lock.acquire()
        count = 0
        for prod in self.producer_list[producer_id]:
            if prod[1]:
                count += 1

        if (
            self.producer_list[producer_id] != 0
            and self.queue_size_per_producer > count
        ):
            self.producer_list[producer_id].append([product, True])
            self.lock.release()
            return True
        self.lock.release()
        return False

    def new_cart(self):
        
        self.lock.acquire()
        self.cart_id += 1
        self.cart_list.append([])
        self.lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        self.lock.acquire()
        for lists in self.producer_list:
            for item in lists:
                if item[0] == product and item[1]:
                    self.cart_list[cart_id].append(product)
                    item[1] = False
                    self.lock.release()
                    return True
        self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.lock.acquire()
        self.cart_list[cart_id].remove(product)

        for lists in self.producer_list:
            for item in lists:
                if item[0] == product and not item[1]:


                    item[1] = True
        self.lock.release()

    def place_order(self, cart_id):
        
        return self.cart_list[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def publish_product(self, product_id, quantity, cooldown, producer_id):
        for _ in range(quantity):
            status = False
            while not status:
                status = self.marketplace.publish(producer_id, product_id)
                if not status:
                    sleep(self.republish_wait_time)
            sleep(cooldown)

    def run(self):
        producer_id = self.marketplace.register_producer()
        while True:
            for prod in self.products:
                self.publish_product(prod[0], prod[1], prod[2], producer_id)
