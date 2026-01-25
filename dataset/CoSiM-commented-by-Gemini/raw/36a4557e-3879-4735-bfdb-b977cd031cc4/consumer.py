


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.carts = carts


        self.consumer_id = marketplace.new_cart()
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']
        
        pass

    def run(self):
        for cart in range(len(self.carts)):  
            for operation in range(len(self.carts[cart])):  
                op_type = self.carts[cart][operation]['type']
                product = self.carts[cart][operation]['product']
                quantity = self.carts[cart][operation]['quantity']

                if op_type == "add":  
                    while quantity > 0:
                        while True:
                            verdict = self.marketplace.add_to_cart(self.consumer_id, product)

                            if verdict:
                                break

                            time.sleep(self.retry_wait_time)

                        quantity -= 1
                else:  
                    while quantity > 0:
                        self.marketplace.remove_from_cart(self.consumer_id, product)
                        quantity -= 1

        products = self.marketplace.place_order(self.consumer_id)
        for item in products:
            print(self.name + " bought " + str(item))>>>> file: marketplace.py


from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.carts = []  
        self.products = []  
        self.no_of_products = []  
        self.producer_id = -1
        self.consumer_id = -1
        self.producer_lock = Lock()
        self.consumer_lock = Lock()


        self.buffer_lock = Lock()
        

    def register_producer(self):
        
        self.producer_lock.acquire()

        self.producer_id += 1
        ids = self.producer_id
        self.no_of_products.append(0)
        

        self.producer_lock.release()

        return ids

    def publish(self, producer_id, product):
        
        verdict = True

        self.buffer_lock.acquire()

        
        if self.no_of_products[producer_id] >= self.queue_size_per_producer:
            verdict = False
        else:
            self.products.append((product, producer_id))
            self.no_of_products[producer_id] += 1

        self.buffer_lock.release()

        return verdict

    def new_cart(self):
        
        self.consumer_lock.acquire()

        self.consumer_id += 1
        ids = self.consumer_id
        self.carts.append([])

        self.consumer_lock.release()

        return ids

    def add_to_cart(self, cart_id, product):
        
        verdict = False

        self.buffer_lock.acquire()

        for prod, producer_id in self.products:  
            if prod == product:
                self.no_of_products[producer_id] -= 1  
                self.carts[cart_id].append((product, producer_id))
                self.products.remove((product, producer_id))  
                verdict = True
                break

        self.buffer_lock.release()

        return verdict

    def remove_from_cart(self, cart_id, product):
        
        for prod, producer_id in self.carts[cart_id]:  
            if prod == product:
                self.carts[cart_id].remove((product, producer_id))

                self.buffer_lock.acquire()

                self.products.append((product, producer_id))  
                self.no_of_products[producer_id] += 1
                

                self.buffer_lock.release()

                break

    def place_order(self, cart_id):
        
        products = []

        for prod, _ in self.carts[cart_id]:
            products.append(prod)

        return products


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.products = products
        self.producer_id = marketplace.register_producer()
        self.republish_wait_time = republish_wait_time
        

    def run(self):
        while True:
            for product in self.products:
                quantity = product[1]
                while quantity > 0:
                    while True:
                        verdict = self.marketplace.publish(self.producer_id, product[0])
                        if verdict:
                            time.sleep(product[2])
                            break
                        time.sleep(self.republish_wait_time)

                    quantity -= 1
