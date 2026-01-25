


from threading import Thread, Lock
import time


class Consumer(Thread):
    
    cart_id = -1
    name = ''
    my_lock = Lock()

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self)
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.name = kwargs['name']
        

    def run(self):
        self.my_lock.acquire()
        for i in range(len(self.carts)):
            self.cart_id = self.marketplace.new_cart()


            for j in range(len(self.carts[i])):
                if self.carts[i][j]['type'] == 'add':
                    for k in range(self.carts[i][j]['quantity']):
                        verify = False
                        while not verify:
                            verify = self.marketplace.add_to_cart(self.cart_id,
                                                                  self.carts[i][j]['product']
                                                                  )
                            if not verify:
                                time.sleep(self.retry_wait_time)



                elif self.carts[i][j]['type'] == 'remove':
                    for k in range(self.carts[i][j]['quantity']):
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
            list_1 = self.marketplace.place_order(self.cart_id)
            for k in range(len(list_1) - 1, -1, -1):
                print(self.name + ' bought ' + str(list_1[k][0]))
                self.marketplace.remove_from_cart(self.cart_id, list_1[k][0])
        self.my_lock.release()


from threading import Lock


class Marketplace:
    
    id_producer = 0
    id_cart = 0
    queues = []
    carts = []
    my_Lock1 = Lock()
    my_Lock2 = Lock()
    done = 0

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        
        self.queues.append([])
        self.id_producer = self.id_producer + 1
        return self.id_producer - 1

    def publish(self, producer_id, product):
        
        if len(self.queues[producer_id]) >= self.queue_size_per_producer:
            return False
        self.queues[producer_id].append([product, "Disponibil"])
        return True

    def new_cart(self):
        
        self.carts.append([])
        self.id_cart = self.id_cart + 1
        return self.id_cart - 1

    def add_to_cart(self, cart_id, product):
        
        verify = 0
        for i in range(len(self.queues)):
            for j in range(len(self.queues[i])):
                self.my_Lock1.acquire()
                if product == self.queues[i][j][0] \
                        and self.queues[i][j][1] == 'Disponibil' \
                        and verify == 0:
                    self.carts[cart_id].append([product, i])
                    self.queues[i][j][1] = 'Indisponibil'
                    verify = 1
                    self.my_Lock1.release()
                    break
                self.my_Lock1.release()
                if verify == 1:
                    break
        if verify == 1:
            return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        for i in range(len(self.carts[cart_id])):
            if product == self.carts[cart_id][i][0]:
                for j in range(len(self.queues[self.carts[cart_id][i][1]])):
                    self.my_Lock2.acquire()
                    if self.queues[self.carts[cart_id][i][1]][j][0] == product \
                            and self.queues[self.carts[cart_id][i][1]][j][1] == 'Indisponibil':
                        self.queues[self.carts[cart_id][i][1]][j][1] = 'Disponibil'
                        self.carts[cart_id].remove(self.carts[cart_id][i])
                        self.my_Lock2.release()
                        return True
                    self.my_Lock2.release()
        return False

    def place_order(self, cart_id):
        

        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    
    producer_id = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, daemon=kwargs['daemon'])
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        

    def run(self):
        self.producer_id = self.marketplace.register_producer()


        while True:
            for i in range(len(self.products)):
                for j in range(self.products[i][1]):
                    verify = self.marketplace.publish(self.producer_id, self.products[i][0])
                    time.sleep(self.products[i][2])
                    if not verify:
                        time.sleep(self.republish_wait_time)
                        break


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
