


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

            for item in cart:
                quantity = 0

                while quantity < item["quantity"]:

                    if item["type"] == "add":
                        ver = self.marketplace.add_to_cart(cart_id, item["product"])
                    else:
                        ver = self.marketplace.remove_from_cart(cart_id, item["product"])

                    if ver:
                        quantity += 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread

class Marketplace:
    
    def __init__(self, queue_size_per_producer):


        

        self.max_products_per_size = queue_size_per_producer
        self.carts = {}
        self.producers = {}
        self.reserved = {}

        self.id_cart = 0
        self.id_producer = 0

        self.lock_id_cart = Lock()
        self.lock_id_producer = Lock()
        self.lock_print = Lock()

    def register_producer(self):
        
        with self.lock_id_producer:
            self.id_producer += 1
            prod_id = self.id_producer

        self.producers[prod_id] = []
        return prod_id


    def publish(self, producer_id, product):
        

        prod_id = int(producer_id)

        if len(self.producers[prod_id]) >= self.max_products_per_size:
            return False

        self.producers[prod_id].append(product)

        return True

    def new_cart(self):
        
        with self.lock_id_cart:
            self.id_cart += 1
            cart_id = self.id_cart

        self.carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        
        ver = False

        for _ in self.producers:
            if product in self.producers[_]:
                ver = True
                key = _
                break

        if not ver:
            return False

        self.producers[key].remove(product)
        if key in self.reserved.keys():
            self.reserved[key].append(product)
        else:
            self.reserved[key] = []
            self.reserved[key].append(product)

        self.carts[cart_id].append(product)

        return True

    def remove_from_cart(self, cart_id, product):
        

        ver = True
        for key in self.reserved:
            for cnt in self.reserved[key]:
                if cnt == product:
                    ver = False
                    rem = key
                    break
            if not ver:
                break

        self.carts[cart_id].remove(product)


        self.producers[rem].append(product)
        self.reserved[rem].remove(product)
        return True

    def place_order(self, cart_id):
        

        res = []
        res.extend(self.carts[cart_id])
        del self.carts[cart_id]

        for cnt in res:
            with self.lock_print:
                print("{} bought {}".format(currentThread().getName(), cnt))

        return res


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        



        self.prod_id = self.marketplace.register_producer()

    def run(self):

        

        while True:
            for(product, num_prod, wait_time) in self.products:

                for quantity in range(num_prod):

                    if self.marketplace.publish(str(self.prod_id), product):
                        time.sleep(wait_time)
                    else:
                        time.sleep(self.republish_wait_time)
                        quantity -= 1


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
