


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)


        self.name = kwargs["name"]
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for item in cart:
                if item["type"] == "add":
                    for _ in range(item["quantity"]):
                        while not self.marketplace.add_to_cart(cart_id, item["product"]):
                            time.sleep(self.wait_time)

                elif item["type"] == "remove":
                    for _ in range(item["quantity"]):
                        while not self.marketplace.remove_from_cart(cart_id, item["product"]):
                            time.sleep(self.wait_time)

            final_cart = self.marketplace.place_order(cart_id)
            for item in final_cart:
                if item is not None:
                    print(self.name + " bought " + str(item))
    


from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        self.queue_size = queue_size_per_producer
        self.producer_count = 0
        self.cart_count = 0
        self.carts = {}
        self.cart_suppliers = {}
        self.producer_lock_univ = Lock()
        self.producer_lock = []
        self.cart_lock = Lock()
        self.producer_capacity = []
        self.item_lock = {}
        self.product_availability = {}
        self.product_suppliers = {}
        

    def register_producer(self):
        
        with self.producer_lock_univ:
            self.producer_capacity.append(self.queue_size)
            retval = self.producer_count
            self.producer_count += 1
            self.producer_lock += [Lock()]
            return retval

    def publish(self, producer_id, product):
        
        with self.producer_lock[producer_id]:
            if self.producer_capacity[producer_id] > 0:
                amount = self.product_availability.setdefault(product[0], 0)
                producers = self.product_suppliers.setdefault(product[0], [])
                self.product_suppliers.update({product[0]: producers + [producer_id]})

                self.product_availability.update({product[0]: 1 + amount})
                self.producer_capacity[producer_id] -= 1
                return True

        return False

    def new_cart(self):
        
        with self.cart_lock:
            retval = self.cart_count
            self.carts.setdefault(self.cart_count, [])
            self.cart_suppliers.setdefault(self.cart_count, [])
            self.cart_count += 1
            return retval

    def add_to_cart(self, cart_id, product):
        
        lock = self.item_lock.setdefault(product, Lock())

        with lock:
            amount = self.product_availability.setdefault(product, 0)

            if amount == 0:
                return False

            producers = self.product_suppliers.get(product)
            if producers is not None:
                self.producer_capacity[producers[0]] += 1
                self.cart_suppliers[cart_id].append(producers[0])
                producers.pop(0)


            self.product_availability.update({product: amount - 1})
            self.carts[cart_id].append(product)
            return True

    def remove_from_cart(self, cart_id, product):
        
        lock = self.item_lock.setdefault(product, Lock())

        with lock:
            amount = self.product_availability.setdefault(product, 0)
            producers = self.product_suppliers.setdefault(product, [])

            product_idx = self.carts[cart_id].index(product)
            producer_id = self.cart_suppliers[cart_id][product_idx]
            with self.producer_lock[producer_id]:
                self.product_suppliers.update({product: producers + [producer_id]})
                self.producer_capacity[producer_id] -= 1
                self.product_availability.update({product: amount + 1})
                self.cart_suppliers[cart_id][product_idx] = None
                self.carts[cart_id][product_idx] = None
                return True

        return False



    def place_order(self, cart_id):
        
        return self.carts[cart_id]


import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time
        

    def run(self):
        this_id = self.marketplace.register_producer()

        while True:
            for item in self.products:
                for _ in range(item[1]):
                    while not self.marketplace.publish(this_id, item):
                        time.sleep(self.wait_time)

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
