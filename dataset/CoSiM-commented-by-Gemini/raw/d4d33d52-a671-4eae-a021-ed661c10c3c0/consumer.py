


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)

        self.name = kwargs["name"]
        self.no_product_wait_time = retry_wait_time
        self.shop = marketplace
        self.carts = carts

    def run(self):
        
        for cart in self.carts:
            cart_id = self.shop.new_cart()
            for order in cart:
                if order["type"] == "add":
                    self.add_to_cart(cart_id, order)
                elif order["type"] == "remove":
                    self.remove_from_cart(cart_id, order)
            bought = self.shop.place_order(cart_id)
            self.print_what_was_bought(bought)

    def add_to_cart(self, cart_id, order):
        
        i = order["quantity"]
        while i > 0:
            if not self.shop.add_to_cart(cart_id, order["product"]):
                time.sleep(self.no_product_wait_time)
                continue
            i -= 1

    def remove_from_cart(self, cart_id, order):
        
        for _ in range(order["quantity"]):
            self.shop.remove_from_cart(cart_id, order["product"])

    def print_what_was_bought(self, bought):
        
        for product in bought:
            print(self.name, "bought", product)

from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.reg_prod_lock = Lock()
        self.prod_list = []
        self.prod_max_queue = queue_size_per_producer

        
        self.shop_items_lock = Lock()
        self.shop_items = dict()

        
        self.carts = dict()
        self.cart_id_lock = Lock()
        self.cart_id = 0

    def register_producer(self):
        
        self.reg_prod_lock.acquire()
        self.prod_list.append(0)
        ret_id = len(self.prod_list) - 1
        self.reg_prod_lock.release()
        return ret_id



    def publish(self, producer_id, product) -> bool:
        
        if self.prod_list[producer_id] < self.prod_max_queue:
            self.prod_list[producer_id] += 1
            self.shop_items_lock.acquire()
            if product in self.shop_items.keys():
                self.shop_items_lock.release()
                self.shop_items[product][0].acquire()
                self.shop_items[product][1].append(producer_id)
                self.shop_items[product][0].release()
            else:
                self.shop_items[product] = (Lock(), [producer_id])
                self.shop_items_lock.release()
            return True
        return False

    def new_cart(self):
        
        self.cart_id_lock.acquire()
        cart_id_var = self.cart_id
        self.carts[cart_id_var] = []
        self.cart_id += 1
        self.cart_id_lock.release()
        return cart_id_var

    def add_to_cart(self, cart_id, product) -> bool:


        
        
        self.shop_items_lock.acquire()
        if product not in self.shop_items.keys():
            self.shop_items_lock.release()
            return False
        self.shop_items_lock.release()

        self.shop_items[product][0].acquire()
        if len(self.shop_items[product][1]) > 0:
            prod_id = self.shop_items[product][1][0]
            self.shop_items[product][1].pop(0)
            self.shop_items[product][0].release()
            if prod_id != -1:
                self.prod_list[prod_id] -= 1
            self.carts[cart_id].append(product)
            return True
        self.shop_items[product][0].release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        try:
            product_index = self.carts[cart_id].index(product)
            self.carts[cart_id].pop(product_index)
            self.shop_items[product][0].acquire()
            self.shop_items[product][1].append(-1)
            self.shop_items[product][0].release()
        except ValueError:
            return

    def place_order(self, cart_id):
        
        cart = self.carts[cart_id]
        return cart



from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, daemon=True)
        self.production_list = products
        self.shop = marketplace
        self.prod_id = marketplace.register_producer()
        self.shop_full_wait_time = republish_wait_time

    def run(self):
        
        while True:
            for order in self.production_list:
                product = order[0]
                quantity = order[1]
                production_time = order[2]
                while quantity > 0:
                    if self.shop.publish(self.prod_id, product):
                        quantity -= 1
                        time.sleep(production_time)
                    else:
                        time.sleep(self.shop_full_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int

    def __hash__(self) -> int:
        return hash((self.name))

    def __eq__(self, other) -> bool:
        return self.name == other.name


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    
    acidity: str
    roast_level: str
