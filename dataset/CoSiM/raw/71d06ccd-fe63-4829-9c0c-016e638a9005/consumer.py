


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)

        self.marketplace = marketplace
        self.carts = {self.marketplace.new_cart() : cart for cart in carts}
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        for cart_id, cart in self.carts.items():
            for operation in cart:
                if operation['type'] == 'add':
                    for _ in range(operation['quantity']):
                        add_status = \
                                self.marketplace.add_to_cart(
                                    cart_id,
                                    operation['product']
                                )

                        while not add_status:
                            sleep(self.retry_wait_time)
                            add_status = \
                                    self.marketplace.add_to_cart(
                                        cart_id,
                                        operation['product']
                                    )

                if operation['type'] == 'remove':
                    for _ in range(operation['quantity']):
                        remove_status = \
                            self.marketplace.remove_from_cart(
                                cart_id,
                                operation['product']
                            )

                        while not remove_status:
                            sleep(self.retry_wait_time)
                            remove_status = \
                                self.marketplace.remove_from_cart(
                                    cart_id,
                                    operation['product']
                                )

        for cart_id, _ in self.carts.items():
            order = self.marketplace.place_order(cart_id)
            for product in order:
                print(self.name + " bought " + str(product[0]))


from threading import Lock
from tema.products_container import ProductsContainer

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()

        self.avail_producer_id = 0
        self.avail_cart_id = 0

        self.products = {}
        self.carts = {}

    def register_producer(self):
        


        with self.register_producer_lock:
            producer_id = self.avail_producer_id
            self.avail_producer_id += 1

            self.products[producer_id] = ProductsContainer(self.queue_size_per_producer)

            return producer_id

    def publish(self, producer_id, product):
        

        return self.products[producer_id].put((product, producer_id))

    def new_cart(self):
        
        with self.new_cart_lock:
            cart_id = self.avail_cart_id
            self.avail_cart_id += 1

            self.carts[cart_id] = ProductsContainer()

            return cart_id

    def add_to_cart(self, cart_id, product):
        
        for producer_id, products in self.products.items():
            product_data = (product, producer_id)
            if products.has(product_data):
                self.products[producer_id].remove(product_data)
                self.carts[cart_id].put(product_data)
                return True

        return False

    def remove_from_cart(self, cart_id, product):
        

        products_to_remove = [it for it in self.carts[cart_id].get_all() if it[0] == product]
        if len(products_to_remove) == 0:
            return False

        removed_product = products_to_remove[0]

        if removed_product[1] not in self.products:
            return False

        self.carts[cart_id].remove(removed_product)
        self.products[removed_product[1]].put(removed_product)

        return True

    def place_order(self, cart_id):
        
        return self.carts[cart_id].get_all()


from threading import Thread
from random import choice
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, daemon=True)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        while True:
            published_product = choice(self.products)

            for _ in range(published_product[1]):
                publish_status = self.marketplace.publish(self.producer_id, published_product[0])

                while not publish_status:
                    sleep(self.republish_wait_time)
                    publish_status = \
                            self.marketplace.publish(self.producer_id, published_product[0])

                sleep(published_product[2])


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


from threading import Lock

class ProductsContainer:
    
    def __init__(self, max_size=-1):
        
        self.products = []
        self.lock = Lock()
        self.max_size = max_size

    def put(self, product_data):
        
        with self.lock:
            if self.max_size == -1:
                self.products.append(product_data)
                return True

            if len(self.products) >= self.max_size:
                return False

            self.products.append(product_data)
            return True

    def remove(self, product_data):
        
        with self.lock:
            try:
                self.products.remove(product_data)
                return True
            except ValueError:
                return False

    def get_all(self):
        
        with self.lock:
            return self.products

    def has(self, product_data):
        
        with self.lock:
            return product_data in self.products
