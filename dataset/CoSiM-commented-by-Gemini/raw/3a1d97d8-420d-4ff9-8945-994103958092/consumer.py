


import time



from threading import Thread

from tema.marketplace import Marketplace

class Consumer(Thread):
    

    def __init__(self,
                 carts: list,
                 marketplace: Marketplace,
                 retry_wait_time: int,
                 **kwargs) \
    :
        

        super().__init__(**kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for action in cart:
                type_ = action['type']
                product = action['product']
                qty = action['quantity']

                for _ in range(qty):
                    if type_ == 'add':
                        
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif type_ == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)

            order = self.marketplace.place_order(cart_id)

            for product in order:
                print(f'{self.name} bought {product}')


from threading import Lock

from tema.product import Product

class Marketplace:
    

    def __init__(self, queue_size_per_producer: int):
        

        self.queue_size_per_producer = queue_size_per_producer

        self.producer_queues = []
        self.consumer_carts = []

        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()

    def register_producer(self) -> int:
        

        with self.register_producer_lock:
            producer_id = len(self.producer_queues)

            self.producer_queues.append(([], Lock()))

        return producer_id

    def publish(self, producer_id: int, product: Product) -> bool:
        

        queue, lock = self.producer_queues[producer_id]

        with lock:
            if len(queue) >= self.queue_size_per_producer:
                return False

            queue.append(product)

        return True

    def new_cart(self) -> int:
        

        with self.new_cart_lock:
            cart_id = len(self.consumer_carts)

            self.consumer_carts.append([])

        return cart_id

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        

        cart = self.consumer_carts[cart_id]

        for producer_id, (queue, lock) in enumerate(self.producer_queues):
            with lock:
                try:
                    queue.remove(product) 
                except ValueError:
                    continue

            
            
            cart.append((product, producer_id))

            return True

        return False

    def remove_from_cart(self, cart_id: int, product: Product) -> bool:
        

        cart = self.consumer_carts[cart_id]

        for i, (prod, producer_id) in enumerate(cart):
            if prod == product:
                del cart[i] 

                queue, lock = self.producer_queues[producer_id]

                with lock:
                    queue.append(prod) 

                return True

        return False



    def place_order(self, cart_id) -> list:
        

        cart = self.consumer_carts[cart_id]

        return [product for product, producer_id in cart]


import time



from threading import Thread

from tema.marketplace import Marketplace

class Producer(Thread):
    

    def __init__(self,
                 products: list,
                 marketplace: Marketplace,
                 republish_wait_time: int,
                 **kwargs) \
    :
        

        super().__init__(**kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.id_ = self.marketplace.register_producer()

    def run(self):
        while True:
            for product, qty, wait_time in self.products:
                for _ in range(qty):
                    time.sleep(wait_time)

                    
                    while not self.marketplace.publish(self.id_, product):
                        time.sleep(self.republish_wait_time)
