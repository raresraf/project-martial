


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        remove_from_cart = "remove"
        add_to_cart = "add"
        
        self.cart_actions = {remove_from_cart: self.marketplace.remove_from_cart,
                             add_to_cart: self.marketplace.add_to_cart}

    def run(self):
        
        for cart in self.carts:
            id_of_cart = self.marketplace.new_cart()
            for action in cart:
                index = 0
                action_quantity = action["quantity"]
                while index < action_quantity:
                    action_type = action["type"]
                    action_product = action["product"]
                    result = self.cart_actions[action_type](id_of_cart, action_product)

                    if result is False:
                        time.sleep(self.retry_wait_time)
                    elif result is True or result is None:
                        index += 1

            self.marketplace.place_order(id_of_cart)



from threading import Lock, currentThread


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.products = [] 
        self.carts = {} 
        self.map_products_to_producer = {} 
        self.register_lock = Lock() 
        self.new_cart_lock = Lock() 
        self.products_lock = Lock() 
        self.final_lock = Lock() 
        self.cart_id = 0 

    def register_producer(self):
        
        with self.register_lock:
            producer_id = len(self.products)
            self.products.append([])
        return producer_id

    def publish(self, producer_id, product):
        
        with self.products_lock:
            if len(self.products[(producer_id)]) >= self.queue_size_per_producer:
                return False
            self.products[producer_id].append(product)
        self.map_products_to_producer[product] = producer_id
        return True

    def new_cart(self):
        
        with self.new_cart_lock:
            self.cart_id += 1
        self.carts[self.cart_id] = []
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        with self.products_lock:
            if product not in [j for i in self.products for j in i]:
                return False
            if product in self.map_products_to_producer.keys():


                if product in self.products[self.map_products_to_producer[product]]:
                    self.products[self.map_products_to_producer[product]].remove(product)
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        



        with self.products_lock:
            self.carts[cart_id].remove(product)
            self.products[self.map_products_to_producer[product]].append(product)

    def place_order(self, cart_id):
        

        for product_type in range(len(self.carts[cart_id])):
            with self.final_lock:
                print(currentThread().getName(), "bought", self.carts[cart_id][product_type])

        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        
        while True:
            for product in self.products:
                index = 0
                product_type = product[0]
                count_prod = product[1]
                wait_time = product[2]
                while index < count_prod:
                    result = self.marketplace.publish(self.producer_id, product_type)

                    if result is False:
                        time.sleep(self.republish_wait_time)
                    else:
                        time.sleep(wait_time)
                        index += 1
