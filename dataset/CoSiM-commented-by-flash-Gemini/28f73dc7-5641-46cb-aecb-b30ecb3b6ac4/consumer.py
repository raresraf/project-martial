/**
 * @file consumer.py
 * @brief Semantic documentation for consumer.py. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */



from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        


        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.mk_p = marketplace
        self.ops = {"add": self.mk_p.add_to_cart,
                    "remove": self.mk_p.remove_from_cart}
        self.wait = retry_wait_time


    def run(self):
        for cart in self.carts:


            _id = self.mk_p.new_cart()

            for op_in_cart in cart:
                no_of_op = 0
                while no_of_op < op_in_cart["quantity"]:
                    result = self.ops[op_in_cart["type"]](_id, op_in_cart["product"])

                    if result is None:
                        no_of_op += 1
                    elif result is True:
                        no_of_op += 1
                    else:
                        time.sleep(self.wait)

            self.mk_p.place_order(_id)

from threading import Lock, currentThread

class Marketplace:
    
    lock_reg = Lock()
    lock_carts = Lock()
    lock_alter = Lock()
    print = Lock()
    no_carts = None 
    def __init__(self, queue_size_per_producer):
        
        self.no_carts = 0
        self.max_prod_q_size = queue_size_per_producer
        self.prods = []  
        self.carts = {}  
        self.producers = {}  
        self.prod_q_sizes = []  

    def register_producer(self):
        
        with self.lock_reg:
            _id = len(self.prod_q_sizes)
            self.prod_q_sizes.append(0)

        return _id

    def publish(self, producer_id, product):
        
        _id = int(producer_id)

        if self.prod_q_sizes[_id] >= self.max_prod_q_size:
            return False

        self.prod_q_sizes[_id] += 1
        self.prods.append(product)
        self.producers[product] = _id

        return True

    def new_cart(self):
        
        with self.lock_carts:
            self.no_carts += 1
            cart_id = self.no_carts

        self.carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        
        with self.lock_alter:


            if product not in self.prods:
                return False

            self.prod_q_sizes[self.producers[product]] -= 1
            self.prods.remove(product)

        self.carts[cart_id].append(product)

        return True

    def remove_from_cart(self, cart_id, product):
        
        self.carts[cart_id].remove(product)
        self.prods.append(product)

        with self.lock_alter:
            self.prod_q_sizes[self.producers[product]] += 1


    def place_order(self, cart_id):
        
        prod_list = self.carts.pop(cart_id, None)

        for product in prod_list:
            with self.print:
                print(currentThread().getName(), "bought", product)

        return prod_list


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.mk_p = marketplace
        self.wait = republish_wait_time

        self._id = self.mk_p.register_producer()

    def run(self):
        while True:
            for (prod, quant, wait) in self.products:
                i = 0
                while i < quant:
                    ret = self.mk_p.publish(str(self._id), prod)

                    if ret:
                        time.sleep(wait)
                        i += 1
                    else:
                        time.sleep(self.wait)
