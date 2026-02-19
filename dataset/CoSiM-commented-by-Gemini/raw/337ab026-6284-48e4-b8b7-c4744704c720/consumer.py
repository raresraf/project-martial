


"""



@file consumer.py



@brief A multi-threaded simulation of a marketplace with producers and consumers.







This file contains all components for the simulation. It uses a coarse-grained



locking strategy in the central Marketplace, but the implementation contains



several race conditions, making it unsafe for true concurrent execution.



"""











from threading import Thread, BoundedSemaphore, currentThread



import time



from dataclasses import dataclass











@dataclass(init=True, repr=True, order=False, frozen=True)



class Product:



    """An immutable dataclass for a generic product."""



    name: str



    price: int











@dataclass(init=True, repr=True, order=False, frozen=True)



class Tea(Product):



    """An immutable dataclass for Tea."""



    type: str











@dataclass(init=True, repr=True, order=False, frozen=True)



class Coffee(Product):



    """An immutable dataclass for Coffee."""



    acidity: str



    roast_level: str











class Marketplace:



    """



    The central shared resource for the simulation, managing products and carts.







    This class attempts to manage concurrency using a single BoundedSemaphore as a



    global lock. However, the lock is not applied consistently, leading to



    significant race conditions.



    """







    def __init__(self, queue_size_per_producer):



        """Initializes the marketplace state."""



        self.queue_size_per_producer = queue_size_per_producer



        self.producer_ids = -1



        self.cart_ids = -1



        self.products = []



        self.producers_capacity = {}



        self.producers = {}



        self.carts = {}



        # A single semaphore acting as a coarse-grained lock for the entire marketplace.



        self.semaphore = BoundedSemaphore(1)







    def register_producer(self):



        """Atomically registers a new producer and returns a unique ID."""



        self.semaphore.acquire()



        self.producer_ids = self.producer_ids + 1



        self.semaphore.release()



        self.producers_capacity[self.producer_ids] = 0



        return self.producer_ids







    def publish(self, producer_id, product):



        """



        Allows a producer to publish a product.







        @warning This method has a critical race condition. The check for producer



        capacity and the modifications to `self.products` and `self.producers`



        are not protected by the lock, allowing for data corruption if multiple



        producers call this concurrently.



        """



        if self.producers_capacity[int(producer_id)] < self.queue_size_per_producer:



            self.producers_capacity[int(producer_id)] += 1



            self.products.append(product)



            self.producers[product] = int(producer_id)



            return True



        return False







    def new_cart(self):



        """Atomically creates a new cart for a consumer and returns its ID."""



        self.semaphore.acquire()



        self.cart_ids = self.cart_ids + 1



        self.semaphore.release()



        self.carts[self.cart_ids] = []



        return self.cart_ids







    def add_to_cart(self, cart_id, product):



        """



        Adds a product to a consumer's cart.







        @warning This method has a Time-of-check to time-of-use (TOCTOU) race



        condition. The check `if product in self.products` is not protected by



        the lock, meaning another thread could remove the product between the



        check and the subsequent operations, leading to an error.



        """



        if product in self.products:



            self.semaphore.acquire()



            self.producers_capacity[self.producers[product]] -= 1



            self.semaphore.release()



            self.carts[cart_id].append(product)



            return True



        else:



            return False







    def remove_from_cart(self, cart_id, product):



        """



        Removes a product from a cart.







        @warning This method has a race condition. The modification of



        `self.products` is not protected by the lock.



        """



        self.carts[cart_id].remove(product)



        self.products.append(product)



        self.semaphore.acquire()



        self.producers_capacity[self.producers[product]] += 1



        self.semaphore.release()







    def place_order(self, cart_id):



        """Finalizes an order and prints the items bought."""



        for product in self.carts[cart_id]:



            self.semaphore.acquire()



            print(F"{currentThread().getName()} bought {product}")



            self.semaphore.release()



        return self.carts[cart_id]











class Producer(Thread):



    """A worker thread that simulates a producer publishing products."""







    def __init__(self, products, marketplace, republish_wait_time, **kwargs):



        """Initializes the producer."""



        Thread.__init__(self, **kwargs)



        self.products = products



        self.marketplace = marketplace



        self.republish_wait_time = republish_wait_time



        self.kwargs = kwargs







    def run(self):



        """



        Main loop for the producer. It registers and then continuously attempts



        to publish products, retrying with a delay on failure.



        """



        producer_id = str(self.marketplace.register_producer())



        while True:



            for product in self.products:



                count = product[1]



                while count > 0:



                    if self.marketplace.publish(producer_id, product[0]):



                        time.sleep(product[2])



                        count -= 1



                    else:



                        # If publishing fails, wait and retry.



                        time.sleep(self.republish_wait_time)











class Consumer(Thread):



    """A worker thread that simulates a consumer processing a shopping list."""







    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):



        """Initializes the consumer."""



        Thread.__init__(self, **kwargs)



        self.carts = carts



        self.marketplace = marketplace



        self.retry_wait_time = retry_wait_time



        self.kwargs = kwargs







    def run(self):



        """



        Main loop for the consumer. Processes a list of cart operations,



        retrying with a delay if a product is not available.



        """



        for cart in self.carts:



            cart_id = self.marketplace.new_cart()



            for operation in cart:



                count = 0



                while count < operation["quantity"]:



                    if operation["type"] == "add":



                        result = self.marketplace.add_to_cart(cart_id, operation["product"])



                        if result:



                            count += 1



                        else:



                            # If adding fails, wait and retry.



                            time.sleep(self.retry_wait_time)



                    elif operation["type"] == "remove":



                        self.marketplace.remove_from_cart(cart_id, operation["product"])



                        count += 1



            self.marketplace.place_order(cart_id)


