
"""
A multi-threaded producer-consumer simulation for a marketplace.

This module defines the classes and data structures to simulate a marketplace
where producers create products and consumers purchase them. The simulation
utilizes threading to have concurrent producers and consumers interacting
with a central marketplace.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer that purchases products from the marketplace.

    Each consumer runs in its own thread, processing a list of shopping carts.
    For each cart, it adds and removes products as specified in its order list.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of orders, where each order is a list of product requests.
            marketplace (Marketplace): The central marketplace from which to purchase products.
            retry_wait_time (float): The time in seconds to wait before retrying to add a product.
            **kwargs: Additional keyword arguments, including the thread's 'name'.
        """
        super().__init__(name=kwargs["name"])
        self.carts: list = carts
        self.marketplace = marketplace
        self.retry_time = retry_wait_time
        self.output_str = "%s bought %s"

    def run(self):
        """
        The main execution method for the consumer thread.

        Processes each cart in the `self.carts` list. For each product in an order,
        it interacts with the marketplace to add or remove items from its cart.
        Once an order is complete, it is placed, and the purchased items are printed.
        """
        while len(self.carts) != 0:
            
            order = self.carts.pop(0)
            
            cart_id = self.marketplace.new_cart()

            while len(order) != 0:
                
                request = order.pop(0)

                
                if request["type"] == "add":
                    added_products = 0                           
                    while added_products < request["quantity"]:  
                        
                        if self.marketplace.add_to_cart(cart_id, request["product"]):
                            added_products += 1
                        else:
                            sleep(self.retry_time)               

                
                if request["type"] == "remove":
                    for _ in range(0, request["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, request["product"])

            
            cart_items = self.marketplace.place_order(cart_id)
            for product in cart_items:
                print(self.output_str % (self.name, product))    


from threading import Lock
from queue import Queue, Full, Empty
from typing import Dict


class Marketplace:
    """
    A central marketplace for producers to publish products and consumers to buy them.

    This class manages the inventory from multiple producers and the shopping carts
    of multiple consumers. It uses locks to ensure thread-safe operations on shared data.
    """

    def __init__(self, queue_size_per_producer: int):
        
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum size of the product queue for each producer.
        """
        
        self.register_lock = Lock()                     
        self.producers_no = 0                           
        self.queue_size = queue_size_per_producer       
        self.producer_queues: Dict[int, Queue] = {}     

        
        self.cart_lock = Lock()                         
        self.consumers_no = 0                           
        self.consumer_carts: Dict[int, list] = {}       

        
        self.register_producer(ignore_limit=True)

    def register_producer(self, ignore_limit: bool = False) -> int:
        """
        Registers a new producer with the marketplace.

        Args:
            ignore_limit (bool): If True, the producer's queue has an infinite size.

        Returns:
            int: The ID assigned to the new producer.
        """
        
        self.register_lock.acquire()                            
        producer_id = self.producers_no                         
        if ignore_limit:
            
            self.producer_queues[producer_id] = Queue()
        else:
            
            self.producer_queues[producer_id] = Queue(self.queue_size)
        self.producers_no += 1                                  
        self.register_lock.release()                            
        return producer_id

    def publish(self, producer_id: int, product) -> bool:
        """
        Publishes a product to a producer's queue.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the queue was full.
        """
        
        try:
            self.producer_queues[producer_id].put_nowait(product)
        except Full:
            return False


        return True

    def new_cart(self) -> int:
        """
        Creates a new shopping cart for a consumer.

        Returns:
            int: The ID of the new cart.
        """
        
        self.cart_lock.acquire()                    
        cart_id = self.consumers_no
        self.consumer_carts[cart_id] = []           
        self.consumers_no += 1                      
        self.cart_lock.release()                    
        return cart_id



    def add_to_cart(self, cart_id: int, product) -> bool:
        """
        Adds a product to a consumer's shopping cart.

        It searches for the product in all producer queues. If found, it's moved
        to the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            bool: True if the product was added, False otherwise.
        """
        
        
        cart = self.consumer_carts[cart_id]
        for producer_id in range(0, self.producers_no):
            try:
                
                queue_head = self.producer_queues[producer_id].get_nowait()

                if queue_head == product:
                    
                    cart.append(queue_head)
                    return True

                
                while True:
                    
                    try:
                        self.producer_queues[producer_id].put_nowait(queue_head)
                        break
                    except Full:
                        
                        continue

            except Empty:
                
                continue

        return False

    def remove_from_cart(self, cart_id: int, product) -> None:
        """
        Removes a product from a consumer's shopping cart.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product: The product to remove.
        """
        
        try:
            
            self.consumer_carts[cart_id].remove(product)
            
            self.publish(0, product)
        except ValueError:
            
            pass



    def place_order(self, cart_id: int) -> list:
        """
        Places an order for the items in a shopping cart.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: A list of products in the cart.
        """
        
        return self.consumer_carts[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer that creates and publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products to produce.
            marketplace (Marketplace): The central marketplace to publish products to.
            republish_wait_time (float): The time in seconds to wait before retrying to publish.
            **kwargs: Additional keyword arguments.
        """
        
        super().__init__(name=kwargs["name"], daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        The main execution method for the producer thread.

        Continuously produces and publishes products to the marketplace.
        """
        while True:
            for product in self.products:                   
                produced = 0                                
                waited = False                              

                while produced < product[1]:                
                    if not waited:
                        sleep(product[2])                   

                    
                    if self.marketplace.publish(self.producer_id, product[0]):
                        produced += 1
                        waited = False
                    else:
                        sleep(self.republish_time)          
                        waited = True


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True, eq=True)
class Product:
    """
    A simple dataclass representing a generic product.
    """
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    A dataclass representing Tea, inheriting from Product.
    """
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    A dataclass representing Coffee, inheriting from Product.
    """
    
    acidity: str
    roast_level: str

