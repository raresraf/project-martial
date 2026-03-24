"""
This module implements a multithreaded marketplace simulation featuring producers, consumers,
and product definitions. Producers generate products and publish them to the marketplace,
while consumers create carts, add/remove products, and place orders.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    A consumer thread that interacts with the marketplace to buy products.
    Each consumer has a list of carts (orders) to process.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer thread.

        Args:
            carts (list): A list of orders (each order is a list of product requests).
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): The time to wait before retrying an unsuccessful purchase.
            **kwargs: Arbitrary keyword arguments, including 'name' for the thread.
        """

        
        super().__init__(name=kwargs["name"])
        self.carts: list = carts
        self.marketplace = marketplace
        self.retry_time = retry_wait_time
    def run(self):
        """
        The main execution loop for the consumer thread.
        It processes each cart in the consumer's list, adding or removing products
        from the marketplace, and finally placing the order.
        """


        # Block Logic: Continues processing orders as long as there are carts in the list.
        while len(self.carts) != 0:
            # Retrieves the next order to process from the front of the carts list.
            order = self.carts.pop(0)
            
            # Creates a new shopping cart in the marketplace for this order.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Processes each product request within the current order.
            while len(order) != 0:
                # Retrieves the next product request from the order.
                request = order.pop(0)

                # Block Logic: Handles "add" type requests.
                if request["type"] == "add":
                    added_products = 0                           
                    # Attempts to add the requested quantity of products to the cart.
                    while added_products < request["quantity"]:  
                        # If adding to cart is successful, increment count.
                        if self.marketplace.add_to_cart(cart_id, request["product"]):
                            added_products += 1
                        # If unsuccessful, wait and retry.
                        else:
                            sleep(self.retry_time)               

                # Block Logic: Handles "remove" type requests.
                if request["type"] == "remove":
                    # Removes the specified quantity of product from the cart.
                    for _ in range(0, request["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, request["product"])

            # Block Logic: Places the final order in the marketplace and prints purchased items.
            cart_items = self.marketplace.place_order(cart_id)
            for product in cart_items:
                print(self.output_str % (self.name, product))    


from threading import Lock
from queue import Queue, Full, Empty
from typing import Dict


class Marketplace:
    """
    Simulates a marketplace where producers can publish products and consumers can
    create carts, add/remove products, and place orders. It manages product queues
    for producers and shopping carts for consumers, ensuring thread-safe operations.
    """
    

    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the Marketplace with a specified queue size for each producer.

        Args:
            queue_size_per_producer (int): The maximum number of products a producer can
                                           have in its queue at any given time.
        """

        

        # Block Logic: Producer-related attributes and their associated lock.
        # 'register_lock': Protects access to producer registration.
        # 'producers_no': Counter for the number of registered producers.
        # 'queue_size': Max size for producer queues.
        # 'producer_queues': Stores product queues for each producer, keyed by producer ID.
        self.register_lock = Lock()                     
        self.producers_no = 0                           
        self.queue_size = queue_size_per_producer       
        self.producer_queues: Dict[int, Queue] = {}     

        # Block Logic: Consumer/Cart-related attributes and their associated lock.
        # 'cart_lock': Protects access to consumer cart operations.
        # 'consumers_no': Counter for the number of created consumer carts.
        # 'consumer_carts': Stores shopping carts for each consumer, keyed by cart ID.
        self.cart_lock = Lock()                         
        self.consumers_no = 0                           
        self.consumer_carts: Dict[int, list] = {}       

        # Block Logic: Registers an initial producer (producer_id 0) to handle returned items.
        self.register_producer(ignore_limit=True)

    def register_producer(self, ignore_limit: bool = False) -> int:
        """
        Registers a new producer with the marketplace, assigning it a unique ID
        and creating a product queue.

        Args:
            ignore_limit (bool): If True, the producer's queue will have no size limit.
                                 Otherwise, it uses the marketplace's default queue size.

        Returns:
            int: The unique ID assigned to the registered producer.
        """

        
        # Block Logic: Acquires a lock to ensure thread-safe producer registration.
        self.register_lock.acquire()                            
        # Assigns a unique producer ID.
        producer_id = self.producers_no                         
        # Creates a new queue for the producer, with or without a size limit.
        if ignore_limit:
            self.producer_queues[producer_id] = Queue()
        else:
            self.producer_queues[producer_id] = Queue(self.queue_size)
        # Increments the producer count and releases the lock.
        self.producers_no += 1                                  
        self.register_lock.release()                            
        return producer_id

    def publish(self, producer_id: int, product) -> bool:
        """
        Publishes a product to the specified producer's queue.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Any): The product to publish.

        Returns:
            bool: True if the product was successfully published, False if the queue is full.
        """

        
        # Block Logic: Attempts to add the product to the producer's queue without blocking.
        try:
            self.producer_queues[producer_id].put_nowait(product)
        # If the queue is full, catches the Full exception and returns False.
        except Full:
            return False

        # If successful, returns True.
        return True

    def new_cart(self) -> int:
        """
        Creates a new empty shopping cart and assigns it a unique cart ID.

        Returns:
            int: The unique ID of the newly created cart.
        """

        
        # Block Logic: Acquires a lock to ensure thread-safe cart creation.
        self.cart_lock.acquire()                    
        # Assigns a unique cart ID.
        cart_id = self.consumers_no
        # Initializes an empty list for the new cart.
        self.consumer_carts[cart_id] = []           
        # Increments the cart counter and releases the lock.
        self.consumers_no += 1                      
        self.cart_lock.release()                    
        return cart_id



    def add_to_cart(self, cart_id: int, product) -> bool:
        """
        Attempts to add a specified product to a consumer's cart.
        It searches through producer queues to find the product.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Any): The product to add.

        Returns:
            bool: True if the product was successfully added, False otherwise.
        """

        
        # Retrieves the consumer's cart.
        cart = self.consumer_carts[cart_id]
        # Block Logic: Iterates through each producer's queue to find the requested product.
        for producer_id in range(0, self.producers_no):
            try:
                # Attempts to get an item from the producer's queue without blocking.
                queue_head = self.producer_queues[producer_id].get_nowait()

                # If the retrieved item matches the requested product, add it to the cart.
                if queue_head == product:
                    cart.append(queue_head)
                    return True

                # If the item is not the requested product, put it back into the queue.
                while True:
                    try:
                        self.producer_queues[producer_id].put_nowait(queue_head)
                        break
                    except Full:
                        # If the queue is full when trying to put back, continue retrying.
                        continue

            # If the producer's queue is empty, move to the next producer.
            except Empty:
                continue

        # If the product was not found in any producer's queue, returns False.
        return False

    def remove_from_cart(self, cart_id: int, product) -> None:
        """
        Removes a specified product from a consumer's cart and returns it to the marketplace
        (specifically to producer 0's queue, acting as a return mechanism).

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Any): The product to remove.
        """

        
        # Block Logic: Attempts to remove the product from the consumer's cart.
        try:
            # Removes the product from the cart list.
            self.consumer_carts[cart_id].remove(product)
            # Publishes the removed product back to producer_id 0's queue (acting as a return).
            self.publish(0, product)
        # If the product is not found in the cart, catches ValueError and does nothing.
        except ValueError:
            pass



    def place_order(self, cart_id: int) -> list:
        """
        Places a consumer's order by returning the contents of their cart.
        Note: This method currently just returns the cart contents without further processing.

        Args:
            cart_id (int): The ID of the consumer's cart.

        Returns:
            list: A list of products in the placed order.
        """

        
        # Block Logic: Returns the list of items currently in the specified cart.
        return self.consumer_carts[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    A producer thread that continuously generates products and publishes them to the marketplace.
    Each producer has a set of products to produce, with specified quantities and production times.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains (product_object, quantity, production_delay).
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (float): The time to wait before retrying to publish if the queue is full.
            **kwargs: Arbitrary keyword arguments, including 'name' and 'daemon' for the thread.
        """

        
        super().__init__(name=kwargs["name"], daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread.
        It continuously produces and publishes products to the marketplace,
        waiting for specified production and republishing times.
        """
        while True:
            # Block Logic: Iterates through each product type the producer is configured to create.
            for product in self.products:                   
                produced = 0                                
                waited = False                              

                # Block Logic: Continues producing the current product until the desired quantity is reached.
                while produced < product[1]:                
                    # If not waiting (i.e., this is a fresh production attempt), sleep for the product's production delay.
                    if not waited:
                        sleep(product[2])                   

                    # Attempts to publish the product to the marketplace.
                    if self.marketplace.publish(self.producer_id, product[0]):
                        produced += 1 # Increment count of successfully produced items.
                        waited = False # Reset waited flag.
                    else:
                        sleep(self.republish_time) # If publishing fails (queue full), wait before retrying.
                        waited = True # Set waited flag to avoid immediate sleep on retry.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True, eq=True)
class Product:
    """
    Base class for all products in the marketplace.
    Uses dataclass for automatic __init__, __repr__, and comparison methods.
    It's frozen, meaning instances are immutable.
    """
    
    name: str  # The name of the product.
    price: int # The price of the product.


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a Tea product, inheriting from Product.
    Includes an additional attribute 'type' specific to Tea.
    """
    
    type: str # The type of tea (e.g., "Green", "Black", "Herbal").



@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a Coffee product, inheriting from Product.
    Includes additional attributes 'acidity' and 'roast_level' specific to Coffee.
    """
    
    acidity: str       # The acidity level of the coffee.
    roast_level: str   # The roast level of the coffee (e.g., "Light", "Medium", "Dark").