"""
This module defines classes for a simulated multi-threaded marketplace system, including:
- Consumer: Represents a buyer interacting with the marketplace.
- Marketplace: Manages product queues, consumer carts, and ensures thread safety.
- Producer: Represents a seller supplying products to the marketplace.
- Product, Tea, Coffee: Data classes for defining various product types.

The system uses threading primitives like Locks, Events, and Queues for concurrent operations.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace. Inherits from `threading.Thread`.
    Consumers interact with the marketplace to add and remove products from their cart,
    and finally place an order.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of products the consumer intends to buy. Each element in the list is an order.
            marketplace (Marketplace): The marketplace instance the consumer interacts with.
            retry_wait_time (float): Time in seconds to wait before retrying an action (e.g., adding to cart).
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method,
                      including 'name' for the consumer's identifier.
        """
        
        super().__init__(name=kwargs["name"])
        self.carts: list = carts
        self.marketplace = marketplace
        self.retry_time = retry_wait_time
        self.output_str = "%s bought %s"

    def run(self):
        """
        Executes the consumer's shopping logic.
        - Processes each order in `self.carts`.
        - Creates a new cart in the marketplace for each order.
        - Adds and removes products from the cart based on the order requests.
        - Places the final order and prints the purchased products.
        """
        # Block Logic: Continue processing orders as long as there are items in the carts list.
        while len(self.carts) != 0:
            
            order = self.carts.pop(0) # Get the next order to process.
            
            cart_id = self.marketplace.new_cart() # Create a new cart for the current order.

            # Block Logic: Process each request (add/remove product) within the current order.
            while len(order) != 0:
                
                request = order.pop(0) # Get the next request from the order.

                
                if request["type"] == "add": # Block Logic: Handle product addition.
                    added_products = 0                           # Counter for products successfully added
                    # Invariant: Loop until the requested quantity is added.
                    while added_products < request["quantity"]:  
                        
                        # Block Logic: Attempt to add the product to the cart.
                        if self.marketplace.add_to_cart(cart_id, request["product"]):
                            added_products += 1 # Increment counter if successful.
                        else:
                            sleep(self.retry_time) # Wait and retry if adding fails.             

                
                if request["type"] == "remove": # Block Logic: Handle product removal.
                    # Block Logic: Remove the specified quantity of the product from the cart.
                    for _ in range(0, request["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, request["product"])

            
            # Block Logic: Place the final order and print the purchased products.
            cart_items = self.marketplace.place_order(cart_id)
            for product in cart_items:
                print(self.output_str % (self.name, product))    


from threading import Lock
from queue import Queue, Full, Empty
from typing import Dict


class Marketplace:
    """
    Manages the overall marketplace operations, including producer queues,
    consumer carts, and ensuring thread-safe access to shared resources.
    It provides functionalities for producer registration, publishing products,
    creating new carts, adding/removing products from carts, and placing orders.
    """
    

    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the Marketplace with a specified queue size per producer.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have in its queue.
        """
        

        
        self.register_lock = Lock()                     # Lock for protecting producer registration
        self.producers_no = 0                           # Counter for assigning unique producer IDs
        self.queue_size = queue_size_per_producer       # Max size for each producer's queue
        self.producer_queues: Dict[int, Queue] = {}     # Dictionary to store product queues for each producer

        
        self.cart_lock = Lock()                         # Lock for protecting consumer cart operations
        self.consumers_no = 0                           # Counter for assigning unique consumer (cart) IDs
        self.consumer_carts: Dict[int, list] = {}       # Dictionary to store carts for each consumer

        # Block Logic: Register a default producer (producer_id 0) that can ignore queue size limits.
        self.register_producer(ignore_limit=True)

    def register_producer(self, ignore_limit: bool = False) -> int:
        """
        Registers a new producer with the marketplace and assigns a unique ID.
        Ensures thread-safe registration using `register_lock`.

        Args:
            ignore_limit (bool): If True, the producer's queue will have no size limit.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        
        self.register_lock.acquire()                            # Acquire lock to protect shared producer counters/queues.
        producer_id = self.producers_no                         # Assign new producer ID.
        # Block Logic: Initialize the producer's queue based on the ignore_limit flag.
        if ignore_limit:
            
            self.producer_queues[producer_id] = Queue()         # Create an unbounded queue.
        else:
            
            self.producer_queues[producer_id] = Queue(self.queue_size) # Create a bounded queue.
        self.producers_no += 1                                  # Increment total producers count.
        self.register_lock.release()                            # Release lock.
        return producer_id



    def publish(self, producer_id: int, product) -> bool:
        """
        Publishes a product to the marketplace from a specific producer.
        The product is added to the producer's queue.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Any): The product to publish.

        Returns:
            bool: True if the product was successfully published, False if the queue is full.
        """
        
        try:
            self.producer_queues[producer_id].put_nowait(product) # Attempt to add product without blocking.
        except Full:
            return False # Return False if the queue is full.
        return True

    def new_cart(self) -> int:
        """
        Creates a new shopping cart for a consumer and assigns a unique ID.
        Ensures thread-safe cart creation using `cart_lock`.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        
        self.cart_lock.acquire()                    # Acquire lock to protect shared consumer cart counters.
        cart_id = self.consumers_no                 # Assign new cart ID.
        self.consumer_carts[cart_id] = []           # Initialize an empty list for the new cart.
        self.consumers_no += 1                      # Increment total consumers count.
        self.cart_lock.release()                    # Release lock.
        return cart_id

    def add_to_cart(self, cart_id: int, product) -> bool:
        """
        Adds a product to a specific cart.
        It searches through all producer queues for the product, removes it from the first found,
        and adds it to the consumer's cart. Ensures thread-safe operation.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (Any): The product to add.

        Returns:
            bool: True if the product was successfully added to the cart, False otherwise.
        """


        
        cart = self.consumer_carts[cart_id] # Get the specific cart.
        # Block Logic: Iterate through all producer queues to find the desired product.
        for producer_id in range(0, self.producers_no):
            try:
                # Block Logic: Attempt to get a product from the current producer's queue without blocking.
                queue_head = self.producer_queues[producer_id].get_nowait()

                # Block Logic: If the retrieved product matches the desired product.
                if queue_head == product:
                    
                    cart.append(queue_head) # Add to consumer's cart.
                    return True # Product successfully added.

                # Block Logic: If the retrieved product does not match, put it back into the queue.
                while True:
                    
                    try:
                        self.producer_queues[producer_id].put_nowait(queue_head)
                        break
                    except Full:
                        # Block Logic: If the queue is full, continue trying to put it back (busy-wait).
                        continue

            except Empty:
                # Block Logic: If the producer's queue is empty, continue to the next producer.
                continue

        return False

    def remove_from_cart(self, cart_id: int, product) -> None:
        """
        Removes a product from a specific cart and returns it to the default producer's queue (producer 0).

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (Any): The product to remove.
        """
        
        try:
            
            self.consumer_carts[cart_id].remove(product) # Remove the product from the consumer's cart.
            
            self.publish(0, product) # Return the product to producer 0's queue.
        except ValueError:
            # Block Logic: If the product is not found in the cart, do nothing (pass).
            pass

    def place_order(self, cart_id: int) -> list:
        """
        Places an order for all items currently in the specified cart.
        The items are returned as a list.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: A list of products that were part of the placed order.
        """
        
        return self.consumer_carts[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace. Inherits from `threading.Thread`.
    Producers register with the marketplace and continuously publish products.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products the producer will publish.
                             Each product is a tuple: (product_id, quantity, produce_time).
            marketplace (Marketplace): The marketplace instance the producer interacts with.
            republish_wait_time (float): Time in seconds to wait before retrying a publish operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method,
                      including 'name' and 'daemon' for the producer's identifier and daemon status.
        """
        
        super().__init__(name=kwargs["name"], daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        Executes the producer's main logic.
        - Continuously produces and publishes products defined in `self.products`.
        - Handles re-publishing attempts if the marketplace queue is full.
        """
        # Block Logic: Main loop for continuous product production and publishing.
        while True:
            # Block Logic: Iterate through each product defined for this producer.
            # Pre-condition: `product` is a tuple (product_id, quantity, produce_time).
            for product in self.products:                   
                produced = 0                                # Counter for products successfully produced in the current batch.
                waited = False                              # Flag to indicate if the producer has waited due to a full queue.

                # Block Logic: Loop until the desired quantity of the current product is produced.
                while produced < product[1]:                
                    # Block Logic: If the producer hasn't waited, pause for the specified produce_time.
                    if not waited:
                        sleep(product[2])                   

                    # Block Logic: Attempt to publish the product to the marketplace.
                    if self.marketplace.publish(self.producer_id, product[0]):
                        produced += 1 # Increment produced count if successful.
                        waited = False # Reset waited flag.
                    else:
                        sleep(self.republish_time)          # Wait and retry if publishing fails (queue full).
                        waited = True # Set waited flag.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True, eq=True)
class Product:
    """
    Base data class for a product.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Data class for a Tea product, inheriting from Product.

    Attributes:
        type (str): The type of tea (e.g., Green, Black, Herbal).
    """
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Data class for a Coffee product, inheriting from Product.

    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    
    acidity: str
    roast_level: str
