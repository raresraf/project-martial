

"""
This module implements a producer-consumer simulation for a marketplace.

It includes three main classes:
- Marketplace: A shared resource that manages products and shopping carts.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places orders.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer in the marketplace. A consumer processes a list of
    shopping carts, adding and removing products as specified in each cart's
    actions, and then places an order for each cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer thread.

        Args:
            carts: A list of carts, where each cart is a list of actions
                   (add or remove products).
            marketplace: The shared Marketplace object.
            retry_wait_time: The time to wait before retrying a failed action.
            **kwargs: Additional keyword arguments for the Thread constructor,
                      such as 'name'.
        """
        Thread.__init__(self, name=kwargs["name"])
        self.carts = carts
        self.marketplace = marketplace


        self.retry_wait_time = retry_wait_time
        # Convenience references to the marketplace methods.
        self.add_to_cart = self.marketplace.add_to_cart
        self.remove_from_cart = self.marketplace.remove_from_cart

    def run(self):
        """
        The main execution loop for the consumer thread. It iterates through
        its assigned carts, processes the actions in each cart, and places an
        order.
        """
        for cart in self.carts:
            # Create a new cart in the marketplace for each set of actions.
            cart_id = self.marketplace.new_cart()

            # Process each action in the cart (add or remove products).
            for action in cart:
                num_of_rep = action["quantity"]
                while num_of_rep > 0:
                    result = False
                    
                    if action["type"] == "add":
                        result = self.add_to_cart(cart_id=cart_id, product=action["product"])
                    elif action["type"] == "remove":
                        result = self.remove_from_cart(cart_id=cart_id, product=action["product"])

                    # If the action fails, wait before retrying.
                    if result is False:
                        sleep(self.retry_wait_time)
                    else:
                        num_of_rep = num_of_rep - 1

            self.marketplace.place_order(cart_id=cart_id)


from threading import Lock, currentThread

INITIAL_NUM_PRODUCTS = 0


class Marketplace:
    """
    A thread-safe marketplace that manages products, producers, and consumer carts.
    It provides methods for registering producers, publishing products, creating carts,
    adding/removing products from carts, and placing orders.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace with a specified queue size for each producer.

        Args:
            queue_size_per_producer: The maximum number of products a single
                                     producer can have in the marketplace at one time.
        """
        
        self.queue_size_per_producer = queue_size_per_producer
        
        self.size_per_producer = []
        
        self.products = []
        
        self.carts = {}
        
        self.mapper_products = {}
        
        
        self.index_producer_id = 0
        
        
        self.index_cart_id = 0

        self.register_producer_lock = Lock()
        self.publish_product_lock = Lock()
        self.new_cart_create_lock = Lock()
        self.operate_product_lock = Lock()
        self.print_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer with the marketplace and returns a unique producer ID.
        """
        
        
        
        
        self.register_producer_lock.acquire()
        producer_id = self.index_producer_id
        self.index_producer_id = self.index_producer_id + 1
        self.size_per_producer.append(INITIAL_NUM_PRODUCTS)
        self.register_producer_lock.release()

        return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace from a specific producer.

        Args:
            producer_id: The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            True if the product was successfully published, False otherwise (e.g., if
            the producer's queue is full).
        """
        
        
        
        self.publish_product_lock.acquire()
        if self.size_per_producer[producer_id] >= self.queue_size_per_producer:
            self.publish_product_lock.release()
            return False

        self.size_per_producer[producer_id] += 1
        self.products.append(product)
        self.mapper_products[product] = producer_id
        self.publish_product_lock.release()

        return True

    def new_cart(self):
        """
        Creates a new shopping cart and returns its unique ID.
        """
        
        
        
        self.new_cart_create_lock.acquire()
        self.index_cart_id += 1
        cart_id = self.index_cart_id
        self.carts[cart_id] = []
        self.new_cart_create_lock.release()

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.

        Args:
            cart_id: The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            True if the product was successfully added, False otherwise (e.g.,
            if the product is not available).
        """
        
        
        self.operate_product_lock.acquire()
        if product not in self.products:
            self.operate_product_lock.release()
            return False

        producer_id = self.mapper_products[product]
        self.size_per_producer[producer_id] = self.size_per_producer[producer_id] - 1
        self.products.remove(product)
        self.carts[cart_id].append(product)
        self.operate_product_lock.release()

        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the marketplace.
        """
        
        
        self.operate_product_lock.acquire()
        product_id = self.mapper_products[product]
        self.carts[cart_id].remove(product)
        self.size_per_producer[product_id] = self.size_per_producer[product_id] + 1
        self.products.append(product)
        self.operate_product_lock.release()

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart, printing the purchased products.
        """
        
        
        
        
        products_list = self.carts.pop(cart_id)
        
        for product in products_list:
            self.print_lock.acquire()
            print(currentThread().getName() + " bought " + str(product))
            self.print_lock.release()

        return products_list


from threading import Thread
import time

PRODUCT_INDEX = 0
QUANTITY_INDEX = 1
SLEEP_TIME_INDEX = 2


class Producer(Thread):
    """
    Represents a producer in the marketplace. A producer is responsible for
    publishing products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer thread.

        Args:
            products: A list of products that the producer can publish.
            marketplace: The shared Marketplace object.
            republish_wait_time: The time to wait before retrying to publish a
                                 product if the marketplace is full.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, name=kwargs["name"], daemon=kwargs["daemon"])

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread. It continuously
        publishes its products to the marketplace.
        """
        while True:
            
            
            for elem in self.products:
                
                for count in range(0, elem[QUANTITY_INDEX]):
                    
                    if self.marketplace.publish(product=elem[PRODUCT_INDEX],
                                                producer_id=self.prod_id):
                        time.sleep(elem[SLEEP_TIME_INDEX])
                    
                    else:
                        time.sleep(self.republish_wait_time)
                        count = count - 1
