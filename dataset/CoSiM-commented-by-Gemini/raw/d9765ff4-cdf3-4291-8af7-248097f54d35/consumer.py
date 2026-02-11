
"""
This module implements a complete Producer-Consumer simulation.

It defines `Producer` and `Consumer` threads that interact through a central
`Marketplace` class, which acts as the shared resource for publishing and
retrieving products.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that shops at the marketplace.

    Each consumer processes a list of shopping carts, adding and removing
    products, and then places an order.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        The main execution loop for the consumer.

        Iterates through its assigned shopping lists, interacts with the
        marketplace to fulfill them, and prints the final purchased products.
        """
        for shopping in self.carts:
            num_cart = self.marketplace.new_cart()
            for product in shopping:
                number_action = int(product['quantity'])
                command = product['type']
                name_product = product['product']
                while number_action != 0:
                    # Logic for adding a product to the cart.
                    if command == "add":
                        # If adding fails (e.g., product not available), wait and retry.
                        if self.marketplace.add_to_cart(num_cart, name_product):
                            number_action = number_action - 1
                        else:
                            sleep(self.wait_time)
                    # Logic for removing a product from the cart.
                    if command == "remove":
                        self.marketplace.remove_from_cart(num_cart, name_product)
                        number_action = number_action - 1

            # Finalize the order and print the results.
            shopping = self.marketplace.place_order(num_cart)
            for _, product in shopping:
                print(self.name, "bought", product)

from threading import Lock


class Marketplace:
    """
    Manages the inventory and transactions between producers and consumers.

    This class uses class-level variables for storing producers and consumers,
    meaning all instances of Marketplace share the same underlying data. This
    effectively makes it a singleton in terms of state.
    """
    
    # Class-level variables to hold the state shared across all instances.
    producers = {}
    consumers = {}
    id_prod = 1
    id_cons = 1
    lock_producer = Lock()
    lock_cart = Lock()
    def __init__(self, queue_size_per_producer):
        
        self.size = queue_size_per_producer


    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        self.lock_producer.acquire()
        products = []
        self.producers[self.id_prod] = products
        self.id_prod = self.id_prod+1
        self.lock_producer.release()
        return self.id_prod-1


    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Returns:
            bool: True if the product was successfully published, False if the
                  producer's queue is full.
        """


        if len(self.producers[producer_id]) == self.size:
            return False
        self.producers[producer_id].append(product)
        return True


    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        self.lock_cart.acquire()
        cart = []
        self.consumers[self.id_cons] = cart
        self.id_cons = self.id_cons + 1
        self.lock_cart.release()
        return self.id_cons - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart by searching for it among all producers.

        This method performs a linear search through all producers' inventories.
        If the product is found, it is moved to the consumer's cart.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        for producer in self.producers:
            for prod in self.producers[producer]:
                if product == prod:


                    self.consumers[cart_id].insert(0, [producer, product])
                    self.producers[producer].remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart and returns it to the producer.
        """
        for cart in self.consumers:
            if cart == cart_id:


                for index, prod in self.consumers[cart]:
                    if prod == product:
                        self.consumers[cart_id].remove([index, product])
                        self.producers[index].append(product)
                        return None
        return None

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the contents of the cart.
        """
        return self.consumers[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time


    def run(self):
        """
        The main execution loop for the producer.

        It continuously tries to publish its products to the marketplace, waiting
        and retrying if the marketplace queue is full.
        """
        id_producer = self.marketplace.register_producer()
        while True:
            for product in self.products:
                name_product = product[0]
                number_pieces = int(product[1])
                time_product = product[2]

                while number_pieces != 0:
                    # Attempt to publish a product.
                    if self.marketplace.publish(id_producer, name_product):
                        sleep(time_product)
                    else:
                        # If publishing fails, wait before retrying.
                        sleep(self.wait_time)
                    number_pieces = number_pieces - 1
