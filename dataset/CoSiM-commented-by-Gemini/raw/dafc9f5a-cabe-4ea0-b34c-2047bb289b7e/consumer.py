"""
A multithreaded producer-consumer simulation of an online marketplace.

This script models a marketplace with producers who publish products and
consumers who add products to carts and place orders. The simulation uses
Python's threading capabilities to run producers and consumers concurrently,
with a central Marketplace class managing the state of products and carts
using locks to prevent race conditions.
"""


from threading import Thread, Lock
import time


class Consumer(Thread):
    """
    Represents a consumer that simulates purchasing items from the marketplace.

    Each consumer runs in its own thread, creates a new cart, adds or removes
    items based on a predefined list of actions, and finally places an order.
    """
    
    cart_id = -1
    name = ''
    my_lock = Lock()

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of cart actions for the consumer to perform.
            marketplace (Marketplace): The central marketplace instance.
            retry_wait_time (int): Time in seconds to wait before retrying to add a product.
            **kwargs: Additional keyword arguments, expects 'name'.
        """
        Thread.__init__(self)
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.name = kwargs['name']
        

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through its assigned carts, processes add/remove operations for each,
        places an order, and prints the items bought. A coarse-grained lock is
        used to ensure that a consumer completes all its cart operations before
        another consumer starts, which may not be an ideal concurrency pattern.
        """
        # Acquire a lock to ensure this consumer completes all its work atomically.
        self.my_lock.acquire()
        # Iterate over each shopping journey (cart).
        for i in range(len(self.carts)):
            # Functional Utility: Initializes a new shopping cart in the marketplace for this session.
            self.cart_id = self.marketplace.new_cart()


            # Block Logic: Processes all actions (add/remove) for the current cart.
            for j in range(len(self.carts[i])):
                if self.carts[i][j]['type'] == 'add':
                    # Block Logic: Attempts to add a specified quantity of a product to the cart.
                    for k in range(self.carts[i][j]['quantity']):
                        verify = False
                        # Invariant: Loop continues until the product is successfully added to the cart.
                        # This simulates waiting for a product to become available.
                        while not verify:
                            verify = self.marketplace.add_to_cart(self.cart_id,
                                                                  self.carts[i][j]['product']
                                                                  )
                            # If adding fails, wait before retrying.
                            if not verify:
                                time.sleep(self.retry_wait_time)



                elif self.carts[i][j]['type'] == 'remove':
                    # Block Logic: Removes a specified quantity of a product from the cart.
                    for k in range(self.carts[i][j]['quantity']):
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
            
            # Finalizes the transaction for the current cart.
            list_1 = self.marketplace.place_order(self.cart_id)
            # Block Logic: Prints the purchased items and clears them from the cart post-order.
            for k in range(len(list_1) - 1, -1, -1):
                print(self.name + ' bought ' + str(list_1[k][0]))
                # Functional Utility: Removes the item from the cart after purchase confirmation.
                self.marketplace.remove_from_cart(self.cart_id, list_1[k][0])
        # Release the lock after all operations are complete.
        self.my_lock.release()


from threading import Lock


class Marketplace:
    """
    A central marketplace that manages producers, products, and customer carts.

    This class is intended to be thread-safe, using locks to manage concurrent
    access to its internal state from multiple producer and consumer threads.
    It maintains queues of products from producers and lists of items in customer carts.
    """
    
    id_producer = 0
    id_cart = 0
    queues = []
    carts = []
    my_Lock1 = Lock()
    my_Lock2 = Lock()
    done = 0

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have in the marketplace at one time.
        """
        
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        Registers a new producer, giving them a dedicated queue for their products.

        Returns:
            int: The ID assigned to the newly registered producer.
        """
        
        self.queues.append([])
        self.id_producer = self.id_producer + 1
        return self.id_producer - 1

    def publish(self, producer_id, product):
        """
        Publishes a product from a specific producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue is full.
        """
        
        # Pre-condition: Check if the producer's product queue has space.
        if len(self.queues[producer_id]) >= self.queue_size_per_producer:
            return False
        # A product is represented as a list [product_data, availability_status].
        self.queues[producer_id].append([product, "Disponibil"])
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The ID assigned to the new cart.
        """
        
        self.carts.append([])
        self.id_cart = self.id_cart + 1
        return self.id_cart - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.

        This method iterates through all producer queues to find an available
        instance of the requested product. It uses a lock to ensure that
        the product's state transition (from 'Disponibil' to 'Indisponibil')
        is atomic.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        
        verify = 0
        # Block Logic: Search all producer queues for the requested product.
        for i in range(len(self.queues)):
            for j in range(len(self.queues[i])):
                # Atomically check and claim an available product.
                self.my_Lock1.acquire()
                # Pre-condition: The product must match and be available ('Disponibil').
                if product == self.queues[i][j][0] 
                        and self.queues[i][j][1] == 'Disponibil' 
                        and verify == 0:
                    self.carts[cart_id].append([product, i])
                    # Invariant: Mark product as unavailable to prevent other consumers from taking it.
                    self.queues[i][j][1] = 'Indisponibil'
                    verify = 1
                    self.my_Lock1.release()
                    break
                self.my_Lock1.release()
                if verify == 1:
                    break
        if verify == 1:
            return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart, making it available again.

        This method uses a lock to ensure that the product's state transition
        (from 'Indisponibil' back to 'Disponibil') is atomic.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product: The product to remove.

        Returns:
            bool: True if the product was found and removed, False otherwise.
        """
        
        for i in range(len(self.carts[cart_id])):
            if product == self.carts[cart_id][i][0]:
                for j in range(len(self.queues[self.carts[cart_id][i][1]])):
                    # Atomically update the product's status back to available.
                    self.my_Lock2.acquire()
                    if self.queues[self.carts[cart_id][i][1]][j][0] == product 
                            and self.queues[self.carts[cart_id][i][1]][j][1] == 'Indisponibil':
                        # Invariant: The product is marked as available for other consumers.
                        self.queues[self.carts[cart_id][i][1]][j][1] = 'Disponibil'
                        self.carts[cart_id].remove(self.carts[cart_id][i])
                        self.my_Lock2.release()
                        return True
                    self.my_Lock2.release()
        return False

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the items currently in the cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: The list of items in the cart.
        """
        

        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer that generates and publishes products to the marketplace.

    Each producer runs in its own thread and continuously publishes a list of
    products according to specified quantities and timings.
    """
    
    producer_id = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products for the producer to publish.
                             Each item contains product data, quantity, and publish interval.
            marketplace (Marketplace): The central marketplace instance.
            republish_wait_time (int): Time to wait before retrying to publish
                                       if the marketplace queue is full.
            **kwargs: Additional keyword arguments, expects 'daemon'.
        """
        Thread.__init__(self, daemon=kwargs['daemon'])
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers itself with the marketplace and then enters an infinite loop
        to publish its products.
        """
        self.producer_id = self.marketplace.register_producer()

        # Invariant: The producer continuously tries to publish its products.
        while True:
            # Iterate through the defined list of products to publish.
            for i in range(len(self.products)):
                for j in range(self.products[i][1]):
                    # Attempt to publish one unit of the product.
                    verify = self.marketplace.publish(self.producer_id, self.products[i][0])
                    # Wait for the product-specific interval before the next publish attempt.
                    time.sleep(self.products[i][2])
                    # If publishing failed (e.g., queue full), wait and break to retry later.
                    if not verify:
                        time.sleep(self.republish_wait_time)
                        break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A data class representing a generic product with a name and price."""
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class representing Tea, inheriting from Product and adding a 'type' attribute."""
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class for Coffee, adding 'acidity' and 'roast_level' attributes."""
    
    acidity: str
    roast_level: str
