
"""
Provides a multi-threaded simulation of an e-commerce marketplace
with producers, consumers, and a central marketplace.
"""

from threading import Thread, Lock
import time


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer simulates a user who creates a shopping cart, adds or removes
    products based on a predefined list of actions, and then places an order.
    """
    cart_id = -1
    name = ''
    my_lock = Lock()

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping operations to perform.
            marketplace (Marketplace): The marketplace object to interact with.
            retry_wait_time (float): Time to wait before retrying to add a product.
            **kwargs: Additional keyword arguments, including the consumer's name.
        """
        Thread.__init__(self)
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.name = kwargs['name']
        

    def run(self):
        """The main execution logic for the consumer thread."""
        self.my_lock.acquire()
        for i in range(len(self.carts)):
            self.cart_id = self.marketplace.new_cart()


            for j in range(len(self.carts[i])):
                if self.carts[i][j]['type'] == 'add':
                    for k in range(self.carts[i][j]['quantity']):
                        verify = False
                        while not verify:
                            verify = self.marketplace.add_to_cart(self.cart_id,
                                                                  self.carts[i][j]['product']
                                                                  )
                            if not verify:
                                time.sleep(self.retry_wait_time)



                elif self.carts[i][j]['type'] == 'remove':
                    for k in range(self.carts[i][j]['quantity']):
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
            list_1 = self.marketplace.place_order(self.cart_id)
            for k in range(len(list_1) - 1, -1, -1):
                print(self.name + ' bought ' + str(list_1[k][0]))
                self.marketplace.remove_from_cart(self.cart_id, list_1[k][0])
        self.my_lock.release()


from threading import Lock


class Marketplace:
    """
    Manages the inventory, producers, and carts in the simulation.
    
    This class acts as the central hub for all interactions, providing
    thread-safe methods for publishing products, creating carts, and
    managing orders.
    """
    id_producer = 0
    id_cart = 0
    queues = []
    carts = []
    my_Lock1 = Lock()
    my_Lock2 = Lock()
    done = 0

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products
                a single producer can have in their queue.
        """
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """Registers a new producer and assigns them an ID and a product queue."""
        self.queues.append([])
        self.id_producer = self.id_producer + 1
        return self.id_producer - 1

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to their queue.

        Args:
            producer_id (int): The ID of the producer.
            product: The product to be published.

        Returns:
            bool: True if publishing was successful, False otherwise (e.g., queue is full).
        """
        if len(self.queues[producer_id]) >= self.queue_size_per_producer:
            return False
        self.queues[producer_id].append([product, "Disponibil"])
        return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its ID."""
        self.carts.append([])
        self.id_cart = self.id_cart + 1
        return self.id_cart - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.

        This method searches all producer queues for an available product. If found,
        it's added to the cart, and its status is marked as 'Indisponibil'. This
        operation is protected by a lock to ensure thread safety.

        Args:
            cart_id (int): The ID of the cart.
            product: The product to add.

        Returns:
            bool: True if the product was successfully added, False otherwise.
        """
        verify = 0
        for i in range(len(self.queues)):
            for j in range(len(self.queues[i])):
                self.my_Lock1.acquire()
                if product == self.queues[i][j][0] \
                        and self.queues[i][j][1] == 'Disponibil' \
                        and verify == 0:
                    self.carts[cart_id].append([product, i])
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

        This operation is protected by a lock to ensure thread safety.

        Args:
            cart_id (int): The ID of the cart.
            product: The product to remove.

        Returns:
            bool: True if the product was successfully removed, False otherwise.
        """
        for i in range(len(self.carts[cart_id])):
            if product == self.carts[cart_id][i][0]:
                for j in range(len(self.queues[self.carts[cart_id][i][1]])):
                    self.my_Lock2.acquire()
                    if self.queues[self.carts[cart_id][i][1]][j][0] == product \
                            and self.queues[self.carts[cart_id][i][1]][j][1] == 'Indisponibil':
                        self.queues[self.carts[cart_id][i][1]][j][1] = 'Disponibil'
                        self.carts[cart_id].remove(self.carts[cart_id][i])
                        self.my_Lock2.release()
                        return True
                    self.my_Lock2.release()
        return False

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the contents of the cart.

        Args:
            cart_id (int): The ID of the cart to place an order for.

        Returns:
            list: The list of products in the cart.
        """
        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """
    producer_id = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer instance.

        Args:
            products (list): A list of products that the producer can create.
            marketplace (Marketplace): The marketplace object to interact with.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Additional keyword arguments, including daemon status.
        """
        Thread.__init__(self, daemon=kwargs['daemon'])
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        

    def run(self):
        """The main execution logic for the producer thread."""
        self.producer_id = self.marketplace.register_producer()


        while True:
            for i in range(len(self.products)):
                for j in range(self.products[i][1]):
                    verify = self.marketplace.publish(self.producer_id, self.products[i][0])
                    time.sleep(self.products[i][2])
                    if not verify:
                        time.sleep(self.republish_wait_time)
                        break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
