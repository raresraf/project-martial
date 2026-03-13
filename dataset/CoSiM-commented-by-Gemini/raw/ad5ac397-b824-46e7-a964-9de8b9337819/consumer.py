"""
A producer-consumer simulation for a marketplace.

This module contains classes to simulate a marketplace with producers and
consumers. Producers create products and add them to the marketplace, while
consumers add products to their carts and place orders. The simulation uses
threading to run producers and consumers concurrently and locks to protect
shared data.
"""


from threading import Thread
import time


class Consumer(Thread):
    """
    A consumer thread that interacts with the marketplace.

    Each consumer has a set of carts and adds products to them, then places an
    order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts: A list of carts, where each cart is a list of products.
            marketplace: The Marketplace object to interact with.
            retry_wait_time: The time to wait before retrying to add a product.
            **kwargs: Additional arguments for the Thread constructor.
        """
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time: int = retry_wait_time
        Thread.__init__(self, **kwargs)

    def print_carts(self, id_cart):
        """
        Places an order for a cart and prints the contents.

        Args:
            id_cart: The ID of the cart to place an order for.
        """

        
        list_order = self.marketplace.place_order(id_cart)
        for product in list_order:
            self.marketplace.print_cons(self.name, product)


    def add_product_to_cart(self, id_cart, prod):
        """
        Adds a product to a cart, retrying if it fails.

        Args:
            id_cart: The ID of the cart to add the product to.
            prod: The product to add.
        """
        
        
        go_next = self.marketplace.add_to_cart(id_cart,prod)
        if go_next is False:
            time.sleep(self.retry_wait_time)
            self.add_product_to_cart(id_cart, prod)


    def run(self):
        """
        The main execution loop for the consumer thread.

        This method simulates the consumer's behavior: getting a new cart,
        adding or removing products, and finally placing an order.
        """
        id_cart = self.marketplace.new_cart()
        for products in self.carts:
            for produs in products:
                for _ in range(produs["quantity"]):
                    if produs["type"] == "remove":
                        self.marketplace.remove_from_cart(id_cart, produs)
                    else:
                        self.add_product_to_cart(id_cart, produs)
        self.print_carts(id_cart)

from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import time


logger = logging.getLogger('loggerOne')
logger.setLevel(logging.INFO)


handler = RotatingFileHandler('file.log', maxBytes=500000, backupCount=10)


formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


logging.Formatter.converter = time.gmtime


class Marketplace:
    """
    The central marketplace that manages producers, products, and carts.

    This class provides a thread-safe interface for producers to publish
    products and for consumers to interact with their carts.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer: The maximum number of products a single
                producer can have in the marketplace at one time.
        """
        

        logger.info("init Marketplace, argument qsise: %d", queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer
        self.id_prod = 0
        self.id_cart = 0
        self.producers = []
        self.producers.append(0)
        self.products = []
        self.products.append([])
        self.carts = []
        
        self.add_to_cart_lock = Lock()
        self.publish_lock = Lock()
        self.print_lock = Lock()
        self.new_cart_lock = Lock()
        self.register_producer_lock = Lock()
        self.remove_from_cart_lock = Lock()

        logger.info("init Marketplace, all")

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Returns:
            The ID of the newly registered producer.
        """


        
        
        self.register_producer_lock.acquire()
        logger.info("register_producer, id_prod =%d", self.id_prod)
        self.producers.append(self.queue_size_per_producer)
        self.id_prod = self.id_prod + 1
        self.products.append([])


        self.register_producer_lock.release()

        logger.info("register_producer, id_prod-exit =%d", self.id_prod)
        return self.id_prod

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id: The ID of the producer publishing the product.
            product: The product to publish.

        Returns:
            True if the product was published successfully, False otherwise.
        """
        
        
        it_produced = False
        
        self.publish_lock.acquire()
        logger.info("publish; id_producer =%d", producer_id)
        if self.producers[producer_id] > 0:
            self.products[producer_id].append(product[0])
            self.producers[producer_id] = self.producers[producer_id] -1
            it_produced = True

        logger.info("publish; exit =%d", producer_id, )
        self.publish_lock.release()
        return it_produced


    def new_cart(self):
        """
        Creates a new, empty cart for a consumer.

        Returns:
            The ID of the newly created cart.
        """


        
        

        self.new_cart_lock.acquire()
        logger.info("new_cart;")

        self.carts.append([])
        current_cart_id = self.id_cart
        self.id_cart = self.id_cart+1

        logger.info("new_cart; iese")
        self.new_cart_lock.release()

        return current_cart_id


    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.

        Args:
            cart_id: The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            True if the product was added successfully, False otherwise.
        """
        
        
        producer_found_id = 0
        product_found = False
        available_product = []

        
        
        self.add_to_cart_lock.acquire()
        logger.info("ad_to_cart %d;", cart_id)

        for producer in self.products:
            for available_product in producer:
                if product["product"] == available_product:
                    product_found = True
                    break

            if product_found:
                break
            producer_found_id = producer_found_id + 1

        if product_found:
            self.products[producer_found_id].remove(available_product)
            self.producers[producer_found_id] = self.producers[producer_found_id] + 1
            self.carts[cart_id].append(available_product)
            logger.info("ad_to_cart exit True;")
            self.add_to_cart_lock.release()
            return True



        logger.info("ad_to_cart exit False;")
        self.add_to_cart_lock.release()
        return False


    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart.

        Args:
            cart_id: The ID of the cart to remove the product from.
            product: The product to remove.
        """

        product_found = False
        producer_found_id = 0
        available_product = []

        
        
        self.remove_from_cart_lock.acquire()
        logger.info("reomce_from_cart start True;%d", cart_id)
        for available_product in self.carts[cart_id]:
            if product["product"] == available_product:
                product_found = True
                break
            producer_found_id = producer_found_id + 1

        if product_found:
            del self.carts[cart_id][producer_found_id]
            self.products[0].append(available_product)

        logger.info("ad_to_cart exit;")
        self.remove_from_cart_lock.release()


    def place_order(self, cart_id):
        """
        Places an order for a cart, returning the products and clearing the cart.

        Args:
            cart_id: The ID of the cart to place an order for.

        Returns:
            A list of products that were in the cart.
        """
        
        
        logger.info("place_order start %d;", cart_id)
        copie =self.carts[cart_id]
        self.carts[cart_id] = []
        logger.info("place_order end;")
        return copie


    def print_cons(self, name, product):
        """
        Prints a message indicating that a consumer bought a product.

        Args:
            name: The name of the consumer.
            product: The product that was bought.
        """
        
        
        self.print_lock.acquire()
        logger.info("print_cons start; name=%s;", name)


        print(name, "bought", product)
        self.print_lock.release()


from threading import Thread
import time
class Producer(Thread):
    """
    A producer thread that creates products and publishes them to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products: A list of products that the producer can create.
            marketplace: The Marketplace object to interact with.
            republish_wait_time: The time to wait before retrying to publish a
                product.
            **kwargs: Additional arguments for the Thread constructor.
        """

        Thread.__init__(self, **kwargs)
        self.products  = products

        self.marketplace = marketplace
        self.republish_wait_time: int = republish_wait_time
        self.my_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread.

        This method continuously produces products and tries to publish them to
        the marketplace.
        """
        while True:
            for  prod in self.products:
                for _ in range(prod[1]):
                    it_worked = self.marketplace.publish(self.my_id, prod)
                    if it_worked:
                        time.sleep(prod[2])
                    else:
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple data class for representing a product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class for representing a tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    A data class for representing a coffee product, inheriting from Product.
    """
    acidity: str
    roast_level: str
