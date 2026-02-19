"""
@file consumer.py
@brief A multi-threaded producer-consumer simulation of an e-commerce marketplace.
@details This module defines a `Marketplace` and corresponding `Producer` and `Consumer` threads.
It also includes dataclass definitions for products, suggesting a more complex data model,
though the simulation itself appears to use simple strings for products. The implementation
uses multiple semaphores for fine-grained locking. Variable names suggest a non-English origin (likely Romanian).
"""

import time
from threading import Thread, Semaphore

# The following dataclasses define a product hierarchy but are not directly instantiated
# or used within the core simulation logic provided in this file. They suggest a
# planned or more extensive data model for the marketplace items.
from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea, inheriting from Product and adding a 'type' attribute."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, inheriting from Product and adding acidity and roast level."""
    acidity: str
    roast_level: str


class Consumer(Thread):
    """
    @brief Represents a consumer thread that buys products from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts A list of shopping lists, each with 'add'/'remove' commands.
        @param marketplace The shared Marketplace object.
        @param retry_wait_time Time to wait before retrying to add an unavailable product.
        """
        Thread.__init__(self, kwargs=kwargs)
        self.name = kwargs['name']
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.carts_list =[]

    def afisare(self, lista):
        """
        @brief Helper function to print the items bought.
        @details 'afisare' is Romanian for 'display'.
        """
        for item in lista:
            print(f'{self.name} bought {item}')

    def run(self):
        """
        @brief Main loop for the consumer, processing each cart's commands.
        """
        for item in self.carts:
            id_cart = self.marketplace.new_cart()
            self.carts_list.append(id_cart)
            # 'comanda' is Romanian for 'command'.
            for comanda in item:
                if comanda['type'] == 'add':
                    # Block Logic: Attempt to add the specified quantity of a product,
                    # waiting and retrying if the product is not available.
                    for _ in range(comanda['quantity']):
                        while self.marketplace.add_to_cart(id_cart, comanda['product']) is False:
                            time.sleep(self.retry_wait_time)
                if comanda['type'] == 'remove':
                    for _ in range(comanda['quantity']):
                        self.marketplace.remove_from_cart(
                            id_cart, comanda['product'])
        # After processing commands, place the orders for all created carts.
        for cart in self.carts_list:
            self.afisare(self.marketplace.place_order(cart))


class Marketplace:
    """
    @brief The central marketplace managing inventory and carts.
    @details This class uses fine-grained locking with multiple semaphores to control
    concurrent access from producer and consumer threads.
    """

    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.id_cart = 0
        
        # Using a dictionary to store each producer's inventory, keyed by producer ID.
        self.database = {}
        # Using a dictionary to store each consumer's cart contents, keyed by cart ID.
        self.carts = {}

        # These semaphores initialized to 1 act as locks for different methods.
        self.sem_cart = Semaphore(1)       # For new_cart
        self.sem_cons = Semaphore(1)       # For add_to_cart
        self.sem_remove = Semaphore(1)     # For remove_from_cart
        self.sem_register = Semaphore(1)   # For register_producer
        self.sem_place = Semaphore(1)      # For place_order

    def register_producer(self):
        """@brief Registers a new producer and allocates an inventory list for them."""
        self.sem_register.acquire()
        id_producer_act = self.id_producer+1
        self.id_producer = self.id_producer+1
        self.database[f'id{str(id_producer_act)}'] = []
        self.sem_register.release()
        return f'id{str(id_producer_act)}'

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add a product to their inventory if not full.
        @return True on success, False if the producer's inventory queue is full.
        """
        if len(self.database[producer_id]) < self.queue_size_per_producer:
            self.database[producer_id].append(product)
            return True
        return False

    def new_cart(self):
        """@brief Creates a new, empty cart for a consumer."""
        self.sem_cart.acquire()
        id_cart_actual = self.id_cart
        self.id_cart = self.id_cart+1
        self.carts[id_cart_actual] = []
        self.sem_cart.release()
        return id_cart_actual

    def add_to_cart(self, cart_id, product):
        """@brief Moves a product from a producer's inventory to a consumer's cart."""
        self.sem_cons.acquire()
        # 'producator' is Romanian for 'producer'.
        for producator in self.database:
            for item in self.database[producator]:
                if item == product:
                    # Store product and its original producer ID in the cart.
                    self.carts[cart_id].append((item, producator))
                    self.database[producator].remove(item)
                    self.sem_cons.release()
                    return True
        self.sem_cons.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a cart and returns it to the producer's inventory.
        @warning This method has a potential deadlock bug. If the product is not found in the
        cart, the 'break' is never hit and the semaphore `sem_remove` is never released.
        """
        self.sem_remove.acquire()
        # 'produs' is Romanian for 'product'.
        for produs, producator in self.carts[cart_id]:
            if produs == product:
                self.database[producator].append(produs)
                self.carts[cart_id].remove((produs, producator))
                self.sem_remove.release()
                break

    def place_order(self, cart_id):
        """@brief Finalizes an order and returns the list of products bought."""
        self.sem_place.acquire()
        result = []
        for produs in self.carts[cart_id]:
            result.append(produs[0])
        self.sem_place.release()
        return result


class Producer(Thread):
    """
    @brief Represents a producer thread that publishes products to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, kwargs=kwargs)
        self.name = kwargs['name']
        self.daemon = kwargs['daemon']
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.products = products

    def run(self):
        """
        @brief Main loop for the producer, continuously publishing products.
        """
        id_prod = self.marketplace.register_producer()
        while True:
            # 'produs' = product, 'cantitate' = quantity, 'timp' = time
            for (produs, cantitate, timp) in self.products:
                for _ in range(cantitate):
                    # Block Logic: Retry publishing until successful.
                    # If the marketplace queue is full, wait and retry.
                    while self.marketplace.publish(id_prod, produs) is False:
                        time.sleep(self.republish_wait_time)
                    # Wait for a product-specific time after successful publication.
                    time.sleep(timp)
