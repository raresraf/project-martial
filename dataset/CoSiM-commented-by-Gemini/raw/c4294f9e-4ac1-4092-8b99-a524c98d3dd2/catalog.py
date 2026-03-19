"""
This module implements a multi-threaded simulation of an online marketplace.

It defines the core components of the simulation:
- A thread-safe Catalog for managing product inventory.
- Producer and Consumer threads that interact with the marketplace.
- A central Marketplace that orchestrates all operations, including inventory
  management, shopping carts, and order placement, using locks to ensure
  thread safety.
- Dataclasses for defining product types.
"""

from collections.abc import MutableMapping
from threading import RLock


class Catalog():
    """
    Represents a thread-safe inventory for a single producer.
    
    Manages products by tracking both available count and 'frozen' count,
    representing items that are reserved in customer carts.
    """
    
    def __init__(self, max_elems):
        """
        Initializes a catalog with a maximum capacity.

        Args:
            max_elems (int): The maximum number of items the catalog can hold.
        """
        self.lock = RLock()
        # inventory maps a product to a tuple: (available_count, frozen_count)
        self.inventory = {}
        self.max_elems = max_elems
        self.size = 0 # Total items, both available and frozen.

    def add_product(self, product):
        """
        Adds a product to the inventory. If the catalog is full, fails.
        If the product already exists, its available count is incremented.
        """
        with self.lock:
            if self.size == self.max_elems:
                return False
            try:
                tup = self.inventory[product]
                (count, frozen) = tup
                self.inventory[product] = (count + 1, frozen)
            except KeyError:
                # If product is new, initialize with 1 available, 0 frozen.
                self.inventory[product] = (1, 0)
            self.size += 1
        return True

    def order_product(self, product):
        """
        Finalizes an order for a product, consuming a 'frozen' item.
        """
        with self.lock:
            (count, frozen) = self.inventory[product]
            # Decrement the frozen count as the item is now sold.
            self.inventory[product] = (count, frozen - 1)
            self.size -= 1

    def free_product(self, product):
        """
        Returns a reserved ('frozen') product back to the available stock.
        This happens when a customer removes an item from their cart.
        """
        with self.lock:
            if product not in self.inventory:
                return False
            (count, frozen) = self.inventory[product]
            # Move item from frozen back to available count.
            self.inventory[product] = (count + 1, frozen - 1)
            return True

    def reserve_product(self, product):
        """
        Reserves an available product, moving it to the 'frozen' state.
        This happens when a customer adds an item to their cart.
        """
        with self.lock:
            if product not in self.inventory:
                with open("inventory.txt", "w") as f:
                    f.write(str(product) + " not found in " +
                            str(self.inventory))
                return False
            (count, frozen) = self.inventory[product]
            # Fails if no items are available.
            if count == 0:
                return False
            # Move item from available to frozen count.
            self.inventory[product] = (count - 1, frozen + 1)
        return True


from threading import Thread
from time import sleep
from tema.marketplace import Marketplace


class Consumer(Thread):
    """
    A thread that simulates a customer shopping in the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of operations.
            marketplace (Marketplace): The central marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Executes the consumer's lifecycle: creating carts, adding/removing
        items with retry logic, and finally placing the order.
        """
        customer_id = self.marketplace.new_customers()
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                for _ in range(operation['quantity']):
                    sleep(self.retry_wait_time)
                    if operation['type'] == 'add':
                        # Retry adding to cart until successful.
                        finished = self.marketplace.add_to_cart(
                            cart_id, operation['product'])
                        while not finished:
                            sleep(self.retry_wait_time)
                            finished = self.marketplace.add_to_cart(
                                cart_id, operation['product'])
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(
                            cart_id, operation['product'])

            prods = self.marketplace.place_order(cart_id)
            for prod in prods:
                print("cons{} bought {}".format(customer_id, str(prod)))

import logging
import logging.handlers
from threading import RLock
from tema.catalog import Catalog


class Marketplace:
    """
    The central marketplace that orchestrates all producers, consumers, and transactions.
    
    This class is thread-safe and manages shared state such as producer inventories
    (catalogs) and customer shopping carts.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The max inventory size for each producer.
        """
        # Locks to protect shared data structures.
        self.catalogs_lock = RLock()
        self.cartlock = RLock()
        self.customerslock = RLock()
        self.customeridlock = RLock()
        self.loglock = RLock()

        self.producers_catalogs = []
        self.carts = []
        self.queue_size_per_producer = queue_size_per_producer
        self.customers_active = 0
        self.customers_total = 0
        
        # Setup for logging market events.
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            "marketplace.log", backupCount=4, maxBytes=10000000)
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def customers_left(self):
        """Check if there are any active customers in the marketplace."""
        return self.customers_active > 0

    def new_customers(self):
        """Registers a new customer, returning a unique customer ID."""
        with self.customerslock:
            self.customers_total += 1
        return self.customers_total

    def register_producer(self):
        """
        Registers a new producer, creating a dedicated catalog for them.
        
        Returns:
            int: The unique ID for the new producer.
        """
        catalog = Catalog(self.queue_size_per_producer)
        with self.catalogs_lock:
            producer_id = len(self.producers_catalogs)
            self.producers_catalogs.append(catalog)
        self.log(f"{producer_id} = register_producer()")
        return producer_id

    def publish(self, producer_id, product):
        """Allows a producer to add a product to their catalog."""
        catalog = self.producers_catalogs[producer_id]
        ret = catalog.add_product(product)
        self.log(f"{ret} = publish({producer_id}, {product})")
        return ret

    def new_cart(self):
        """Creates a new, empty shopping cart for a customer."""
        with self.customerslock:
            self.customers_active += 1
        with self.cartlock:
            cart_id = len(self.carts)
            self.carts.append([])
        self.log(f"{cart_id} = new_cart()")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a customer's cart.
        
        It iterates through all producer catalogs to find and reserve the product.
        """
        cart = self.carts[cart_id]
        ret = False
        for catalog in self.producers_catalogs:
            ret = catalog.reserve_product(product)
            if ret:
                cart.append((product, catalog))
                break
        self.log(f"{ret} = add_to_cart({cart_id}, {product})")
        return ret

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart, freeing the reservation in the catalog."""
        cart = self.carts[cart_id]
        for (searched_product, catalog) in cart:
            if product == searched_product:
                self.log("{} == {}".format(product, searched_product))
                catalog.free_product(product)
                cart.remove((searched_product, catalog))
                break
        self.log(f"remove_from_cart({cart_id}, {product})")

    def place_order(self, cart_id):
        """Finalizes an order, consuming the reserved products from the cart."""
        product_list = []
        cart = self.carts[cart_id]
        for (product, catalog) in cart:
            catalog.order_product(product)
            product_list.append(product)
        with self.customerslock:
            self.customers_active -= 1
        self.log(f"{product_list} = place_order({cart_id})")
        return product_list

    def log(self, message):
        """Placeholder for logging market events."""
        pass
        
            


from threading import Thread
from time import sleep


class Producer(Thread):
    """A thread that simulates a supplier who produces and publishes products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products the producer can create.
            marketplace (Marketplace): The central marketplace instance.
            republish_wait_time (float): Time to wait before retrying a failed publish.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        Executes the producer's lifecycle.
        
        It registers with the marketplace and then continuously publishes its
        products as long as there are active customers.
        """
        sleep(self.republish_wait_time)
        producer_id = self.marketplace.register_producer()
        while self.marketplace.customers_left():
            for bundle in self.products:
                product = bundle[0]
                quantity = bundle[1]
                wait_time = bundle[2]
                for _ in range(quantity):
                    # Retry publishing until the catalog has space.
                    finished = self.marketplace.publish(producer_id, product)
                    while not finished:
                        sleep(self.republish_wait_time)
                        finished = self.marketplace.publish(producer_id, product)
                    sleep(wait_time)


from dataclasses import dataclass


# Dataclasses provide a clean way to define product types.
@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A generic product with a name and a price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A specialized product, Tea, with a 'type' attribute."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A specialized product, Coffee, with acidity and roast level."""
    acidity: str
    roast_level: str