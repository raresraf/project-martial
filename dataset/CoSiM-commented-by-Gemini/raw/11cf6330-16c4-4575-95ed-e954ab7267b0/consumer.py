"""
This module simulates a marketplace with producers, consumers, and various
product types using a multi-threaded approach.

It defines classes for:
- Consumer: A thread that simulates a customer adding and removing items from a
  shopping cart and placing an order.
- Marketplace: The central class intended to manage the inventory and cart
  transactions in a shared environment.
- Producer: A thread that simulates a producer publishing products to the marketplace.
- Product, Tea, Coffee: Dataclasses for representing products.

Warning: The Marketplace class has an inconsistent, complex, and buggy locking
strategy. Some methods are not thread-safe at all, while others have confusing
lock management, leading to race conditions and unpredictable behavior.
"""

import time
from threading import Thread, RLock
from dataclasses import dataclass

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution loop for the consumer."""
        for i in range(len(self.carts)):
            cart_id = self.marketplace.new_cart()

            for cart in self.carts[i]:
                # Adds or removes a product based on the cart command.
                if cart['type'] == 'add':
                    for _ in range(cart['quantity']):
                        # Retries adding to the cart until successful.
                        while not self.marketplace.add_to_cart(cart_id, cart['product']):
                            time.sleep(self.retry_wait_time)
                elif cart['type'] == 'remove':
                    for _ in range(cart['quantity']):
                        self.marketplace.remove_from_cart(cart_id, cart['product'])
            
            # Places the order and prints the results.
            products_to_buy = self.marketplace.place_order(cart_id)
            for item in products_to_buy:
                print(str(self.kwargs['name']) + " bought " + str(item))
            self.marketplace.clean_cart(cart_id)

class Marketplace:
    """
    Manages inventory and transactions between producers and consumers.

    @warning This class is NOT thread-safe. Locking is applied inconsistently.
    Methods like `publish` have no locking, while `add_to_cart` and `remove_from_cart`
    have confusing locking patterns that still allow for race conditions.
    """

    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        # A single re-entrant lock for the entire marketplace.
        self.lock = RLock()
        # Shared state that is modified concurrently.
        self.producers = []
        self.carts = []
        self.counter = 0
        self.consumer_id = 0

    def register_producer(self):
        """Registers a new producer. This method is thread-safe."""
        with self.lock:
            producer_dict = dict()
            queue = list()
            self.counter += 1
            producer_dict['id'] = self.counter
            producer_dict['published_products'] = queue
            self.producers.append(producer_dict)
            return producer_dict['id']

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace.

        @warning Not Thread-Safe: This method modifies the shared `self.producers`
        list of lists without acquiring any lock, which is a major race condition.
        """
        for producer in self.producers:
            if producer['id'] == producer_id:
                if len(producer['published_products']) < self.queue_size_per_producer:
                    # The boolean flag seems to indicate if the product is available.
                    producer['published_products'].append([product, True])
                    return True
                return False
        return False

    def new_cart(self):
        """Creates a new cart for a consumer. This method is thread-safe."""
        with self.lock:
            cart_dict = dict()
            products_in_cart = list()
            self.consumer_id += 1
            cart_dict['id'] = self.consumer_id
            cart_dict['products_in_cart'] = products_in_cart
            self.carts.append(cart_dict)
            return cart_dict['id']

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.

        @warning Race Condition: This method iterates over `self.carts` *before*
        acquiring a lock. Another thread could modify the list during this
        iteration. The locking logic is also complex and hard to reason about.
        """
        for cart in self.carts:
            if cart['id'] == cart_id:
                with self.lock:
                    for producer in self.producers:
                        for published_product in producer['published_products']:
                            if published_product[0][0] == product and published_product[1]:
                                cart['products_in_cart'].append(product)
                                # Marks the product as "in cart" / unavailable.
                                published_product[1] = False
                                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart.

        @warning Race Condition: Similar to `add_to_cart`, this method iterates
        over `self.carts` before acquiring a lock. The locking logic is also complex.
        """
        for cart in self.carts:
            if cart['id'] == cart_id:
                with self.lock:
                    for product_in_cart in cart['products_in_cart']:
                        if product_in_cart == product:
                            cart['products_in_cart'].remove(product_in_cart)
                            # Makes the product available again.
                            for producer in self.producers:
                                for published_product in producer['published_products']:
                                    if published_product[0][0] == product and not published_product[1]:
                                        published_product[1] = True
                                        return None # Exits within the lock
        return None

    def place_order(self, cart_id):
        """
        Finalizes an order by 'removing' items from producer stock.

        @warning Dangerous Logic: This method modifies `producer['published_products']`
        while iterating over it (implicitly via the nested loops). This can lead
        to unpredictable behavior. The logic is very complex and likely buggy.
        """
        for cart in self.carts:
            if cart['id'] == cart_id:
                with self.lock:
                    index = 0
                    while index < len(cart['products_in_cart']):
                        for producer in self.producers:
                            for product in producer['published_products']:
                                if product[0][0] == cart['products_in_cart'][index] and not product[1]:
                                    producer['published_products'].remove(product)
                                    index += 1
                                    break
                            if index == len(cart['products_in_cart']):
                                return cart['products_in_cart']
                    return cart['products_in_cart']
        return []

    def clean_cart(self, cart_id):
        """Removes all products from a cart list."""
        for cart in self.carts:
            if cart['id'] == cart_id:
                cart['products_in_cart'].clear()


class Producer(Thread):
    """Represents a producer that supplies products to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.daemon = True
        self.is_published = False
        self.kwargs = kwargs

    def run(self):
        """The main execution loop for the producer."""
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    # Retries publishing until successful.
                    self.is_published = self.marketplace.publish(producer_id, product)
                    while not self.is_published:
                        self.is_published = self.marketplace.publish(producer_id, product)
                        time.sleep(self.republish_wait_time)
                    time.sleep(product[2])


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A dataclass for representing a generic product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for representing a Tea product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for representing a Coffee product."""
    acidity: str
    roast_level: str
