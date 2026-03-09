"""
This module contains a producer-consumer simulation, including classes for
`Consumer`, `Marketplace`, `Producer`, and `Product` data types.

WARNING: This implementation has critical thread-safety issues. The `register_producer`,
`new_cart`, and `remove_from_cart` methods in the Marketplace lack the necessary
locks, which will lead to race conditions under concurrency. The `add_to_cart`
method is also very inefficient.
"""
from threading import Thread, Lock
from time import sleep
from dataclasses import dataclass


class Consumer(Thread):
    """
    A thread that simulates a consumer purchasing products from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        Processes a list of shopping trips, creating a new cart for each one.
        It retries adding items if they are not available.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for opp in cart:
                for i in range(0, opp["quantity"]):
                    # Invariant: Loop until the item is successfully added.
                    if opp["type"] == "add":
                        # Pre-condition: Try to add a product. If it fails, wait and retry.
                        while self.marketplace.add_to_cart(cart_id, opp["product"]) is False:
                            sleep(self.retry_wait_time)
                    elif opp["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, opp["product"])

            prod_list = self.marketplace.place_order(cart_id)

            for product in prod_list:
                print(str(self.name) + " bought " + str(product))


class Marketplace:
    """
    Manages product inventory and shopping carts.

    This implementation is NOT thread-safe and has significant performance issues.
    """

    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.consumer_id = 0
        # Maps producer_id to a list of products they have published.
        self.prod_dict = {}
        # Maps cart_id to a list of [product, producer_id] pairs.
        self.cart_dict = {}
        self.lock_add_cart = Lock()
        self.lock_publish = Lock()
        pass

    def register_producer(self):
        """
        Registers a new producer.

        WARNING: This method is NOT thread-safe. Two threads calling this at the
        same time could receive the same producer_id.
        """
        self.producer_id += 1
        self.prod_dict[self.producer_id] = []
        return self.producer_id
        pass

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.
        This operation is thread-safe.
        """
        self.lock_publish.acquire()
        # Pre-condition: Check if the producer's queue has space.
        if len(self.prod_dict[producer_id]) < self.queue_size_per_producer:
            self.prod_dict[producer_id].append(product)
            self.lock_publish.release()
            return True
        self.lock_publish.release()
        return False
        pass

    def new_cart(self):
        """
        Creates a new cart for a consumer.

        WARNING: This method is NOT thread-safe. Two threads calling this at the
        same time could receive the same cart_id.
        """
        self.consumer_id += 1
        self.cart_dict[self.consumer_id] = []
        return self.consumer_id
        pass

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        This method is inefficient because it performs a nested loop search through
        all products from all producers to find an available item.
        """
        self.lock_add_cart.acquire()

        # Inefficient Search: Iterates through all producers and all their products.
        for prod_id in self.prod_dict.keys():
            for p in self.prod_dict[prod_id]:
                if p == product:
                    # If found, move the product from the producer's stock to the consumer's cart.
                    self.prod_dict[prod_id].remove(product)
                    self.cart_dict[cart_id].append([product, prod_id])
                    self.lock_add_cart.release()
                    return True
        self.lock_add_cart.release()

        return False
        pass

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer.

        WARNING: This method is NOT thread-safe. It does not use any locks,
        leading to race conditions if multiple consumers remove items concurrently.
        """
        for prod in self.cart_dict[cart_id]:
            if prod[0] == product:
                self.cart_dict[cart_id].remove(prod)
                # Return the product to the original producer's list.
                self.prod_dict[prod[1]].append(prod[0])
                break

    def place_order(self, cart_id):
        """Returns the list of products in the cart, confirming the order."""
        prod_list = []
        for prod in self.cart_dict[cart_id]:
           prod_list.append(prod[0])
        return prod_list


class Producer(Thread):
    """
    A thread that simulates a producer creating and publishing products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs

    def run(self):
        """
        Continuously produces and publishes products, waiting and retrying if the
        producer's queue in the marketplace is full.
        """
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                sleep(product[2])
                for i in range(0, product[1]):
                    # Invariant: Loop until the product is successfully published.
                    while self.marketplace.publish(producer_id, product[0]) is False:
                        sleep(self.republish_wait_time)


# --- Data models for products ---
@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A tea product with an additional 'type' attribute."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A coffee product with acidity and roast level attributes."""
    acidity: str
    roast_level: str