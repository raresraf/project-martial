"""
This module implements a producer-consumer simulation for a marketplace.

WARNING: This implementation contains critical concurrency flaws and will not
function correctly in a multi-threaded environment. Key issues include:
1.  Improper use of class variables for instance-specific data (e.g., in both
    Marketplace and Consumer), which means all instances share the same state,
    leading to data corruption.
2.  Ineffective locking strategies where locks are acquired and released inside
    loops, failing to protect shared resources against race conditions.
3.  Complete serialization of Consumer threads, defeating the purpose of
    concurrency for consumers.
"""

from threading import Thread, Lock
import time


class Consumer(Thread):
    """
    Intended to represent a consumer thread that purchases products.

    @warning This class is non-functional for concurrency. It uses class
    variables for instance state (`cart_id`, `name`) and a single class lock
    (`my_lock`) that serializes the execution of all consumer threads,
    meaning only one consumer can run at a time.
    """
    cart_id = -1
    name = ''
    my_lock = Lock()

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self)
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        # This correctly assigns to an instance variable, but `name` is also
        # a class variable, which is confusing.
        self.name = kwargs['name']
        

    def run(self):
        """
        Executes the consumer's purchasing logic.
        
        The lock acquired here spans the entire method, preventing any other
        consumer from running concurrently.
        """
        self.my_lock.acquire()
        for i in range(len(self.carts)):
            # `self.cart_id` is a class variable, so all consumers will overwrite it.
            self.cart_id = self.marketplace.new_cart()
            for j in range(len(self.carts[i])):
                if self.carts[i][j]['type'] == 'add':
                    for k in range(self.carts[i][j]['quantity']):
                        verify = False
                        # This loop retries adding a product until successful.
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
            # After "buying" the products, this loop immediately removes them.
            for k in range(len(list_1) - 1, -1, -1):
                print(self.name + ' bought ' + str(list_1[k][0]))
                self.marketplace.remove_from_cart(self.cart_id, list_1[k][0])
        self.my_lock.release()


from threading import Lock


class Marketplace:
    """
    Intended to be the central marketplace for producers and consumers.

    @warning This class is critically flawed. It uses class variables for all
    its state (queues, carts, IDs, locks). In a multi-threaded application,
    this means all threads interact with a single, shared state, which will
    be instantly corrupted. These should be instance variables defined in `__init__`.
    """
    id_producer = 0
    id_cart = 0
    queues = []
    carts = []
    my_Lock1 = Lock()
    my_Lock2 = Lock()
    done = 0

    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """Registers a new producer, adding a new queue for it."""
        self.queues.append([])
        self.id_producer = self.id_producer + 1
        return self.id_producer - 1

    def publish(self, producer_id, product):
        """Allows a producer to publish a product."""
        if len(self.queues[producer_id]) >= self.queue_size_per_producer:
            return False
        # Uses a string "Disponibil" (Available) to track state.
        self.queues[producer_id].append([product, "Disponibil"])
        return True

    def new_cart(self):
        """Creates a new cart for a consumer."""
        self.carts.append([])
        self.id_cart = self.id_cart + 1
        return self.id_cart - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by searching all producer queues.

        @warning The locking strategy here is incorrect. The lock is acquired
        and released inside the loop, creating a window for race conditions where
        multiple threads could try to claim the same product simultaneously.
        """
        verify = 0
        for i in range(len(self.queues)):
            for j in range(len(self.queues[i])):
                self.my_Lock1.acquire() # Incorrectly placed lock
                if product == self.queues[i][j][0] \
                        and self.queues[i][j][1] == 'Disponibil' \
                        and verify == 0:
                    self.carts[cart_id].append([product, i])
                    self.queues[i][j][1] = 'Indisponibil' # Mark as unavailable
                    verify = 1
                    self.my_Lock1.release()
                    break
                self.my_Lock1.release()
                if verify == 1:
                    break
        return verify == 1

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and makes it available again.

        @warning Like `add_to_cart`, this method has an incorrect locking
        strategy that fails to protect against race conditions.
        """
        for i in range(len(self.carts[cart_id])):
            if product == self.carts[cart_id][i][0]:
                producer_id = self.carts[cart_id][i][1]
                for j in range(len(self.queues[producer_id])):
                    self.my_Lock2.acquire() # Incorrectly placed lock
                    if self.queues[producer_id][j][0] == product \
                            and self.queues[producer_id][j][1] == 'Indisponibil':
                        self.queues[producer_id][j][1] = 'Disponibil'
                        self.carts[cart_id].pop(i)
                        self.my_Lock2.release()
                        return True
                    self.my_Lock2.release()
        return False

    def place_order(self, cart_id):
        """Returns the contents of a cart."""
        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    """
    Intended to represent a producer thread that adds products to the marketplace.

    @warning This class uses a class variable for `producer_id`, which is incorrect.
    """
    producer_id = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, daemon=kwargs['daemon'])
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        

    def run(self):
        """Continuously produces and publishes products."""
        # `self.producer_id` is a class variable and will be overwritten by other producers.
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
    """A simple dataclass for a product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
