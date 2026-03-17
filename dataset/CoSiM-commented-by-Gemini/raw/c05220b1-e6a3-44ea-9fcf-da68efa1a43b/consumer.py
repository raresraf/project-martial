"""
This module contains a multi-threaded producer-consumer simulation for a marketplace.

It includes `Producer`, `Consumer`, and a central `Marketplace` class. Products
are defined using `dataclass`.

NOTE: This implementation has several severe design and concurrency flaws:
- The `Consumer` class uses a lock that serializes the execution of all consumer
  threads, defeating the purpose of multi-threading.
- The `Marketplace` uses class-level variables, acting as a Singleton, but fails
  to lock critical sections in `register_producer` and `new_cart`, creating
  race conditions.
- Locking in `add_to_cart` and `remove_from_cart` is implemented incorrectly,
  failing to protect the entire operation and leaving it vulnerable to races.
- The `place_order` logic is flawed; purchased items are immediately returned
  to stock by the consumer.
"""

from threading import Thread, Lock
import time


class Consumer(Thread):
    """
    Simulates a consumer that processes a list of shopping carts.

    SEVERE FLAW: Uses a class-level lock (`my_lock`) around the entire run
    method, which means only one consumer instance can run at a time,
    completely nullifying any concurrency.
    """
    cart_id = -1
    name = ''
    my_lock = Lock()

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.
        """

        Thread.__init__(self)
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.name = kwargs['name']
        

    def run(self):
        # This lock prevents any other Consumer thread from running concurrently.
        self.my_lock.acquire()
        for i in range(len(self.carts)):
            self.cart_id = self.marketplace.new_cart()
            for j in range(len(self.carts[i])):
                # Block Logic: Add or remove items based on the cart's instructions.
                if self.carts[i][j]['type'] == 'add':
                    for k in range(self.carts[i][j]['quantity']):
                        verify = False
                        # Polling loop to retry adding an item until it succeeds.
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

            # FLAW: After "buying" the products, this loop immediately returns
            # them to the marketplace stock by calling `remove_from_cart`.
            for k in range(len(list_1) - 1, -1, -1):
                print(self.name + ' bought ' + str(list_1[k][0]))
                self.marketplace.remove_from_cart(self.cart_id, list_1[k][0])
        self.my_lock.release()


from threading import Lock


class Marketplace:
    """
    A Singleton marketplace using class-level variables for shared state.

    FLAW: This class has multiple race conditions due to missing or
    improperly used locks.
    """
    id_producer = 0
    id_cart = 0
    queues = []
    carts = []
    my_Lock1 = Lock()
    my_Lock2 = Lock()
    done = 0

    def __init__(self, queue_size_per_producer):
        """Initializes a marketplace instance."""
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        Registers a new producer.

        RACE CONDITION: This method modifies shared lists and counters
        (`queues`, `id_producer`) without any locking, making it not thread-safe.
        """
        self.queues.append([])
        self.id_producer = self.id_producer + 1
        return self.id_producer - 1

    def publish(self, producer_id, product):
        """
        Adds a product to a producer's queue if there is capacity.

        Instead of removing items, it marks them as 'Disponibil' (Available)
        or 'Indisponibil' (Unavailable).
        """
        if len(self.queues[producer_id]) >= self.queue_size_per_producer:
            return False
        self.queues[producer_id].append([product, "Disponibil"])
        return True

    def new_cart(self):
        """
        Creates a new cart.

        RACE CONDITION: This method modifies shared state (`carts`, `id_cart`)
        without any locking.
        """
        self.carts.append([])
        self.id_cart = self.id_cart + 1
        return self.id_cart - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by marking it as 'Indisponibil'.

        RACE CONDITION: The lock is acquired *inside* the loop. Two threads can
        check the same item's availability before one has a chance to lock and
        modify it, leading to the same item being added to multiple carts.
        """
        verify = 0
        for i in range(len(self.queues)):
            for j in range(len(self.queues[i])):
                self.my_Lock1.acquire()
                if product == self.queues[i][j][0] 
                        and self.queues[i][j][1] == 'Disponibil' 
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
        Removes a product from a cart by marking it as 'Disponibil'.

        RACE CONDITION: Locking is again done incorrectly inside the loop,
        making the operation non-atomic.
        """
        for i in range(len(self.carts[cart_id])):
            if product == self.carts[cart_id][i][0]:
                for j in range(len(self.queues[self.carts[cart_id][i][1]])):
                    self.my_Lock2.acquire()
                    if self.queues[self.carts[cart_id][i][1]][j][0] == product 
                            and self.queues[self.carts[cart_id][i][1]][j][1] == 'Indisponibil':
                        self.queues[self.carts[cart_id][i][1]][j][1] = 'Disponibil'
                        self.carts[cart_id].remove(self.carts[cart_id][i])
                        self.my_Lock2.release()
                        return True
                    self.my_Lock2.release()
        return False

    def place_order(self, cart_id):
        """
        Returns the contents of a cart.

        FLAW: This method does not clear the cart's contents after returning them.
        The consumer is responsible for manually emptying the cart.
        """
        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    """
    A thread that simulates a producer creating and publishing products.
    """
    producer_id = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread."""
        Thread.__init__(self, daemon=kwargs['daemon'])
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        

    def run(self):
        """
        Main execution loop: registers and then continuously produces items.
        """
        self.producer_id = self.marketplace.register_producer()
        while True:
            for i in range(len(self.products)):
                for j in range(self.products[i][1]):
                    # Polling loop to retry publishing until it succeeds.
                    verify = self.marketplace.publish(self.producer_id, self.products[i][0])
                    time.sleep(self.products[i][2]) # Simulate production time
                    if not verify:
                        time.sleep(self.republish_wait_time)
                        break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple, immutable data class for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """An immutable data class for a Tea product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """An immutable data class for a Coffee product."""
    acidity: str
    roast_level: str
