
from threading import Thread
import time


class Consumer(Thread):

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        pass

    def run(self):
        id_cos = self.marketplace.new_cart()  
        for lista in self.carts:
            for dictionar in lista:  
                for _ in range(dictionar.get("quantity")):  
                    if dictionar.get("type") == "add":  
                        while not self.marketplace.add_to_cart(id_cos, dictionar.get("product")):
                            time.sleep(self.retry_wait_time)  
                    elif dictionar.get("type") == "remove":  
                        self.marketplace.remove_from_cart(id_cos, dictionar.get("product"))

        lista_comanda = self.marketplace.place_order(id_cos)  
        for value in lista_comanda:
            print(self.name, "bought", value)
        pass
from threading import Lock
import unittest


class Marketplace:

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0  
        self.dictionar_producers = {}  
        self.id_consumer = 0  
        self.dictionar_cos = {}  
        self.publish_lock = Lock()  
        self.add_to_cart_lock = Lock()  
        pass

    def register_producer(self):
        
        self.id_producer = self.id_producer + 1  
        self.dictionar_producers[self.id_producer] = []  
        return self.id_producer
        pass

    def publish(self, producer_id, product):
        
        with self.publish_lock:  
            
            if len(self.dictionar_producers.get(producer_id)) < self.queue_size_per_producer:
                self.dictionar_producers.get(producer_id).append(product)  
                return True
            else:
                return False
        pass

    def new_cart(self):
        
        self.id_consumer = self.id_consumer + 1  
        self.dictionar_cos[self.id_consumer] = []  
        return self.id_consumer
        pass

    def add_to_cart(self, cart_id, product):
        
        ok = False
        key_aux = None
        with self.add_to_cart_lock:  
            for key, values in self.dictionar_producers.items():  
                for value in values:
                    if product == value:  
                        ok = True
                        key_aux = key
                        break
            if ok:
                self.dictionar_producers.get(key_aux).remove(product)  
                self.dictionar_cos.get(cart_id).append([product, key_aux])  
        return ok
        pass

    def remove_from_cart(self, cart_id, product):
        
        for value, id_value in self.dictionar_cos.get(cart_id):  
            if product == value:  
                self.dictionar_cos.get(cart_id).remove([value, id_value])  
                self.dictionar_producers.get(id_value).append(value)  
                break
        pass

    def place_order(self, cart_id):
        
        lista_comanda = []  
        for value, id_value in self.dictionar_cos.get(cart_id):
            lista_comanda.append(value)
        return lista_comanda
        pass


class TestMarketplace(unittest.TestCase):  

    def setUp(self):
        from product import Tea, Coffee
        self.marketplace = Marketplace(15)  
        self.tea = Tea("Lipton", 9, "Green")
        self.coffee = Coffee("Doncafe", 10, "5.05", "MEDIUM")

    def test_register_producer(self):
        id_prod = self.marketplace.register_producer()  
        i = 1
        while i < 10:
            id_prod = self.marketplace.register_producer()  
            i = i + 1
        self.assertEqual(id_prod, 10)  

    def test_publish(self):
        id_prod = self.marketplace.register_producer()  
        
        is_published = self.marketplace.publish(id_prod, self.tea)
        self.assertEqual(is_published, True)  

    def test_new_cart(self):
        id_consumer = self.marketplace.new_cart()  
        i = 1
        while i < 10:
            id_consumer = self.marketplace.new_cart()  
            i = i + 1
        self.assertEqual(id_consumer, 10)  

    def test_add_to_cart(self):
        id_prod = self.marketplace.register_producer()  
        self.marketplace.publish(id_prod, self.coffee)  
        id_consumer = self.marketplace.new_cart()  
        is_added_to_cart_coffee = self.marketplace.add_to_cart(id_consumer, self.coffee)  
        self.assertEqual(is_added_to_cart_coffee, True)  
        is_added_to_cart_tea = self.marketplace.add_to_cart(id_consumer, self.tea)
        
        self.assertEqual(is_added_to_cart_tea, False)

    def test_remove_from_cart(self):
        id_consumer = self.marketplace.new_cart()  
        id_prod1 = self.marketplace.register_producer()  
        self.marketplace.publish(id_prod1, self.coffee)
        id_prod2 = self.marketplace.register_producer()  
        self.marketplace.publish(id_prod2, self.tea)
        self.marketplace.add_to_cart(id_consumer, self.coffee)  
        self.marketplace.add_to_cart(id_consumer, self.tea)  
        self.marketplace.remove_from_cart(id_consumer, self.coffee)  
        for key, values in self.marketplace.dictionar_cos.items():
            for value in values:
                produs = value[0]
                self.assertEqual(produs, self.tea)  
                break

    def test_place_order(self):
        id_consumer = self.marketplace.new_cart()
        id_prod1 = self.marketplace.register_producer()
        self.marketplace.publish(id_prod1, self.coffee)
        id_prod2 = self.marketplace.register_producer()
        self.marketplace.publish(id_prod2, self.tea)
        self.marketplace.add_to_cart(id_consumer, self.coffee)
        self.marketplace.add_to_cart(id_consumer, self.tea)  
        lista_creata = self.marketplace.place_order(id_consumer)  
        lista = [self.coffee, self.tea]  
        self.assertEqual(lista_creata, lista)
from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        pass

    def run(self):
        id_producer = self.marketplace.register_producer()  
        while True:
            
            for value in self.products:
                for _ in range(value[1]):  
                    if self.marketplace.publish(id_producer, value[0]):
                        time.sleep(value[2])  
                    else:
                        time.sleep(self.republish_wait_time)  
        pass


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    
    acidity: str
    roast_level: str
