


from threading import Condition, Semaphore, Lock

class BarieraReentrantaCond(object):
    

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class BarieraReentrantaSem(object):
    

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.prima_faza()
        self.a_doua_faza()

    def prima_faza(self):
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def a_doua_faza(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

from threading import Event, Thread, Lock
from bariera import BarieraReentrantaCond, BarieraReentrantaSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.nr_thread = 0
        self.threaduri = []
        self.vecini = []
        self.lista_loc = []
        self.scripturi = [[] for _ in range(8)]
        self.bar_sync_threaduri = BarieraReentrantaCond(8)
        self.initializare_gata = Event()
        self.timepoint_gata = Event()
        self.bariera_disp = None
    def __str__(self):
        
        return "Dispozitiv %d" % self.device_id

    def setup_devices(self, devices):
        
        maxim_locatii = 0       
        lista_disp = devices
        if self.device_id == 0: 
            self.bariera_disp = BarieraReentrantaSem(len(devices))

            
            for loc in self.sensor_data.keys():
                maxim_locatii = max(maxim_locatii, loc)
            lista_disp.remove(self)     
            
            for dispozitiv in lista_disp:
                for loc in dispozitiv.sensor_data.keys():
                    maxim_locatii = max(maxim_locatii, loc)
                dispozitiv.bariera_disp = self.bariera_disp
            
            for _ in range(maxim_locatii + 1):
                self.lista_loc.append(Lock())
            
            for dispozitiv in lista_disp:
                dispozitiv.lista_loc = self.lista_loc
            
            
            self.initializare_gata.set()
        else:   
            lista_disp.remove(self)             
            for dispozitiv in lista_disp:       
                if dispozitiv.device_id == 0:   
                    dispozitiv.initializare_gata.wait()
                    break

        
        for th_curr in range(8):
            thrd = ThreadDispozitiv(self, th_curr, self.bar_sync_threaduri,
                                    self.bariera_disp)
            self.threaduri.append(thrd)
            self.threaduri[-1].start()


    def assign_script(self, script, zona):
        

        
        
        if script is not None:
            self.scripturi[self.nr_thread].append((script, zona))
            self.nr_thread = (self.nr_thread + 1) % 8
        else:
            self.timepoint_gata.set()

    def get_data(self, zona):
        
        return self.sensor_data[zona] if zona in self.sensor_data else None

    def set_data(self, zona, info):
        
        if zona in self.sensor_data:
            self.sensor_data[zona] = info

    def shutdown(self):
        
        for thrd in self.threaduri:
            thrd.join()


class ThreadDispozitiv(Thread):
    

    def __init__(self, device, nr_thread, bar_sync_threaduri, bar_sync_div):
        
        self.device = device
        self.nr_thread = nr_thread
        self.bar_sync_threaduri = bar_sync_threaduri
        self.bar_sync_div = bar_sync_div


        Thread.__init__(self, name="Dispozitiv %d Thread %d" % (device.device_id, nr_thread))

    def run(self):

        while True:
            
            
            if self.nr_thread == 0:
                self.bar_sync_div.wait()
                self.device.vecini = self.device.supervisor.get_neighbours()

            self.bar_sync_threaduri.wait() 

            if self.device.vecini is None:
                break

            self.device.timepoint_gata.wait() 

            for (script, zona) in self.device.scripturi[self.nr_thread]:
                date_script = []
                
                self.device.lista_loc[zona].acquire()
                for dispozitiv in self.device.vecini:
                    info = dispozitiv.get_data(zona)
                    if info is not None:
                        date_script.append(info)
                
                info = self.device.get_data(zona)
                
                if info is not None:
                    date_script.append(info)

                if date_script != []:
                    
                    result = script.run(date_script)
                    
                    for dispozitiv in self.device.vecini:
                        dispozitiv.set_data(zona, result)
                    
                    self.device.set_data(zona, result)

                self.device.lista_loc[zona].release()
                
            
            self.bar_sync_threaduri.wait()

            if self.nr_thread == 0: 
                self.device.timepoint_gata.clear()
