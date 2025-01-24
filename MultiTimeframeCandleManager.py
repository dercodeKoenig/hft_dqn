
from collections import deque
import copy
from Candle import Candle
from collections import deque

class MultiTimeframeCandleManager:
    
    def __init__(self):
        l = 60
        self.m1_candles = deque(maxlen = l)
        self.m5_candles = deque(maxlen = l)
        self.m15_candles = deque(maxlen = l)
        self.m60_candles = deque(maxlen = 1000)  
        self.d1_candles = deque(maxlen = 1000)   


        self.midnight_open = 0
        self.midnight_opening_range_low = 0
        self.midnight_opening_range_high = 0
    
        self.new_midnight_opening_range_low = 0
        self.new_midnight_opening_range_high = 0
    
        self.opening_range_gaps = deque(maxlen = 3)
        self.ndogs = deque(maxlen = 5)
        self.fps = deque(maxlen = 3)
        
        
        self.london_highs_lows = deque(maxlen = 3)
        self.current_lh = 0;
        self.current_ll = 0;

        self.asia_highs_lows = deque(maxlen = 3)
        self.current_ah = 0;
        self.current_al = 0;

        self.ny_am_highs_lows = deque(maxlen = 3)
        self.current_nah = 0;
        self.current_nal = 0;

        self.ny_lunch_highs_lows = deque(maxlen = 3)
        self.current_nlh = 0;
        self.current_nll = 0;

        self.ny_pm_highs_lows = deque(maxlen = 3)
        self.current_nph = 0;
        self.current_npl = 0;

        
            
        self.last_settlement_price = 0
    
        self.normal_pdas = deque(maxlen = 120)
    
        self.sod = 0
        self.sod_open = 0
        self.sod_max_b = 0
        self.sod_max_w = 0
        self.sod_min_b = 0
        self.sod_min_w = 0
        self.sod_s = 0
    
        self.hunt_fvg = False

    

    
    def push_m1_candle(self, __candle):
        candle = copy.deepcopy(__candle)
        candle_minute = candle.t.minute
        candle_hour = candle.t.hour

        self.m1_candles.append(candle)
    
        if(candle.t.hour == 0 and candle.t.minute == 0):
            self.midnight_open  = candle.o
            self.new_midnight_opening_range_low = candle.l
            self.new_midnight_opening_range_high = candle.h
    
        if(candle.t.hour == 0 and candle.t.minute < 30):
            self.new_midnight_opening_range_low = min(self.new_midnight_opening_range_low, candle.l)
            self.new_midnight_opening_range_high = max(self.new_midnight_opening_range_high, candle.h)
        
        if(candle.t.hour == 0 and candle.t.minute == 30):
            self.midnight_opening_range_low = self.new_midnight_opening_range_low
            self.midnight_opening_range_high = self.new_midnight_opening_range_high
    
        if(candle.t.hour == 16 and candle.t.minute == 15 and len(self.m1_candles)>1):
            last_candle = self.m1_candles[-2]
            self.last_settlement_price = last_candle.c
    
        if(candle.t.hour == 9 and candle.t.minute == 30 and candle.t.second == 0):
            opening_price = candle.o
            self.hunt_fvg = True
            self.opening_range_gaps.append([self.last_settlement_price, (opening_price+self.last_settlement_price) / 2, opening_price])

        if self.hunt_fvg and candle.t.hour == 9 and candle.t.minute >= 32:
            new_fvg = [0,0,0]
            if self.m1_candles[-1].l > self.m1_candles[-3].h:
                new_fvg = [self.m1_candles[-3].h, (self.m1_candles[-3].h+self.m1_candles[-1].l) / 2 ,self.m1_candles[-1].l]
            if self.m1_candles[-1].h < self.m1_candles[-3].l:
                new_fvg = [self.m1_candles[-3].l, (self.m1_candles[-3].l+self.m1_candles[-1].h) / 2 ,self.m1_candles[-1].h]

            if(new_fvg[1] != 0):
                self.hunt_fvg = False
                self.fps.append(new_fvg)
                

        if(candle.t.hour == 2 and candle.t.minute == 0):
            self.current_lh = candle.h
            self.current_ll = candle.l
            
        if(candle.t.hour >= 2 and candle.t.hour < 5):
            self.current_lh = max(candle.h, self.current_lh)
            self.current_ll = min(candle.l, self.current_ll)
        
        if(candle.t.hour == 2 and candle.t.minute == 0):
            self.london_highs_lows.append([self.current_lh, 0, self.current_ll, 0])

        
        if(candle.t.hour == 20 and candle.t.minute == 0):
            self.current_h = candle.h
            self.current_al = candle.l
            
        if(candle.t.hour >= 20):
            self.current_ah = max(candle.h, self.current_ah)
            self.current_al = min(candle.l, self.current_al)
        
        if(candle.t.hour == 0 and candle.t.minute == 0):
            self.asia_highs_lows.append([self.current_ah, 0, self.current_al, 0])

        
        if(candle.t.hour == 7 and candle.t.minute == 0):
            self.current_nah = candle.h
            self.current_nal = candle.l
            
        if(candle.t.hour >= 7 and candle.t.hour < 11):
            self.current_nah = max(candle.h, self.current_nah)
            self.current_nal = min(candle.l, self.current_nal)
        
        if(candle.t.hour == 11 and candle.t.minute == 0):
            self.ny_am_highs_lows.append([self.current_nah, 0, self.current_nal, 0])

            
        if(candle.t.hour == 11 and candle.t.minute == 0):
            self.current_nlh = candle.h
            self.current_nll = candle.l
            
        if(candle.t.hour >= 11 and candle.t.hour < 13):
            self.current_nlh = max(candle.h, self.current_nlh)
            self.current_nll = min(candle.l, self.current_nll)
        
        if(candle.t.hour == 13 and candle.t.minute == 0):
            self.ny_lunch_highs_lows.append([self.current_nlh, 0, self.current_nll, 0])

        
        if(candle.t.hour == 13 and candle.t.minute == 0):
            self.current_nph = candle.h
            self.current_npl = candle.l
            
        if(candle.t.hour >= 13 and candle.t.hour < 16):
            self.current_nph = max(candle.h, self.current_nph)
            self.current_npl = min(candle.l, self.current_npl)
        
        if(candle.t.hour == 16 and candle.t.minute == 0):
            self.ny_pm_highs_lows.append([self.current_nph, 0, self.current_npl, 0])

            

        for n in range(len(self.london_highs_lows)):
            if candle.h > self.london_highs_lows[n][0]:
                self.london_highs_lows[n][1] = 1
            if candle.l < self.london_highs_lows[n][2]:
                self.london_highs_lows[n][3] = 1
        
        
        for n in range(len(self.asia_highs_lows)):
            if candle.h > self.asia_highs_lows[n][0]:
                self.asia_highs_lows[n][1] = 1
            if candle.l < self.asia_highs_lows[n][2]:
                self.asia_highs_lows[n][3] = 1

                
        for n in range(len(self.ny_am_highs_lows)):
            if candle.h > self.ny_am_highs_lows[n][0]:
                self.ny_am_highs_lows[n][1] = 1
            if candle.l < self.ny_am_highs_lows[n][2]:
                self.ny_am_highs_lows[n][3] = 1

                
        for n in range(len(self.ny_lunch_highs_lows)):
            if candle.h > self.ny_lunch_highs_lows[n][0]:
                self.ny_lunch_highs_lows[n][1] = 1
            if candle.l < self.ny_lunch_highs_lows[n][2]:
                self.ny_lunch_highs_lows[n][3] = 1

                
        for n in range(len(self.ny_pm_highs_lows)):
            if candle.h > self.ny_pm_highs_lows[n][0]:
                self.ny_pm_highs_lows[n][1] = 1
            if candle.l < self.ny_pm_highs_lows[n][2]:
                self.ny_pm_highs_lows[n][3] = 1


        ##### other pd arrays
        #def center(a,b):
            #return (a+b) / 2
        ### normal fvg
        if(len(self.m1_candles) > 3):
            if self.m1_candles[-1].l > self.m1_candles[-3].h:
                fvg = [self.m1_candles[-3].h, self.m1_candles[-1].l, 0]
                self.normal_pdas.append(fvg)
                
            if self.m1_candles[-1].h < self.m1_candles[-3].l:
                fvg = [self.m1_candles[-1].h, self.m1_candles[-3].l, 0]
                self.normal_pdas.append(fvg)

        ### normal ob
        if self.m1_candles[-1].c < self.m1_candles[-1].o:
            if self.sod == -1:
                self.sod_max_b = max(self.sod_max_b, self.m1_candles[-1].o)
                self.sod_max_w =  max(self.sod_max_w, self.m1_candles[-1].h)
                self.sod_min_b = min(self.sod_min_b, self.m1_candles[-1].c)
                self.sod_min_w =  max(self.sod_min_w, self.m1_candles[-1].l)
                self.sod_s += 1
            if self.sod == 1:
                ob = [self.sod_min_w, self.sod_max_w, 0]
                self.normal_pdas.append(ob)

            if self.sod != -1:
                self.sod = -1
                self.sod_max_b = self.m1_candles[-1].o
                self.sod_max_w =  self.m1_candles[-1].h
                self.sod_min_b = self.m1_candles[-1].c
                self.sod_min_w =  self.m1_candles[-1].l
                self.sod_s = 1
                
        if self.m1_candles[-1].c > self.m1_candles[-1].o:
            if self.sod == 1:
                self.sod_max_b = max(self.sod_max_b, self.m1_candles[-1].c)
                self.sod_max_w =  max(self.sod_max_w, self.m1_candles[-1].h)
                self.sod_min_b = min(self.sod_min_b, self.m1_candles[-1].o)
                self.sod_min_w =  max(self.sod_min_w, self.m1_candles[-1].l)
                self.sod_s += 1
            if self.sod == -1:
                ob = [self.sod_min_w, self.sod_max_w, 0]
                self.normal_pdas.append(ob)

            if self.sod != 1:
                self.sod = 1
                self.sod_max_b = self.m1_candles[-1].c
                self.sod_max_w =  self.m1_candles[-1].h
                self.sod_min_b = self.m1_candles[-1].o
                self.sod_min_w =  self.m1_candles[-1].l
                self.sod_s = 1
        #####
        for n in range(len(self.normal_pdas)):
            self.normal_pdas[n][2]+=1

        
        while len(self.normal_pdas) > 0 and self.normal_pdas[0][2] > 20:
                del self.normal_pdas[0]
    

        # m5 candles
        if len(self.m5_candles) > 0:
            last_minute = self.m5_candles[-1].t.minute
            last_hour = self.m5_candles[-1].t.hour
            minute_rounded = int(candle_minute/5) * 5
            if minute_rounded != last_minute or candle_hour != last_hour:
                
                c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m5_candles.append(c)
            else:
                self.m5_candles[-1].c = candle.c
                self.m5_candles[-1].h = max(candle.h, self.m5_candles[-1].h)
                self.m5_candles[-1].l = min(candle.l, self.m5_candles[-1].l)
        else:

            c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m5_candles.append(c)
            

       # m15 candles
        if len(self.m15_candles) > 0:
            last_minute = self.m15_candles[-1].t.minute
            last_hour = self.m15_candles[-1].t.hour
            minute_rounded = int(candle_minute/15) * 15
            if minute_rounded != last_minute or candle_hour != last_hour:
               
                c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m15_candles.append(c)
            else:
                self.m15_candles[-1].c = candle.c
                self.m15_candles[-1].h = max(candle.h, self.m15_candles[-1].h)
                self.m15_candles[-1].l = min(candle.l, self.m15_candles[-1].l)
        else:
            
            c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m15_candles.append(c)

        # m60 candles
        if len(self.m60_candles) > 0:
            last_minute = self.m60_candles[-1].t.minute
            last_hour = self.m60_candles[-1].t.hour
            
            if candle_hour != last_hour:
                
                c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m60_candles.append(c)
            else:
                self.m60_candles[-1].c = candle.c
                self.m60_candles[-1].h = max(candle.h, self.m60_candles[-1].h)
                self.m60_candles[-1].l = min(candle.l, self.m60_candles[-1].l)
        else:
           
            c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m60_candles.append(c)


       # d1 candles
        if len(self.d1_candles) > 0:
            last_candle_hour = self.m1_candles[-2].t.hour
            #print(candle_hour)
            if candle_hour != last_candle_hour and candle_hour == 18:
                
                c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.d1_candles.append(c)

                last_close = self.m1_candles[-2].c
                new_open = candle.o
                ce = (new_open+last_close) / 2
                self.ndogs.append([last_close, ce, new_open])
                
                
            else:
                self.d1_candles[-1].c = candle.c
                self.d1_candles[-1].h = max(candle.h, self.d1_candles[-1].h)
                self.d1_candles[-1].l = min(candle.l, self.d1_candles[-1].l)
        else:
            
            c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.d1_candles.append(c)


        pdas = [
           
            ]
        for g in self.opening_range_gaps:
            pdas.extend(g)
        for g in self.fps:
            pdas.extend(g)
        for g in self.ndogs:
            pdas.extend(g)

        for hl in self.asia_highs_lows:
            #print("a:", hl)
            pdas.extend(hl)
        for hl in self.london_highs_lows:
            pdas.extend(hl)
            #print("l:", hl)
        for hl in self.ny_am_highs_lows:
            pdas.extend(hl)
            #print("na:", hl)
        for hl in self.ny_lunch_highs_lows:
            pdas.extend(hl)
            #print("nl:", hl)
        for hl in self.ny_pm_highs_lows:
            pdas.extend(hl)
            #print("np:", hl)
            

        ret_candles =       [
                self.m15_candles,
                 self.m5_candles,
                 self.m1_candles
        ]
        
        return [
            self.midnight_open, 
            self.midnight_opening_range_high,
            self.midnight_opening_range_low, 
            pdas,
    
            self.m1_candles[-1].c, 
            self.m1_candles[-1].t,
            
            ret_candles
        ]


