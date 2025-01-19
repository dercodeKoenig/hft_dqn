from datetime import timedelta
from mt5_tools import Candle

class CandleManager:
    def __init__(self):
        self.tick_data = []  # Store tick data
        self.m1_candles = []  # List of 1-minute candles
        self.s15_candles = []  # List of 15-second candles
        self.current_m1_candle = None  # Current 1-minute candle in progress
        self.current_15s_candle = None  # Current 15-second candle in progress
        self.m1_start_time = None  # Start time for the current 1-minute candle
        self.s15_start_time = None  # Start time for the current 15-second candle

    def set_tick_data(self, ticks):
        """
        Sets the tick data manually for backtesting purposes.
        """
        self.tick_data = ticks

    def _get_candle_start_time(self, tick_time, period_seconds):
        """
        Aligns the tick time to the closest past interval based on the period.
        E.g., for 1 minute, if the tick is at 10:01:02, the candle start time is 10:01:00.
        """
        return tick_time - timedelta(seconds=tick_time.second % period_seconds,
                                     microseconds=tick_time.microsecond)

    def _initialize_candle(self, tick, start_time):
        """
        Initializes a new candle with the given tick.
        The start time is aligned to the nearest candle open time.
        """
        return Candle(o=tick['bid'], h=tick['bid'], l=tick['bid'], c=tick['bid'], t=start_time)

    def _update_candle(self, candle, tick):
        """
        Updates the existing candle with the given tick.
        """
        candle.c = tick['bid']  # Update close price
        candle.h = max(candle.h, tick['bid'])  # Update high price
        candle.l = min(candle.l, tick['bid'])  # Update low price

    def _complete_and_store_candle(self, candle_list, candle):
        """
        Store the completed candle in the appropriate list.
        """
        candle_list.append(candle)

    def _process_tick(self, tick):
        """
        Process an individual tick and update both 15s and 1m candles.
        """
        tick_time = tick['time']
        #print(tick_time)
        # Process 1-minute candle
        if not self.m1_start_time:
            self.m1_start_time = self._get_candle_start_time(tick_time, 60)

        if not self.current_m1_candle:
            self.current_m1_candle = self._initialize_candle(tick, self.m1_start_time)
        else:
            self._update_candle(self.current_m1_candle, tick)

            if tick_time >= self.m1_start_time + timedelta(minutes=1):
                # The 1-minute candle is complete
                self._complete_and_store_candle(self.m1_candles, self.current_m1_candle)
                self.m1_start_time += timedelta(minutes=1)
                self.current_m1_candle = self._initialize_candle(tick, self.m1_start_time)

        # Process 15-second candle
        if not self.s15_start_time:
            self.s15_start_time = self._get_candle_start_time(tick_time, 15)

        if not self.current_15s_candle:
            self.current_15s_candle = self._initialize_candle(tick, self.s15_start_time)
        else:
            self._update_candle(self.current_15s_candle, tick)

            if tick_time >= self.s15_start_time + timedelta(seconds=15):
                # The 15-second candle is complete
                self._complete_and_store_candle(self.s15_candles, self.current_15s_candle)
                self.s15_start_time += timedelta(seconds=15)
                self.current_15s_candle = self._initialize_candle(tick, self.s15_start_time)

    def get_next_15_candle(self):
        """
        Iterate over tick data until the next 15-second candle is completed.
        """
        last_complete_15_candle = None
        previous_complete_15_candle = None

        if len(self.s15_candles) > 0:
            previous_complete_15_candle = self.s15_candles[-1]

        while self.tick_data:
            tick = self.tick_data.pop(0)
            self._process_tick(tick)
            # print("#",end="")

            # Return the last complete 15-second candle
            if len(self.s15_candles) > 0:
                last_complete_15_candle = self.s15_candles[-1]
                if previous_complete_15_candle == None:
                    return last_complete_15_candle

            if last_complete_15_candle and previous_complete_15_candle and last_complete_15_candle.t != previous_complete_15_candle.t:
                return last_complete_15_candle

        return None

    def get_next_m1_candle(self):
        """
        Iterate over tick data until the next 1-minute candle is completed.
        """
        last_complete_m1_candle = None
        previous_complete_m1_candle = None

        if len(self.m1_candles) > 0:
            previous_complete_m1_candle = self.m1_candles[-1]  # Track the last closed 1-minute candle

        while self.tick_data:
            tick = self.tick_data.pop(0)  # Process incoming tick data one by one
            self._process_tick(tick)

            # Check for the last complete 1-minute candle
            if len(self.m1_candles) > 0:
                last_complete_m1_candle = self.m1_candles[-1]

                # If there was no previous candle, return the current one
                if previous_complete_m1_candle is None:
                    return last_complete_m1_candle

            # Return the new candle if its timestamp differs from the previous complete one
            if last_complete_m1_candle and previous_complete_m1_candle and last_complete_m1_candle.t != previous_complete_m1_candle.t:
                return last_complete_m1_candle

        return None  # If no new candle has been completed, return None

    def get_last_15s_candles(self, count=2000):
        """
        Returns the last 200 (or specified count) of complete 15-second candles.
        """
        return self.s15_candles[-count:]

    def get_last_m1_candles(self, count=2000):
        """
        Returns the last 200 (or specified count) of complete 1-minute candles.
        """
        return self.m1_candles[-count:]