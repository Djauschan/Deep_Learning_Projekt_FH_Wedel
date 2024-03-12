from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from numpy import ndarray

from src.preprocessing.services.TimeModificationService import TimeModificationService


class TimeSeriesBuilder:

    def __init__(self):
        pass

    # todo fix singel source of trutuh similar process, in buildTimeSeries but not reuseable bc label identification and
    # todo mutliple iteration, no time to dix
    def buildSingleTimeSeries(
            self,
            df: pd.DataFrame,
            features,
            length: int,
            interval: int,
            tolerance: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
            build a single TimeSeries of a dataframe,
            the timeseries has as the last element of the series the last element of the dataframe
            as the dataframe is already filtered to only contain the elements within the desired range
            :param:

            :return:
                tupel np.array with (length_series, features), np.array(dateTime entries of series)
        """
        len_df = len(df) - 1
        x = length - 1 #runner idx, items in series
        RETRY_TRESHOLD = 10
        previousCandidateIdx = 0
        dateTimeArr = np.zeros(length+1).astype(datetime)
        singleTimeSeriesArr = np.zeros((length+1, len(features)))
        posixTestArr = np.zeros(length)
        isValid = True
        df_row = df.iloc[len_df]  # put in last element of series
        singleTimeSeriesArr[length] = df_row[[*features]].values
        dateTimeArr[length] = df_row['DateTime']
        previousCandidate = df_row
        while x > -1 and isValid:
            # from current element the posix minute
            currentMinute = previousCandidate['posixMinute']
            # the next element in the series should be the item with the posixTime of last ele - the interval
            nextOptimalMinute = currentMinute - interval
            # get closest element to optimal element, with binary search, start looking at previous ele bc list is sorted
            candidateIdx, minimalAbw, closestCandidateIdx = \
                self._binary_search(df, 0, len_df, nextOptimalMinute, tolerance, -1, False)
            # no exact candidate found
            if candidateIdx == -1:
                # is the next candidate in tolerance range
                if abs(minimalAbw) < tolerance:
                    candidateIdx = closestCandidateIdx
                # go to previous day
                else:
                    # ersten Wert des folge Tages nehmen (oder nächster Handelstag...)
                    reTryCounter: int = 1
                    noValidAlternativeFound = True
                    while reTryCounter < RETRY_TRESHOLD and noValidAlternativeFound:
                        optimalDateTimeFromPosix = pd.Timestamp.fromtimestamp(nextOptimalMinute * 60) - \
                                                   pd.Timedelta(hours=1)
                        prevDay = TimeModificationService.getPreviousDayVal(optimalDateTimeFromPosix, reTryCounter)
                        candidateIdx, minimalAbw, closestCandidateIdx = \
                            self._binary_search_dateTime(df=df, lowIdx=0, highIdx=len_df,
                                                         optimalDateToFind=prevDay, minimal_abw=100000, closestIdx=-1,
                                                         closestVal_isBigger=True)
                        reTryCounter += 1
                        if candidateIdx != -1:
                            noValidAlternativeFound = False
                        # in seconds = 180second = 3min is allowed on next day
                        if not noValidAlternativeFound and abs(minimalAbw) <= 600:
                            candidateIdx = closestCandidateIdx
                            noValidAlternativeFound = False
                            # wenn bis 8:20 gefunden -> zeitreihe verwerfen
            if candidateIdx == -1:
                isValid = False
                print(' NO VALID CANDIDATE ')
            else:
                'Bei pd.TimeStamp = 16.01.2024 = 2024-01-16 bei data 2. day'
                candidate = df.iloc[candidateIdx]
                previousCandidate = candidate
                previousCandidateIdx = candidateIdx
                candidate_minuteOfDay = int(candidate['posixMinute'])
                ### END FIND CANDIDATE
                # Second check if the day fits as well
                singleTimeSeriesArr[x] = candidate[[*features]].values
                dateTimeArr[x] = candidate['DateTime']
                posixTestArr[x] = candidate_minuteOfDay
                x = x - 1

        return singleTimeSeriesArr, dateTimeArr

    def buildTimeSeries(self, data: pd.DataFrame, length: int, interval: int, time_ahead: int, tolerance: int,
                        nextDay_retry_threshold: int, featureArray: list, label_name: str):
        """
            Creates timeseriesArray, with constant time intervals
            This function is supposed to be modular, each feature will be return in a seperate numpy arr,
            so that it can be serialized individually. This enables that different combinations of features can
            be explored without redoing the building process

            @:param
                _df = represents the dataframe, with spaces and emptyRows..
                length = the desired length of a single timeSeries
                interval = the space <=> interval between single timeSeries
                tolerance = the allowed difference between the intervals
                time_ahead = the amount of time <=> timestamp, the model trys to predict <=> timeStamp of Label
                    time_ahead is provided as amount of intervals ahead <=> 3 => 3 * interval at the last entry of series
                featureArray = a List of Strings with Column names that represent the features
        """
        df = data
        CLOEST_CANDIDATE_TRESHOLD = 10000000
        # A List with multiple DF
        COUNT_OF_FEATURES = len(featureArray)
        RETRY_TRESHOLD = nextDay_retry_threshold
        df_length = len(df)
        # result array = länge datengrundlage - interval * length, weil letzte serie nicht aufgeht
        resultNp = np.zeros((df_length - (length - 1), length, COUNT_OF_FEATURES))
        label_arr = np.zeros((df_length - (length - 1)))
        previousCandidate = df.iloc[0]
        previousCandidateIdx = 0
        i = 0  # 1 by 1 over all data, and build series out of it
        r = 0  # result array indx, dont have to be equal if i=2 has no valid series, I => 3 while r stays 2
        print(df.columns)
        print("length: " + str(df_length))
        while i < (df_length - length):
            if i % 10000 == 0:
                # Logging
                print("At index: ")
                perc_done = int((i * 100) / df_length)
                print('% done: ' + str(perc_done))
                print(i)

            isValid = True
            singleTimeSeriesArr = np.zeros((length, COUNT_OF_FEATURES))
            dateTimeArr = np.zeros(length).astype(datetime)
            posixTestArr = np.zeros(length)
            baseElement = df.iloc[i]
            previousCandidate = baseElement
            previousCandidateIdx = i
            baseElement_c_minuteOfDay = int(baseElement['posixMinute'])
            x = 0
            # insert first element of series
            singleTimeSeriesArr[x] = baseElement[[*featureArray]].values
            dateTimeArr[x] = baseElement['DateTime']
            posixTestArr[x] = baseElement_c_minuteOfDay
            # posixTestLabel = 0
            ### START FILL SERIES
            # look for the remaining length - 1 elements to fill the series
            while isValid and (x < length - 1):
                x = x + 1
                # we need to find the candidate without assuming an interval in the file
                # we assume that the list is ordnered but can contain empty slots
                #### START FIND CANDIDATE
                # the optimal time
                optimal_minute = int(previousCandidate['posixMinute']) + interval

                # get closest element to optimal element, with binary search, start looking at previous ele bc list is sorted
                candidateIdx, minimalAbw, closestCandidateIdx = \
                    self._binary_search(df, previousCandidateIdx, df_length - 1, optimal_minute,
                                        CLOEST_CANDIDATE_TRESHOLD, -1, False)

                # wenn gesuchter candidate = optimal_minute nicht gefunden:
                # 1-10minuten toleranz erlaubten und vorherige bzw. folge candidate nehmen
                # sonst wert vom start des nächsten tages
                if candidateIdx == -1:
                    # and abw > tolerance => go the next day, because trading day ended / or holidays..
                    if abs(minimalAbw) < tolerance:
                        candidateIdx = closestCandidateIdx
                    else:
                        # ersten Wert des folge Tages nehmen (oder nächster Handelstag...)
                        reTryCounter: int = 1
                        noValidAlternativeFound = True
                        while reTryCounter < RETRY_TRESHOLD and noValidAlternativeFound:
                            optimalDateTimeFromPosix = pd.Timestamp.fromtimestamp(optimal_minute * 60) - \
                                                       pd.Timedelta(hours=1)
                            nextDayStartDay = TimeModificationService.getNextDayVal(optimalDateTimeFromPosix, reTryCounter)
                            candidateIdx, minimalAbw, closestCandidateIdx = \
                                self._binary_search_dateTime(df, previousCandidateIdx, df_length - 1,
                                                             nextDayStartDay, 1000000, -1, True)
                            reTryCounter += 1
                            if candidateIdx != -1:
                                noValidAlternativeFound = False
                            # in seconds = 180second = 3min is allowed on next day
                            if not noValidAlternativeFound and abs(minimalAbw) <= 180:
                                candidateIdx = closestCandidateIdx
                                noValidAlternativeFound = False

                # wenn bis 8:20 gefunden -> zeitreihe verwerfen
                if candidateIdx == -1:
                    isValid = False
                    previousCandidateIdx = i + 1
                    # print(' NO VALID CANDIDATE ')
                else:
                    'Bei pd.TimeStamp = 16.01.2024 = 2024-01-16 bei data 2. day'
                    candidate = df.iloc[candidateIdx]
                    previousCandidate = candidate
                    previousCandidateIdx = candidateIdx
                    candidate_minuteOfDay = int(candidate['posixMinute'])
                    ### END FIND CANDIDATE
                    # Second check if the day fits as well
                    singleTimeSeriesArr[x] = candidate[[*featureArray]].values
                    dateTimeArr[x] = candidate['DateTime']
                    posixTestArr[x] = candidate_minuteOfDay
                    # if all elements in series are in range (within time)
                    # insert all into main array
                    # AND FINE LABEL
                    if x == length - 1:
                        # We have all items for a valid timeSeries, now we need to find the correct label
                        # START FIND LABEL
                        # optimal label time = last candidate time + time_ahead
                        optimal_label_minuteOfDay = (candidate_minuteOfDay + (time_ahead * interval))
                        #######################################
                        # Looking for label geater than last element of series
                        # (because clostes alternative can also be previous timestamp)
                        # label musst be always ahead of series
                        labelIdx, minimalAbw, closestCandidateIdx = \
                            self._binary_search(df, candidateIdx, df_length - 1, optimal_label_minuteOfDay,
                                                CLOEST_CANDIDATE_TRESHOLD, -1, True)
                        if minimalAbw < 30:
                            labelIdx = closestCandidateIdx

                        notFound = False
                        label_candidate = df.iloc[0]
                        # if no perfect Label Candidate Found. look for alternative
                        if labelIdx == -1:
                            # Optimal Label not found -> get value next day 8pm
                            # previousCandidateDateTime = previousCandidate['DateTime']  # letzter eintrag in timeSeries
                            optimalDateTimeFromPosix = pd.Timestamp.fromtimestamp(optimal_label_minuteOfDay * 60) - \
                                                       pd.Timedelta(hours=1)  # bc local Time != USA
                            nextDayStartDay = TimeModificationService.getNextDayVal(optimalDateTimeFromPosix, 1)
                            labelIdx, minimalAbw, closestCandidateIdx = \
                                self._binary_search_dateTime(df, candidateIdx, df_length - 1, nextDayStartDay,
                                                             CLOEST_CANDIDATE_TRESHOLD, -1, True)
                            if labelIdx == -1:
                                if abs(minimalAbw) < 60 * 10:  # 10min
                                    labelIdx = closestCandidateIdx
                                    notFound = False
                                else:
                                    notFound = True
                            else:
                                notFound = False

                        ### END FIND LABEL
                        # If label found, assign series and label to the arrays
                        if not notFound:
                            label_candidate = df.iloc[labelIdx]
                            # assign series to final array
                            resultNp[r] = singleTimeSeriesArr
                            # assign cadidate to final label array
                            label_arr[r] = label_candidate[label_name]
                            # dateTimeLabel = label_candidate['DateTime']
                            # print('dateTimeArr: ')
                            # print(f'for index: {i}: ')
                            # print(dateTimeArr)
                            # print(f'labelTime: {dateTimeLabel}')
                            # time.sleep(10)
                            r = r + 1
                        else:
                            pass
                            # print('NO LABEL')

            i = i + 1

        # REMOVE EMPTY ENTRYS FROM ARRAY, as there are empty rows possible not all of entries are used
        mainDataSeries = np.zeros((r, length, COUNT_OF_FEATURES))
        mainDataLabels = np.zeros(r)
        z = 0
        while z < r:
            mainDataSeries[z] = resultNp[z]
            mainDataLabels[z] = label_arr[z]
            z = z + 1

        # print('LEN SERIES: ')
        # print(len(toReturnLabel))
        print('SOLL_LENGTH = ' + str(i))
        print('LEN AFTER TS_BUILDER = ' + str(len(mainDataSeries)))
        print(mainDataSeries)
        return mainDataSeries, mainDataLabels

    def _binary_search(self, df: pd.DataFrame, lowIdx: int, highIdx: int, optimalTimeToFind: int, minimal_abw: int,
                       closestIdx: int, closestVal_isBigger: bool) -> tuple[int, int, int]:
        '''
        helper Method for finding best candidate
         @:param
            ..
            min_abw_idx_pair = tupel with idx and min abweichung, to get nearest item to the actaul one
        @:return
            fristReturn = the idx of the searched item or -1 if not found
            secondReturn = a Tuple of the idx of the item with smallest distance to nearest item or
                (-1,1) if exact item was found
        '''
        if highIdx >= lowIdx:
            midIdx = (highIdx + lowIdx) // 2
            midVal = int(df.iloc[midIdx]['posixMinute'])

            tmpAbw = abs(midVal - optimalTimeToFind)
            if tmpAbw < minimal_abw:
                if closestVal_isBigger:
                    if optimalTimeToFind < midVal:
                        minimal_abw = tmpAbw
                        closestIdx = midIdx
                else:
                    minimal_abw = tmpAbw
                    closestIdx = midIdx

            # If element is present at the middle itself
            if midVal == optimalTimeToFind:
                return midIdx, minimal_abw, closestIdx

            if midVal > optimalTimeToFind:
                return self._binary_search(df, lowIdx, midIdx - 1, optimalTimeToFind, minimal_abw, closestIdx,
                                           closestVal_isBigger)
            else:
                return self._binary_search(df, midIdx + 1, highIdx, optimalTimeToFind, minimal_abw, closestIdx,
                                           closestVal_isBigger)
        else:
            return -1, minimal_abw, closestIdx

    def _binary_search_dateTime(self, df: pd.DataFrame, lowIdx: int, highIdx: int, optimalDateToFind: pd.Timestamp,
                                minimal_abw: int, closestIdx: int, closestVal_isBigger: bool) -> tuple[int, int, int]:
        if highIdx >= lowIdx:
            midIdx = (highIdx + lowIdx) // 2
            midVal = df.iloc[midIdx]['DateTime']

            tmpAbw = abs(midVal.timestamp() - optimalDateToFind.timestamp())
            if tmpAbw < minimal_abw:
                if closestVal_isBigger:
                    if optimalDateToFind < midVal:
                        minimal_abw = tmpAbw
                        closestIdx = midIdx
                else:
                    minimal_abw = tmpAbw
                    closestIdx = midIdx

            # If element is present at the middle itself
            if tmpAbw < 10 * 60:  # if two dates in same minute => they are equal
                return midIdx, minimal_abw, closestIdx

            if midVal > optimalDateToFind:
                return self._binary_search_dateTime(df, lowIdx, midIdx - 1, optimalDateToFind, minimal_abw, closestIdx,
                                                    closestVal_isBigger)
            else:
                return self._binary_search_dateTime(df, midIdx + 1, highIdx, optimalDateToFind, minimal_abw, closestIdx,
                                                    closestVal_isBigger)
        else:
            return -1, minimal_abw, closestIdx

    def timeSeriesBuilderSimple(self, df: pd.DataFrame, startTimeToKeep: int, endTimeToKeep: int, lengthTs: int,
                                features: list, ahead: int) -> tuple[ndarray, ndarray]:
        """
            a sinple timeSeries Bulding Method without alternative calculations
        """
        # Set 'DateTime' as the index (required for resampling)
        data = df.copy()
        data.set_index('DateTime', inplace=True)
        data = data.resample('120T').mean()
        data.replace([np.inf, -np.inf, np.nan], None, inplace=True)
        data = data[(data.index.hour < endTimeToKeep) & (data.index.hour >= startTimeToKeep)]
        data.reset_index(inplace=True)

        LEN_DF = len(data)
        i = 0
        LEN_RETURN_ARR = LEN_DF - lengthTs
        toReturn_data = np.zeros((LEN_RETURN_ARR, lengthTs, len(features)))
        toReturn_label = np.zeros(LEN_RETURN_ARR)
        singleTs = np.zeros((lengthTs, len(features)))
        dateTimeArr = np.zeros(lengthTs).astype(datetime)
        while i < LEN_DF:
            x = 0
            singleTs = np.zeros((lengthTs, len(features)))
            while x < lengthTs:
                currentRow = data.iloc[i + x]
                singleTs = currentRow[[*features]]
                dateTimeArr = currentRow['DateTime']
                print(dateTimeArr)
                x += 1
            toReturn_data[i] = singleTs
            labelRow = data.iloc[i + x + ahead]
            toReturn_label[i] = labelRow['Open']
            i += 1

        return toReturn_data, toReturn_label
