#/usr/bin/env python3

# Author: Jason Gardner
# Date: 10/30/2024
# Class: CAP6768
# Assignment: Discussion 6

'''
    Compute MSE using the most recent value as the forecast for the next period. What is the forecast for month 8?
    Compute MSE using the average of all the data available as the forecast for the next period. What is the forecast for month 8?
    Which method appears to provide the better forecast?
'''

DEBUG = False

if __name__ == "__main__":
    values = [24, 13, 20, 12, 19, 23, 15]
    forecast1 = []
    forecast2 = []
    mse1 = []
    mse2 = []
    for i, value in enumerate(values):
        if i == 0:
            accumulator = value
            forecast1.append(None)
            forecast2.append(None)
            mse1.append(None)
            mse2.append(None)
            continue
        accumulator += value
        forecast1.append(values[i - 1])
        forecast2.append(accumulator / (i + 1))
        mse1.append((((value - forecast1[i]) ** 2)/(i + 1)))
        mse2.append((((value - forecast2[i]) ** 2)/(i + 1)))
        
    # Calculate forecasts for month 8
    forecast1.append(values[-1])  # Most recent value as forecast
    forecast2.append(accumulator / len(values) + 1)  # Average of all data as forecast
    mse1.append((((values[-1] - forecast1[-1]) ** 2)/(len(values) + 1)))
    mse2.append((((values[-1] - forecast2[-1]) ** 2)/(len(values) + 1)))
    
    
    if DEBUG:
        print(f"Values: {values}")
        print(f"Forecast (Most Recent Value):: {forecast1}")
        print(f"Forecast (Rolling Average): {forecast2}")
        print(f"MSE Most Recent Value):: {mse1}")
        print(f"MSE (Rolling Average): {mse2}")

    print(f"MSE Average (Most Recent Value):: {sum([abs(i) for i in mse1 if i is not None]) / len([i for i in mse1 if i is not None])}")
    print(f"MSE Average (Rolling Average): {sum([abs(i) for i in mse2 if i is not None]) / len([i for i in mse2 if i is not None])}")
    print(f"Month 8 Forecast (Most Recent Value):: {forecast1[-1]}")
    print(f"FMonth 8 Forecast (Rolling Average): {forecast2[-1]}")