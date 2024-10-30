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
    residual1 = []
    residual2 = []
    mse1 = []
    mse2 = []
    r_squared1 = []
    r_squared2 = []
    
    accumulator = values[0]
    cumulative_ssr1 = 0
    cumulative_ssr2 = 0
    cumulative_sst = 0
    
    for i, value in enumerate(values):
        if i == 0:
            # Initial forecast/residual placeholders for first value
            forecast1.append(None)
            forecast2.append(None)
            residual1.append(None)
            residual2.append(None)
            mse1.append(None)
            mse2.append(None)
            r_squared1.append(None)
            r_squared2.append(None)
            continue

        # Update accumulators and compute forecasts
        accumulator += value
        forecast1.append(values[i - 1])  # Using previous value as forecast
        forecast2.append(accumulator / (i + 1))  # Rolling average as forecast
        
        # Calculate residuals
        residual1.append(value - forecast1[i])
        residual2.append(value - forecast2[i])
        
        # Calculate MSE for each method incrementally
        mse1.append((residual1[i] ** 2) / (i + 1))
        mse2.append((residual2[i] ** 2) / (i + 1))
        
        # Update cumulative SSR (sum of squared residuals)
        cumulative_ssr1 += residual1[i] ** 2
        cumulative_ssr2 += residual2[i] ** 2
        
        # Update cumulative SST (total sum of squares) based on the current mean
        mean = accumulator / (i + 1)
        cumulative_sst = sum((val - mean) ** 2 for val in values[:i])
        
        # Calculate R^2 for each method
        r_squared1.append(1 - (cumulative_ssr1 / cumulative_sst) if cumulative_sst != 0 else None)
        r_squared2.append(1 - (cumulative_ssr2 / cumulative_sst) if cumulative_sst != 0 else None)
        
    # Calculate forecasts for month 8 (as done previously)
    forecast1.append(values[-1])
    forecast2.append(accumulator / len(values))
    
    # Debugging output
    if DEBUG:
        print(f"Values: {values}")
        print(f"Forecast (Most Recent Value): {forecast1}")
        print(f"Forecast (Rolling Average): {forecast2}")
        print(f"Residual (Most Recent Value): {residual1}")
        print(f"Residual (Rolling Average): {residual2}")
        print(f"MSE (Most Recent Value): {mse1}")
        print(f"MSE (Rolling Average): {mse2}")
        print(f"R^2 (Most Recent Value): {r_squared1}")
        print(f"R^2 (Rolling Average): {r_squared2}")

    # Final output
    print(f"MSE Average (Most Recent Value): {sum(abs(i) for i in mse1 if i is not None) / len([i for i in mse1 if i is not None])}")
    print(f"MSE Average (Rolling Average): {sum(abs(i) for i in mse2 if i is not None) / len([i for i in mse2 if i is not None])}")
    if DEBUG:
        print(f"Final R^2 (Most Recent Value): {r_squared1[-1]}")
        print(f"Final R^2 (Rolling Average): {r_squared2[-1]}")
    print(f"Month 8 Forecast (Most Recent Value): {forecast1[-1]}")
    print(f"Month 8 Forecast (Rolling Average): {forecast2[-1]}")