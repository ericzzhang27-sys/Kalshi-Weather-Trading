from bucket_schema import Bucket
from error_boundaries import convert_market_to_boundaries, convert_nws_to_boundaries


if __name__ == "__main__":
    market_boundaries = convert_market_to_boundaries([30, 60, 90], "New York")
    nws_boundaries = convert_nws_to_boundaries(75, "New York")
    print("Market Boundaries:")
    for bucket in market_boundaries:
        print(bucket)
    print("\nNWS Boundaries:")
    for bucket in nws_boundaries:
        print(bucket)