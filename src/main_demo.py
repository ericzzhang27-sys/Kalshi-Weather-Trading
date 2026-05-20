from bucket_schema import Bucket
from error_boundaries import convert_market_to_boundaries, convert_nws_to_boundaries


if __name__ == "__main__":
    market_boundaries = convert_market_to_boundaries([56, 57, 58, 59, 60, 61, 62, 63, 64, 65], "Chicago")
    nws_boundaries = convert_nws_to_boundaries(75, "New York")
    print("Market Boundaries:")
    for bucket in market_boundaries:
        print(bucket)
    