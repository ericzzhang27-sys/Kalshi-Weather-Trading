from bucket_schema import Bucket

def convert_market_to_boundaries(temperatures:list[int], location: str) -> list[Bucket]:
    buckets: list[Bucket] = []
    buckets.append(Bucket(left_bound= None, right_bound=temperatures[0], location=location))
    for i in range(1, len(temperatures)-1):
        buckets.append(Bucket(left_bound=temperatures[i], right_bound=temperatures[i+1], location=location))
    buckets.append(Bucket(left_bound=temperatures[-1], right_bound=None, location=location))
    return buckets

def convert_nws_to_boundaries(temperature: int, location: str) -> list[Bucket]:
    buckets: list[Bucket] = []
    buckets.append(Bucket(left_bound=None, right_bound=temperature-2, location=location))
    buckets.append(Bucket(left_bound=temperature-2, right_bound=temperature-1, location=location))
    buckets.append(Bucket(left_bound=temperature-1, right_bound=temperature, location=location))
    buckets.append(Bucket(left_bound=temperature, right_bound=temperature+1, location=location))
    buckets.append(Bucket(left_bound=temperature+1, right_bound=temperature+2, location=location))
    buckets.append(Bucket(left_bound=temperature+2, right_bound=None, location=location))
    return buckets