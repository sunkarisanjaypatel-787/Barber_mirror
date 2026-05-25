from icrawler.builtin import BingImageCrawler
import os

print("[*] INITIATING AUTONOMOUS HARVESTER...")

# The Target Vectors: Optimized to pull distinct male facial geometry
target_shapes = {
    "square": [
        "square face shape men portrait front view", 
        "square jawline male face straight on",
        "men square face short hair portrait"
    ],
    "oval": [
        "oval face shape men portrait front view", 
        "oval face male straight on",
        "men oval face short hair portrait"
    ],
    "oblong": [
        "oblong face shape men portrait front view", 
        "long face male straight on",
        "men oblong face short hair portrait"
    ],
    "round": [
        "round face shape men portrait front view", 
        "round face male straight on",
        "men round face short hair portrait"
    ],
    "diamond": [
        "diamond face shape men portrait front view", 
        "diamond face male straight on",
        "men diamond face short hair portrait"
    ]
}

# The Extraction Loop
for shape, queries in target_shapes.items():
    output_dir = f"dataset/{shape}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[+] LOCKING TARGET: {shape.upper()} FACE GEOMETRY")
    
    for query in queries:
        print(f"    -> Executing Vector: '{query}'")
        # Bing is highly permissive for automated scraping compared to Google
        crawler = BingImageCrawler(
            storage={'root_dir': output_dir},
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4
        )
        # Pull 200 images per query (Yields ~600 raw images per shape)
        crawler.crawl(keyword=query, max_num=200)

print("\n[*] HARVEST COMPLETE. INITIATE MANUAL PURGE.")
