from src.recommender import GreedyRecommender, MMRRecommender

types = {
    "concat": lambda x, y: x + y,
    "GreedyRecommender": GreedyRecommender,
    "MMRRecommender": MMRRecommender,
}
