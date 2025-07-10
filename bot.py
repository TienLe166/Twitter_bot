# coding: utf-8
import requests
import time
import random
import base64
import hmac
import hashlib
import urllib.parse
import schedule
from datetime import datetime, timedelta
import os
import json
import re
from keep_alive import keep_alive

# ------------------------------------------------------------
# Tweet History Tracking System
# ------------------------------------------------------------
TWEET_HISTORY_FILE = "tweet_history.json"

def load_tweet_history():
    """Load tweet history"""
    try:
        if os.path.exists(TWEET_HISTORY_FILE):
            with open(TWEET_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "project_mentions": {},
                "total_tweets": 0,
                "last_tweet_date": None
            }
    except Exception as e:
        print(f"❌ Error loading tweet history: {e}")
        return {
            "project_mentions": {},
            "total_tweets": 0,
            "last_tweet_date": None
        }

def save_tweet_history(history):
    """Save tweet history"""
    try:
        with open(TWEET_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Error saving tweet history: {e}")

def update_project_mention_history(project_key, mention_type="general"):
    """Update project mention history"""
    history = load_tweet_history()
    today = datetime.now().strftime('%Y-%m-%d')

    if project_key not in history["project_mentions"]:
        history["project_mentions"][project_key] = {
            "count": 0,
            "last_mentioned": None,
            "mention_types": [],
            "first_mention_date": today
        }

    # Update
    history["project_mentions"][project_key]["count"] += 1
    history["project_mentions"][project_key]["last_mentioned"] = today
    history["project_mentions"][project_key]["mention_types"].append(mention_type)

    # Keep only last 10 mention types (prevent file bloat)
    if len(history["project_mentions"][project_key]["mention_types"]) > 10:
        history["project_mentions"][project_key]["mention_types"] = history["project_mentions"][project_key]["mention_types"][-10:]

    history["total_tweets"] += 1
    history["last_tweet_date"] = today

    save_tweet_history(history)
    print(f"📊 {project_key} mention history updated: mentioned {history['project_mentions'][project_key]['count']} times")

def get_project_mention_count(project_key):
    """Get how many times project was mentioned"""
    history = load_tweet_history()
    if project_key in history["project_mentions"]:
        return history["project_mentions"][project_key]["count"]
    return 0

def get_days_since_last_mention(project_key):
    """Get days since last mention"""
    history = load_tweet_history()
    if project_key in history["project_mentions"] and history["project_mentions"][project_key]["last_mentioned"]:
        last_date = datetime.strptime(history["project_mentions"][project_key]["last_mentioned"], '%Y-%m-%d')
        today = datetime.now()
        return (today - last_date).days
    return 999  # Never mentioned

def select_smart_opening_style(project_key):
    """Select smart opening style based on history"""
    mention_count = get_project_mention_count(project_key)
    days_since = get_days_since_last_mention(project_key)

    if mention_count == 0:
        # First time mention
        return "first_discovery"
    elif mention_count == 1:
        # Second mention
        if days_since <= 7:
            return "recent_follow_up"
        else:
            return "rediscovery"
    elif mention_count >= 2:
        # Third+ mention
        if days_since <= 3:
            return "frequent_update"
        elif days_since <= 14:
            return "regular_check"
        else:
            return "long_term_follow"

    return "general"  # fallback

# ------------------------------------------------------------
# Load environment variables from .env file and set UTF-8 output
# ------------------------------------------------------------
import sys as _sys

# Prevent console Unicode errors
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")

# Load .env file (using python-dotenv if available, else simple fallback)
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except ModuleNotFoundError:
    import pathlib
    env_path = pathlib.Path(__file__).with_name('.env')
    if env_path.exists():
        for _line in env_path.read_text(encoding='utf-8').splitlines():
            if '=' in _line and not _line.strip().startswith('#'):
                k, v = _line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

# Enhanced Twitter Bot v4 - Ultra Natural Crypto Insider Style
# 🔒 SECURE: All API keys from environment variables

# API Keys - From environment variables
api_key = os.getenv('TWITTER_API_KEY')
api_secret = os.getenv('TWITTER_API_SECRET') 
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_secret = os.getenv('TWITTER_ACCESS_SECRET')
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
openai_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY') or "AIzaSyBKUhepHbuVBiaYzkQkZEvnbfEO5MJEgJM"

# API key check
print(f"🔍 API Key Check:")
print(f"   Twitter API Key: {'✅' if api_key else '❌'} {f'({api_key[:10]}...)' if api_key else ''}")
print(f"   OpenAI Key: {'✅' if openai_key else '❌'} (length: {len(openai_key) if openai_key else 0})")
print(f"   Gemini Key: {'✅' if gemini_key else '❌'} (length: {len(gemini_key) if gemini_key else 0})")
if openai_key:
    print(f"   OpenAI Key start: {openai_key[:20]}...")
    print(f"   OpenAI Key end: ...{openai_key[-10:]}")
if gemini_key:
    print(f"   Gemini Key start: {gemini_key[:20]}...")
    print(f"   Gemini Key end: ...{gemini_key[-10:]}")

print(f"🌍 Environment Variables:")
for key in ['TWITTER_API_KEY', 'TWITTER_API_SECRET', 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_SECRET', 'OPENAI_API_KEY', 'GEMINI_API_KEY']:
    value = os.getenv(key)
    print(f"   {key}: {'✅ SET' if value else '❌ MISSING'}")

# API KEY CHECK - Production only uses environment variables
import sys
if not all([api_key, api_secret, access_token, access_secret]):
    print("❌ Required Twitter API environment variables missing!")
    print("🔧 Please set these environment variables:")
    print("   - TWITTER_API_KEY")
    print("   - TWITTER_API_SECRET") 
    print("   - TWITTER_ACCESS_TOKEN")
    print("   - TWITTER_ACCESS_SECRET")
    print("💡 On Heroku: heroku config:set TWITTER_API_KEY=your_key")
    sys.exit(1)

# AI API check - Gemini priority, OpenAI fallback
if not gemini_key and not openai_key:
    print("❌ No AI API key found!")
    print("🔧 Set at least one:")
    print("   - GEMINI_API_KEY (priority)")
    print("   - OPENAI_API_KEY (fallback)")
    sys.exit(1)

if gemini_key:
    print("✅ Gemini API key found! Will be used as primary AI.")
elif openai_key:
    print("✅ OpenAI API key found! Will be used as fallback AI.")

print("✅ All API keys loaded!")

# Current Tracked Projects - 8 Projects
projects = {
    "Vooi": {
        "mention": "@vooi_io", 
        "focus": "DeFi, DEX aggregation", 
        "specialty": "Aggregating perpetual contract DEXes with zero-gas trading",
        "trends": ["Multi-chain DeFi", "Gasless transactions", "cross-chain DeFi", "Perpetual contract trading"],
        "price_action": "Points program for potential airdrop, active campaign on Cookie platform",
        "ecosystem": "Multi-chain DeFi",
        "personality": "Efficiency and user experience focused",
        "token_status": "pre_token",
        "tech_detail": "Intent-based trading for zero-gas transactions, aggregating liquidity from multiple perpetual DEXes, cross-chain compatibility"
    },
    "recall": {
      "mention": "@recallnet",
      "focus": "AI and blockchain",
      "specialty": "Decentralized intelligence network for AI agents",
      "trends": [
        "AI agents",
        "Decentralized knowledge networks",
        "On-chain AI"
      ],
      "price_action": "Token launched, trading, featured in SNAPS campaign on Cookie",
      "ecosystem": "EVM-compatible chains",
      "personality": "AI and decentralization focused",
      "token_status": "Active",
      "tech_detail": "Decentralized network for AI agents to store and share knowledge, with AgentRank reputation system, token incentives for participation"
    },
    "memex": {
        "mention": "@MemeX_MRC20",
        "focus": "meme coin infrastructure",
        "specialty": "meme coin creation and management platform",
        "trends": ["meme coin season", "retail trader tools", "automated meme trading", "social media integration"],
        "price_action": "gained momentum with meme coin trend",
        "ecosystem": "Meme coin creation platform",
        "personality": "social and entertainment focused", 
        "token_status": "active",
        "tech_detail": "One-click meme coin deployment, automated liquidity provision, social sentiment tracking, viral marketing tools integration."
    },
    "elympics": {
      "mention": "@elympics_ai",
      "focus": "Web3 gaming",
      "specialty": "Protocol for building competitive Web3 games",
      "trends": [
        "Web3 gaming",
        "Competitive gaming",
        "Blockchain in gaming"
      ],
      "price_action": "Token launched, trading, $COOKIE stakers whitelisted for $ELP presale",
      "ecosystem": "Multi-chain gaming",
      "personality": "Gaming and competition focused",
      "token_status": "Active",
      "tech_detail": "Web3 gaming protocol with Unity integration, provable fairness, on-chain leaderboards, and token rewards for players"
    },
    "mitosis": {
        "mention": "@mitosis_org", 
        "focus": "liquidity fragmentation solution", 
        "specialty": "liquidity protocol aiming to set new standards in DeFi",
        "trends": ["liquidity protocol developments", "automated market making", "cross-chain liquidity", "DeFi yield optimization"],
        "price_action": "TVL growing rapidly, governance activity increasing",
        "ecosystem": "Next-gen DeFi protocol",
        "personality": "DeFi and yield focused",
        "token_status": "pre_token",
        "tech_detail": "Dynamic arbitrage bots, multi-chain slippage optimization, smart liquidity routing. Up to 40% gas savings."
    },
    "virtuals": {
        "mention": "@virtuals_io", 
        "focus": "AI agent marketplace", 
        "specialty": "platform tokenizing AI agents to create an economy",
        "trends": ["AI agent tokens gaining interest", "AI tokenization", "GameFi AI integrations", "AI agent market growing"],
        "price_action": "performing in AI token sector",
        "ecosystem": "AI agent marketplace with Base integration and partnerships (Nillion, Virtual Labs, Aikoi.ai)",
        "personality": "AI and tokenization focused",
        "token_status": "active",
        "tech_detail": "Marketplace with task execution, revenue sharing, 1000+ AI agents, Super APP"
    },
    "cookie": {
      "mention": "@cookiedotfun",
      "focus": "AI and data",
      "specialty": "Indexing and providing real-time data on AI agents",
      "trends": [
        "AI data",
        "Decentralized data markets",
        "AI agents"
      ],
      "price_action": "Token launched, trading",
      "ecosystem": "Ethereum or EVM-compatible",
      "personality": "Data and AI focused",
      "token_status": "Active",
      "tech_detail": "Aggregating and indexing data from AI agents, providing analytics and insights, with API access for developers"
    },
    "openledger": {
      "mention": "@OpenledgerHQ",
      "focus": "AI and blockchain",
      "specialty": "Decentralized compute for AI models",
      "trends": [
        "Decentralized AI",
        "AI compute",
        "Blockchain for AI"
      ],
      "price_action": "Points program for potential token, active campaign on Cookie",
      "ecosystem": "Multi-chain or AI-focused chains",
      "personality": "AI and decentralization focused",
      "token_status": "Pre-token",
      "tech_detail": "Decentralized infrastructure for training and deploying AI models, using blockchain for resource allocation and rewards"
    },
    "kaito": {
      "mention": "@KaitoAI",
      "request": "AI-powered crypto research",
      "specialty": "Using AI to analyze crypto data and provide insights",
      "trends": [
        "AI in crypto",
        "Data analysis",
        "Research tools"
      ],
      "price_action": "Token launched, trading",
      "ecosystem": "Ethereum or similar",
      "personality": "Research and data focused",
      "token_status": "Active",
      "tech_detail": "AI-powered search engine and analytics platform for crypto data, providing insights on market trends, sentiment, and more"
    }
}

# Tweet length categories - FLEXIBLE AND LOGICAL
TWEET_LENGTHS = {
    "short": {"weight": 40, "min": 180, "max": 500, "style": "concise"},        # 40% - Short and to the point
    "medium": {"weight": 35, "min": 500, "max": 1200, "style": "normal"},       # 35% - Normal detailed
    "long": {"weight": 20, "min": 1200, "max": 2500, "style": "analysis"},      # 20% - Long analysis
    "thread": {"weight": 5, "min": 2500, "max": 4000, "style": "thread"}        # 5% - Thread format
}

# TWEET TYPES - DETAILED ANALYSIS FOCUSED
TWEET_TYPES = {
    "tech_deep": {
        "weight": 25,
        "style": "Technology focused deep explanation",
        "tone": "Technical but understandable, informative"
    },
    "market_perspective": {
        "weight": 20,
        "style": "Market analysis and opinion",
        "tone": "Analytical but personal opinion"
    },
    "casual_discovery": {
        "weight": 18,
        "style": "Casual discovery like random find",
        "tone": "Curious, discovering, genuine"
    },
    "daily_metaphor": {
        "weight": 15,
        "style": "Technical explanation with daily life metaphors",
        "tone": "Fun but educational, cultural references"
    },
    "comparison": {
        "weight": 12,
        "style": "Comparison with other projects", 
        "tone": "Comparative, objective"
    },
    "quote_commentary": {
        "weight": 8,
        "style": "Commenting on project tweet",
        "tone": "Commentative, adding personal view"
    },
    "experience_share": {
        "weight": 8,
        "style": "Sharing personal experience",
        "tone": "Experience focused, genuine"
    },
    "question_wonder": {
        "weight": 6,
        "style": "Asking questions and wondering",
        "tone": "Curious, thought-provoking"
    },
    "future_prediction": {
        "weight": 4,
        "style": "Future prediction",
        "tone": "Speculative but logical"
    }
}

# Tweet system - 12 tweets per day between 8am and midnight (Europe time)
last_tweet_time = None
MINIMUM_INTERVAL = 1.33 * 60 * 60  # 1.33 hours (seconds) - 12 tweets/day (16 hours ÷ 12 = 1.33 hours)
DAILY_TWEET_COUNT = 12
TWEET_START_HOUR = 8   # 8am (Europe time)
TWEET_END_HOUR = 24    # midnight (00:00)
current_project_index = 0  # For project rotation

def create_oauth_signature(method, url, params, consumer_secret, token_secret):
    """Create OAuth 1.0a signature"""
    # Encode and sort parameters
    encoded_params = []
    for key, value in params.items():
        encoded_key = urllib.parse.quote(str(key), safe='')
        encoded_value = urllib.parse.quote(str(value), safe='')
        encoded_params.append((encoded_key, encoded_value))

    sorted_params = sorted(encoded_params)
    param_string = '&'.join([f"{k}={v}" for k, v in sorted_params])

    # Create base string
    encoded_url = urllib.parse.quote(url, safe='')
    encoded_params_string = urllib.parse.quote(param_string, safe='')
    base_string = f"{method}&{encoded_url}&{encoded_params_string}"

    # Create signing key
    encoded_consumer_secret = urllib.parse.quote(consumer_secret, safe='')
    encoded_token_secret = urllib.parse.quote(token_secret, safe='')
    signing_key = f"{encoded_consumer_secret}&{encoded_token_secret}"

    # Calculate signature
    signature = base64.b64encode(
        hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
    ).decode()

    return signature

def create_oauth_header(method, url, params=None):
    """Create OAuth 1.0a authorization header"""
    if params is None:
        params = {}

    # OAuth parameters
    oauth_params = {
        'oauth_consumer_key': api_key,
        'oauth_token': access_token,
        'oauth_signature_method': 'HMAC-SHA1',
        'oauth_timestamp': str(int(datetime.now().timestamp())),
        'oauth_nonce': str(random.randint(100000, 999999)),
        'oauth_version': '1.0'
    }

    print(f"🔑 OAuth params: {oauth_params}")
    print(f"📋 Extra params: {params}")

    # Combine all parameters (including POST body params)
    all_params = {**oauth_params, **params}

    # Create signature
    signature = create_oauth_signature(method, url, all_params, api_secret, access_secret)
    oauth_params['oauth_signature'] = signature

    print(f"✍️ Generated signature: {signature}")

    # Create authorization header
    auth_parts = []
    for key, value in sorted(oauth_params.items()):
        auth_parts.append(f'{key}="{urllib.parse.quote(str(value), safe="")}"')

    return f"OAuth {', '.join(auth_parts)}"

def search_twitter_sentiment(project_key):
    """Search recent tweets about project and analyze sentiment"""
    try:
        project = projects[project_key]

        # Twitter API v2 search with Bearer token
        url = "https://api.twitter.com/2/tweets/search/recent"

        # Create search query - use project name and mention
        project_name = project['mention'].replace('@', '')
        query_parts = [f'"{project_name}"', f'"{project_key}"']

        query = f"({' OR '.join(query_parts)}) -is:retweet lang:en"

        params = {
            'query': query,
            'max_results': 20,
            'tweet.fields': 'created_at,public_metrics,text',
            'sort_order': 'recency'
        }

        headers = {"Authorization": f"Bearer {bearer_token}"}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            tweets = data.get('data', [])

            if tweets:
                # Analyze tweets for sentiment
                recent_topics = []
                engagement_data = []

                for tweet in tweets:
                    text = tweet['text'].lower()
                    metrics = tweet['public_metrics']

                    # Calculate engagement (like + retweet*2 + reply*3)
                    engagement = metrics['like_count'] + (metrics['retweet_count'] * 2) + (metrics['reply_count'] * 3)
                    engagement_data.append(engagement)

                    # Simple sentiment
                    positive_words = ['good', 'great', 'bullish', 'moon', 'pump', 'win', 'amazing', 'love']
                    negative_words = ['bad', 'dump', 'bearish', 'scam', 'rekt', 'down', 'crash', 'hate']

                    pos_count = sum(1 for word in positive_words if word in text)
                    neg_count = sum(1 for word in negative_words if word in text)

                    if pos_count > neg_count:
                        recent_topics.append("positive_sentiment")
                    elif neg_count > pos_count:
                        recent_topics.append("negative_sentiment") 
                    else:
                        recent_topics.append("neutral_news")

                # Most common sentiment
                if recent_topics:
                    sentiment = max(set(recent_topics), key=recent_topics.count)
                else:
                    sentiment = "neutral_news"

                # Engagement level
                avg_engagement = sum(engagement_data) / len(engagement_data) if engagement_data else 0
                if avg_engagement > 100:
                    engagement_level = "high"
                elif avg_engagement > 20:
                    engagement_level = "medium"
                else:
                    engagement_level = "low"

                return {
                    "sentiment": sentiment,
                    "engagement_level": engagement_level,
                    "topics": recent_topics[:3]
                }

        # Default fallback
        return {
            "sentiment": random.choice(["positive_sentiment", "neutral_news", "negative_sentiment"]),
            "engagement_level": random.choice(["low", "medium", "high"]),
            "topics": ["general_discussion"]
        }

    except Exception as e:
        print(f"🔍 Twitter sentiment search error: {e}")
        return {
            "sentiment": random.choice(["positive_sentiment", "neutral_news"]),
            "engagement_level": "medium",
            "topics": ["general_discussion"]
        }

def find_recent_project_tweet(project_key):
    """Find recent tweets from project account for quote tweets"""
    try:
        project = projects[project_key]
        username = project['mention'].replace('@', '')

        # Get recent tweets from project account using Twitter API v2
        url = f"https://api.twitter.com/2/tweets/search/recent"

        params = {
            'query': f'from:{username} -is:retweet',
            'max_results': 10,
            'tweet.fields': 'created_at,public_metrics,text',
            'sort_order': 'recency'
        }

        headers = {"Authorization": f"Bearer {bearer_token}"}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            tweets = data.get('data', [])

            if tweets:
                # Select most suitable tweet (within last 24 hours, announcement/update type)
                for tweet in tweets:
                    text = tweet['text'].lower()

                    # Prioritize announcement/update tweets
                    announcement_keywords = ['launch', 'announce', 'update', 'release', 'new', 'coming', 'excited', 'partnership']
                    if any(keyword in text for keyword in announcement_keywords):
                        return {
                            'id': tweet['id'],
                            'text': tweet['text'][:100] + "..." if len(tweet['text']) > 100 else tweet['text'],
                            'username': username
                        }

                # If no announcement, take latest tweet
                return {
                    'id': tweets[0]['id'],
                    'text': tweets[0]['text'][:100] + "..." if len(tweets[0]['text']) > 100 else tweets[0]['text'],
                    'username': username
                }

        print(f"🔍 No recent tweet found for {username}")
        return None

    except Exception as e:
        print(f"🔍 Project tweet search error: {e}")
        return None

def get_time_based_tone():
    """Determine tweet tone based on time - Feature #12"""
    current_hour = datetime.now().hour

    if 6 <= current_hour < 12:
        return {
            "tone": "energetic_morning",
            "modifier": "With morning energy, positive and motivational"
        }
    elif 12 <= current_hour < 17:
        return {
            "tone": "analytical_noon", 
            "modifier": "Afternoon analytical approach, more detailed"
        }
    elif 17 <= current_hour < 22:
        return {
            "tone": "casual_evening",
            "modifier": "Evening relaxed atmosphere, more genuine and sharing"
        }
    else:
        return {
            "tone": "chill_night",
            "modifier": "Night calmness, thoughtful and in-depth"
        }

def choose_tweet_length():
    """Weighted random tweet length selection - flexible and logical"""
    rand = random.randint(1, 100)
    if rand <= 40:
        return TWEET_LENGTHS["short"]
    elif rand <= 75:  # 40 + 35
        return TWEET_LENGTHS["medium"] 
    elif rand <= 95:  # 40 + 35 + 20
        return TWEET_LENGTHS["long"]
    else:  # 5% - Thread
        return TWEET_LENGTHS["thread"]

def choose_tweet_type():
    """Weighted random tweet type selection - for natural diversity"""
    total_weight = sum(t["weight"] for t in TWEET_TYPES.values())
    rand = random.randint(1, total_weight)

    current_weight = 0
    for type_name, type_data in TWEET_TYPES.items():
        current_weight += type_data["weight"]
        if rand <= current_weight:
            return type_name, type_data

    # Fallback
    return "casual_discovery", TWEET_TYPES["casual_discovery"]

def clean_tweet(tweet, length_config, clean_project_name):
    """Tweet cleaning function - for both Gemini and OpenAI"""
    if not tweet:
        return None

    # HASHTAG AND LONG DASH CLEANING
    tweet = tweet.replace('—', ' ')
    tweet = tweet.replace('–', ' ')
    tweet = tweet.replace('-', ' ')

    # Remove hashtags
    import re
    tweet = re.sub(r'#\w+', '', tweet)  # Remove #bitcoin, #crypto etc.
    tweet = re.sub(r'\s+', ' ', tweet)  # Collapse multiple spaces
    tweet = tweet.strip()  # Remove leading/trailing spaces

    # Fix if starts with @ (won't show in main timeline otherwise)
    if tweet.startswith('@'):
        # Find @mention and rearrange tweet
        parts = tweet.split(' ', 1)
        if len(parts) > 1:
            mention = parts[0]
            rest = parts[1]
            # Remove @, get project name
            project_name = mention.replace('@', '').replace('_', ' ').title()
            tweet = f"{project_name} {rest}"
            print(f"🔧 Started with @, fixed: {tweet}")

    # Replace all @ mentions with project name and remove unnecessary words
    import re
    for project_key, project_data in projects.items():
        mention = project_data['mention']
        project_name = mention.replace('@', '').replace('_', ' ').title()
        # Replace variations of @ mention and clean name
        variations = [
            mention,  # @campnetworkxyz
            mention.replace('@', ''),  # campnetworkxyz
            mention.replace('@', '').lower(),  # campnetworkxyz
            mention.replace('@', '').capitalize(),  # Campnetworkxyz
            mention.replace('@', '').upper()  # CAMPNETWORKXYZ
        ]
        for var in variations:
            tweet = tweet.replace(var, project_name)

    # Remove unnecessary words
    unwanted_phrases = [
        "this ", "well ", "how I think", "I think how", 
        "how well", "well how", "this project", "that project"
    ]
    for phrase in unwanted_phrases:
        tweet = tweet.replace(phrase, "")

    # Clean multiple spaces and fix
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    # Fix paragraph format - add blank line at logical paragraph transitions
    if len(tweet) > 800:  # Only for long tweets
        # First split tweet into sentences
        sentences = re.split(r'([.!?])\s+', tweet)
        formatted_sentences = []
        current_paragraph = ""

        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            current_paragraph += sentence + " "

            # Paragraph transition conditions:
            # 1. If over 200+ characters and logical transition exists
            # 2. Topic change keywords
            topic_changes = [
                "but", "however", "on the other hand", 
                "another", "meanwhile", "now", "so",
                "in conclusion", "in summary"
            ]

            if len(current_paragraph) > 200:
                # Check next sentence for topic change
                next_sentence_start = sentences[i+2] if i+2 < len(sentences) else ""
                if any(word in next_sentence_start.lower()[:50] for word in topic_changes):
                    formatted_sentences.append(current_paragraph.strip())
                    current_paragraph = ""

        # Add remaining part
        if current_paragraph.strip():
            formatted_sentences.append(current_paragraph.strip())

        # Join paragraphs with blank line
        if len(formatted_sentences) > 1:
            tweet = "\n\n".join(formatted_sentences)
            print(f"📝 {len(formatted_sentences)} paragraphs created")
        else:
            print(f"📝 Left as single paragraph")

    print(f"🧹 Cleaned tweet: {tweet}")

    # Length check - shorten or extend if needed
    if len(tweet) > length_config['max']:
        # Smart shortening - cut at sentence end
        max_length = length_config['max'] - 10  # 10 character buffer
        if max_length < len(tweet):
            # Find last sentence end
            cut_tweet = tweet[:max_length]
            last_period = cut_tweet.rfind('.')
            last_exclamation = cut_tweet.rfind('!')
            last_question = cut_tweet.rfind('?')

            # Find last punctuation
            last_punct = max(last_period, last_exclamation, last_question)

            if last_punct > max_length - 200:  # If not too far back
                tweet = tweet[:last_punct + 1]
                print(f"✂️ Tweet smart shortened (at sentence end): {len(tweet)} characters")
            else:
                tweet = tweet[:max_length] + "..."
                print(f"✂️ Tweet forcibly shortened: {len(tweet)} characters")

    elif len(tweet) < length_config['min']:
        tweet += " Worth following I think."
        print(f"📏 Tweet extended: {len(tweet)} characters")

    print(f"✅ Using AI tweet!")
    return tweet

def call_gemini_api(prompt, length_config, clean_project_name):
    """Gemini API call - create tweet with 2.5 Pro model (strongest reasoning + 2M context)"""
    try:
        # Length based prompt
        is_short = length_config['style'] == 'concise'
        is_thread = length_config['style'] == 'thread'

        # Combine system prompt and user prompt
        combined_prompt = f"""You're an ordinary person following the crypto market. You're not AI, you're a real crypto enthusiast.

RULE:
- Write tweet between {length_config['min']}-{length_config['max']} characters
- Use {clean_project_name} name naturally (no @ symbol)
- Don't use hashtags at all
- Not formal like AI, talk friendly
{f"- Short and to the point, get straight to it" if is_short else "- Give detailed explanation but don't lose genuine tone"}
{f"- Put BLANK LINE between paragraphs" if not is_short else ""}

DON'T DO THESE:
- "when I analyzed", "when I evaluated", "when I examined" AI language
- "important for ecosystem", "notable development" buzzwords
- "today I examined X project" cliché openings
- Don't invent your own opening, use the given one!

DO THESE:
- Start tweet with the given smart opening
- Then continue: "pretty cool", "looks interesting", "not bad"
- "X's this part is pretty cool"
- "still early but X..."

TONE: Crypto follower friend, genuine, curious but not exaggerated

{f"SHORT TWEET STYLE: Get to the point, don't elaborate, say directly what you think" if is_short else ""}
{f"THREAD STYLE: Long article format, BLANK LINE between paragraphs" if is_thread else ""}

{prompt}

Just write the tweet, don't add anything else."""

        # Gemini API URL - 2.5 Pro model (strongest and highest quality)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={gemini_key}"

        # Request data
        data = {
            "contents": [{
                "parts": [{
                    "text": combined_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 1.1,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 12000 if length_config['style'] == 'thread' else 4000
            }
        }

        headers = {"Content-Type": "application/json"}

        print(f"🤖 Gemini 2.5 Pro API call being made...")
        print(f"🔑 API Key: {gemini_key[:20]}...{gemini_key[-10:]}")
        print(f"📝 Prompt length: {len(combined_prompt)} characters")

        response = requests.post(url, headers=headers, json=data)

        print(f"📡 Gemini Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]

                # Check response structure
                if 'content' in candidate and 'parts' in candidate['content']:
                    tweet = candidate['content']['parts'][0]['text'].strip()
                elif 'text' in candidate:
                    tweet = candidate['text'].strip()
                elif candidate.get('finishReason') == 'MAX_TOKENS':
                    print("⚠️ Gemini Pro hit MAX_TOKENS limit, switching to Flash...")
                    # Fallback to Flash model
                    url_flash = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
                    flash_response = requests.post(url_flash, headers=headers, json=data)

                    if flash_response.status_code == 200:
                        flash_result = flash_response.json()
                        if 'candidates' in flash_result and len(flash_result['candidates']) > 0:
                            flash_candidate = flash_result['candidates'][0]
                            if 'content' in flash_candidate and 'parts' in flash_candidate['content']:
                                tweet = flash_candidate['content']['parts'][0]['text'].strip()
                                print(f"✅ Flash fallback successful: {tweet[:50]}...")
                            else:
                                return None
                        else:
                            return None
                    else:
                        return None
                else:
                    print("❌ Unexpected Gemini Pro response structure")
                    print(f"Candidate keys: {candidate.keys()}")
                    print(f"FinishReason: {candidate.get('finishReason')}")
                    return None

                print(f"✅ Gemini Pro Tweet: {tweet}")

                # Show usage metadata
                if 'usageMetadata' in result:
                    usage = result['usageMetadata']
                    print(f"📊 Token usage: {usage.get('promptTokenCount', 0)} input + {usage.get('candidatesTokenCount', 0)} output = {usage.get('totalTokenCount', 0)} total")

                # Tweet cleaning process
                return clean_tweet(tweet, length_config, clean_project_name)
            else:
                print("⚠️ Gemini responded but no content")
                print(f"Response: {result}")
                return None

        elif response.status_code == 429:
            print("⚠️ Gemini rate limit! Wait a bit and try again.")
            return None
        else:
            print(f"❌ Gemini API error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Gemini API exception: {e}")
        return None

def get_enhanced_ai_tweet(project_key, sentiment_data, target_length, tweet_type, type_config):
    """Enhanced AI tweet - with pre-selected tweet type NATURAL HUMAN-LIKE + SMART OPENING"""
    import random
    project = projects[project_key]
    length_config = target_length

    # Select smart opening style based on tweet history
    opening_style = select_smart_opening_style(project_key)
    mention_count = get_project_mention_count(project_key)
    days_since = get_days_since_last_mention(project_key)

    print(f"🧠 Smart opening: {opening_style} (mention count: {mention_count}, last: {days_since} days ago)")

    # Set tone based on time (Feature #12)
    time_tone = get_time_based_tone()

    # Find project tweet for quote tweet
    quoted_tweet = None
    if tweet_type == "quote_commentary":
        quoted_tweet = find_recent_project_tweet(project_key)
        if not quoted_tweet:
            # If quote tweet not found, select fallback type
            tweet_type = random.choice(["tech_deep", "casual_discovery", "market_perspective"])

    # Smart opening phrases
    smart_openings = {
        "first_discovery": [
            "recently saw {clean_project_name}, looks interesting",
            "friend mentioned {clean_project_name}, checked it out",
            "came across {clean_project_name} by chance",
            "today I met {clean_project_name}",
            "randomly discovered {clean_project_name}",
            "hadn't heard before but {clean_project_name}"
        ],
        "recent_follow_up": [
            "mentioned {clean_project_name} before, current status",
            "re-examined {clean_project_name}",
            "there's an update about {clean_project_name}",
            "that {clean_project_name} I mentioned earlier",
            "{clean_project_name} has had innovations",
            "just looked at {clean_project_name} again"
        ],
        "rediscovery": [
            "hadn't looked at {clean_project_name} for a long time",
            "revisited {clean_project_name}",
            "forgot about {clean_project_name} for a while but",
            "rediscovered {clean_project_name}",
            "time has passed, how is {clean_project_name}",
            "returned to {clean_project_name}"
        ],
        "frequent_update": [
            "mentioning {clean_project_name} again",
            "recent status on {clean_project_name}",
            "{clean_project_name} is constantly on agenda",
            "once more about {clean_project_name}",
            "new development related to {clean_project_name}",
            "there's movement in {clean_project_name} again"
        ],
        "regular_check": [
            "I follow {clean_project_name} regularly",
            "current status on {clean_project_name}",
            "regular {clean_project_name} check",
            "reviewing {clean_project_name}",
            "continuing to track {clean_project_name}",
            "in my routine I have {clean_project_name}"
        ],
        "long_term_follow": [
            "{clean_project_name} I've been following for a long time",
            "recent developments about {clean_project_name}",
            "the {clean_project_name} adventure continues",
            "{clean_project_name} I've watched since old times",
            "the story of {clean_project_name}",
            "{clean_project_name} I discovered long ago"
        ]
    }

    # Prepare project name (replace underscores with spaces)
    clean_project_name = project['mention'].replace('@', '').replace('_', ' ').title()

    # Select smart opening
    selected_opening = random.choice(smart_openings.get(opening_style, smart_openings["first_discovery"]))
    selected_opening = selected_opening.format(clean_project_name=clean_project_name)

    type_prompts = {
        "tech_deep": f"""Write a {"long article style" if length_config['style'] == 'thread' else f"{length_config['min']}-{length_config['max']} character"} tweet about {clean_project_name}. Talk like a crypto person.

⚠️ FORMATTING: Put BLANK LINE between paragraphs!

SMART OPENING (MUST USE): "{selected_opening}"

PROJECT: {project['focus']} - {project['specialty']}
TECH: {project.get('tech_detail', '')}
INNOVATION: {project.get('key_innovation', '')}
STATUS: {project.get('development_stage', project['price_action'])}

{"LONG ARTICLE MODE (2000-3000 characters):" if length_config['style'] == 'thread' else ""}

TIME TONE: {time_tone['modifier']}

DON'T DO:
- "important for ecosystem", "should be considered" AI language
- "in-depth analysis", "professional approach" buzzwords  
{"- Too technical jargon, but give detailed explanation" if length_config['style'] == 'thread' else "- Too long sentences"}

DO:  
- Start with given opening: "{selected_opening}"
- "this technology is pretty cool", "could really work"  
- "still early but has potential", "that part is very cleverly done"
- Genuine, friendly tone - like you're explaining to a friend
{f"- Structure like article: Intro-Tech details-Use cases-Conclusion" if length_config['style'] == 'thread' else "- Short, clear sentences"}
{f"- Each paragraph focuses on separate topic" if length_config['style'] == 'thread' else ""}

TONE: {time_tone['tone']} + technically knowledgeable crypto person

EXAMPLE STRUCTURE:
"{selected_opening}. its technology is really different..."
"{selected_opening}, especially this part is very clever..."

Just write the tweet, no explanations.""",

        "casual_discovery": f"""Write {length_config['min']}-{length_config['max']} character casual tweet about {clean_project_name}.

⚠️ FORMATTING: Put BLANK LINE between paragraphs!

STATUS: {project.get('development_stage', project['price_action'])}
FEATURE: {project['specialty']}

SMART OPENING (MUST USE): "{selected_opening}"

IMPORTANT: Start tweet with this opening, then continue!

STYLE: Crypto enthusiast, genuine

DON'T:
- "caught my attention", "came across while researching" cliché openings  
- "should be examined", "important step" formal language
- Don't change the given opening!

DO:  
- Start with given opening: "{selected_opening}"
- Then continue: "pretty cool...", "looks interesting...", "not bad..."
- Use genuine tone

TONE: Genuine, curious
EXAMPLE STRUCTURES:
"{selected_opening}. that feature makes a lot of sense..."
"{selected_opening}, still new I guess but..."
"{selected_opening}. technology looks interesting..."

Just write the tweet.""",

        "market_perspective": f"""Write {length_config['min']}-{length_config['max']} character tweet about {clean_project_name} market status.

⚠️ FORMATTING: Put BLANK LINE between paragraphs!

STATUS: {project['token_status']} - {project['price_action']}
SECTOR: {project['ecosystem']}

DON'T:
- "market perspective", "analysis focus", "investment timing" 
- "volatility risk should be considered" AI language

DO:
- "now might be good timing for {clean_project_name}..."
- "token status not bad, but..."
- "sector overall active, {clean_project_name} too..."
- "still early but seems to have momentum..."

TONE: Market follower but unpretentious
EXAMPLES:
"X's token status not bad, sector active lately..."
"X still early but seems to have gained momentum..."
"Could make sense to look at X now, because..."

No risk warnings, get to the point. Write tweet.""",

        "comparison": f"""Write {length_config['min']}-{length_config['max']} character tweet comparing {clean_project_name} with other projects.

⚠️ FORMATTING: Put BLANK LINE between paragraphs!

PROJECT DIFFERENCE: {project.get('key_innovation', project['specialty'])}
FIELD: {project['focus']}

DON'T:
- "sectoral analysis", "comparative evaluation" formal language
- "superior to other solutions" exaggeration

DO:
- "{clean_project_name} different because..."
- "in this field usually done like this but {clean_project_name}..."
- "compared to classic methods {clean_project_name}..."
- "for example other projects do this, but this..."

TONE: Objective but curious comparison maker
EXAMPLES:
"In this field we usually see solutions like this but X follows different approach..."
"X's most interesting aspect is its departure from classic methods..."
"While other projects generally do this, X..."

Just write the tweet.""",

        "daily_metaphor": f"""Write a {"long article style" if length_config['style'] == 'thread' else f"{length_config['min']}-{length_config['max']} character"} tweet about {clean_project_name}. Explain technical topics with daily life metaphors.

⚠️ FORMATTING: Put BLANK LINE between paragraphs!

PROJECT: {project['focus']} - {project['specialty']}
TECH: {project.get('tech_detail', '')}
INNOVATION: {project.get('key_innovation', '')}

{"LONG ARTICLE MODE - Detailed explanation with daily life metaphors:" if length_config['style'] == 'thread' else ""}

STYLE: Use daily life metaphors like in examples

EXAMPLE STYLE (DO EXACTLY LIKE THIS):
1. "Anoma changes the traditional on-chain transaction model with its 'intent' focused structure. Users define what they want to do, how it's done is left to solvers..."

2. "In Anoma it's like this: You send an intent 'I want to get married'. Wedding hall, jeweler, marriage officer... you don't arrange these. Solvers come into play, match with suitable candidate, marriage happens onchain in one transaction 😄 No wedding costs, no mother-in-law pressure!"

DON'T:
- Ordinary technical explanation
- Too serious tone  
- Foreign references

DO THESE:
- Metaphors from daily life (marriage, family, relatives, neighbors, bazaar-market)
- Cultural references (TV shows, traditions, situations)
- Fun but educational explanations
- Start with "like:" "the thing is:" type genuine beginnings
- Use emojis (like 😄)
{f"- Use different daily life metaphor in each paragraph" if length_config['style'] == 'thread' else ""}
{f"- Tell like long story, create characters" if length_config['style'] == 'thread' else ""}

TONE: Fun teacher, explaining complex things with simple metaphors

CULTURAL REFERENCES:
- "Kismetse Olur", "Gelin Evi", "Kim Milyoner Olmak Ister"  
- "mother-in-law", "brother-in-law", "sister-in-law"
- "mukhtar's office", "coffeehouse", "grocery store", "tradesman"
- "holiday", "wedding", "engagement", "henna night"

DAILY LIFE METAPHORS:
- Marriage procedures (engagement, wedding, marriage)
- Family relationships (mother-in-law/daughter-in-law, brother-in-law/sister-in-law)
- Shopping (bargaining, deal making)
- Neighborhood (gossip, helping each other)

{f'''LONG ARTICLE STRUCTURE (With metaphors):

Intro - Compare project with familiar situation/person

(BLANK LINE)

Technical part - Explain complex tech with neighbor-relative relationships  

(BLANK LINE)

Advantages - Answer "What do we gain" with daily life examples

(BLANK LINE)

Competitors - Compare with other solutions (like comparing other houses/shops)

(BLANK LINE)

Conclusion - Fun prediction about future''' if length_config['style'] == 'thread' else ""}

Just write the tweet, explain with fun metaphors like this!""",

        "quote_commentary": f"""SPECIAL: This will be a quote tweet. Write {length_config['min']}-{length_config['max']} character tweet as if commenting on an official tweet from {clean_project_name}.

QUOTED TWEET: "{quoted_tweet['text'] if quoted_tweet else 'Project shared an update'}"
PROJECT FOCUS: {project['focus']}
FEATURE: {project['specialty']}

TIME TONE: {time_tone['modifier']}

SCENARIO: {clean_project_name} official account shared an update/announcement and you're commenting

DON'T:
- "I wrote quote tweet", "comment on that tweet" meta reference
- Too formal comment

DO:
- "this is good development because..."
- "exactly the news I expected, {clean_project_name}..."
- "interesting approach, especially this part..."
- "this is logical step for {clean_project_name}..."

TONE: Someone following project, knowledgeable but genuine + {time_tone['tone']}
EXAMPLES:
"This is good development, especially this feature makes sense..."
"Exactly the kind of update I was expecting..."
"X team really thought this through, this approach interesting..."

Write tweet as if quote tweeting.""",

        "experience_share": f"""Write a tweet of {length_config['min']}-{length_config['max']} characters about the {clean_project_name} experience.

STATUS: {project.get('development_stage', 'development_stage')}
WHAT'S AVAILABLE: {project['specialty']}

IMPORTANT: Write according to the REAL status!
- If "development" -> research/tracking experience
- If "testnet" -> testing experience
- If "mainnet" -> usage experience
- DO NOT SAY YOU'VE TRIED SOMETHING THAT DOESN'T EXIST!

DO NOT USE:
- Formal language like "experience sharing", "my personal evaluation"
- Claims of using features that are not available

DO USE:
- "I've been following {clean_project_name} for a while..."
- "While researching {clean_project_name}, I..."
- "My observation about {clean_project_name} so far is..."

TONE: Friendly, experienced but not exaggerated
EXAMPLES:
"Been following X for a while, the developments are not bad..."
"While researching X, I noticed that..."
"The impression I've gotten about X so far is..."

Write the tweet.""",

    "question_wonder": f"""Things you're curious about {clean_project_name} - {length_config['min']}-{length_config['max']} character tweet.

TECHNOLOGY: {project['focus']}
FEATURE: {project['specialty']}

DO NOT USE:
- Clichés like "topics I'm curious about", "thought-provoking questions"
- Overly technical questions

DO USE:
- "I wonder if {clean_project_name} really..."
- "I'm curious, {clean_project_name}..."
- "How does {clean_project_name} even work?"
- "I couldn't understand this part..."

TONE: Genuine curiosity, like a friend asking a question
EXAMPLES:
"I wonder if X can really solve this problem?"
"I'm curious how this feature of X works..."
"I didn't understand this about X..."

Just write the tweet.""",

    "future_prediction": f"""Future predictions for {clean_project_name} - {length_config['min']}-{length_config['max']} character tweet.

FIELD: {project['focus']}
INNOVATION: {project.get('key_innovation', project['specialty'])}

DO NOT USE:
- Jargon like "vision-focused analysis", "prediction areas"
- Exaggerations like "it will revolutionize in 2025"

DO USE:
- "I think {clean_project_name} in the future..."
- "In the coming period, {clean_project_name}..."
- "If this trend continues, {clean_project_name}..."
- "Projects like this in the future..."

TONE: Speculative but logical prediction
EXAMPLES:
"I think X will be talked about more next year..."
"If this trend continues, it could be good for X..."
"Projects like this will become more important in the future..."

Write the tweet."""
}

    prompt = type_prompts.get(tweet_type, type_prompts["casual_discovery"])

# AI API call - Gemini priority, OpenAI fallback
    if gemini_key:
    # Gemini API call
        result_tweet = call_gemini_api(prompt, length_config, clean_project_name)
        if result_tweet:
        # Tweet successful, update history
            update_project_mention_history(project_key, opening_style)
        return result_tweet
    elif openai_key:
        # ChatGPT API call (fallback)
        headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}

        # Uzun tweet'ler için daha fazla token (minimum 1000 karakter için)
        max_tokens_value = 3000 if length_config['style'] == 'thread' else 1500

        system_prompt = f"""You are a friendly person who follows crypto. You talk naturally on Twitter.

RULES:
- Write a tweet of {length_config['min']}-{length_config['max']} characters (minimum 1000 characters required)
- Use the name {clean_project_name} naturally
- Do not use @ symbols or hashtags
- Speak in a friendly, casual tone - as if explaining to a friend
- Provide a detailed and in-depth analysis
- Leave a BLANK LINE between paragraphs (for better visual appearance)

FORMATTING:
- Write each main idea in a separate paragraph
- Leave a blank line between paragraphs
- Break up long sentences to make them readable

DESIRED TONE: Crypto enthusiast, real person, not exaggerated but detailed"""


        if length_config['style'] == 'thread':
            system_prompt += f"""

SPECIAL: This is a long article-style tweet (4000-8000 characters)
- Perform a detailed analysis, use multiple paragraphs
- Explain technical topics in depth
- Use the Twitter Blue long tweet format
- Structure it like an article but maintain a friendly tone
- Use an introduction-body-conclusion structure
- Leave a BLANK LINE between each paragraph (very important!)
- You can use subheadings (with emojis)"""
        else:
            system_prompt += f"""

GOOD EXAMPLE SENTENCES:
"This feature of X seems quite logical"
"it's still early, but an interesting approach"
"we're used to seeing solutions like this in this space, but X is different"

Just write the tweet."""

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens_value,
            "temperature": 1.1
        }

        try:
            print(f"🤖 Making ChatGPT API call...")
            print(f"🔑 API Key start: {openai_key[:20]}..." if openai_key else "❌ API Key NOT FOUND!")
            print(f"📝 Prompt: {prompt[:100]}...")

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

            print(f"📡 API Response Status: {response.status_code}")
            print(f"📄 Response Headers: {dict(response.headers)}")

            if response.status_code == 200:
                result = response.json()
                tweet = result['choices'][0]['message']['content'].strip()

                print(f"✅ ChatGPT Tweet: {tweet}")
                cleaned_tweet = clean_tweet(tweet, length_config, clean_project_name)
                if cleaned_tweet:
                # Tweet successful, update history
                    update_project_mention_history(project_key, opening_style)
                return cleaned_tweet
            else:
                print(f"❌ OpenAI API error: {response.status_code}")
                print(f"❌ Response body: {response.text}")
            return None

        except Exception as e:
            print(f"❌ OpenAI request exception: {e}")
        return None
    else:
        print("❌ No AI API key found!")
        return None


def generate_tweet(system_prompt, openai_key, headers):
    # Define missing variables with placeholder values
    prompt = "Write a tweet about the feature of X."  # Example prompt
    length_config = 280  # Example max tweet length
    clean_project_name = True  # Example flag for cleaning project name
    project_key = "project_123"  # Example project key
    opening_style = "default"  # Example opening style
    max_tokens_value = 50  # Example max tokens for the API call

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens_value,
        "temperature": 1.1
    }

    try:
        print(f"🤖 Making ChatGPT API call...")
        print(f"🔑 API Key start: {openai_key[:20]}..." if openai_key else "❌ API Key NOT FOUND!")
        print(f"📝 Prompt: {prompt[:100]}...")

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

        print(f"📡 API Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            tweet = result['choices'][0]['message']['content'].strip()

            print(f"✅ ChatGPT Tweet: {tweet}")
            cleaned_tweet = clean_tweet(tweet, length_config, clean_project_name)
            if cleaned_tweet:
                # Tweet successful, update history
                update_project_mention_history(project_key, opening_style)
            return cleaned_tweet
        else:
            print(f"❌ OpenAI API error: {response.status_code}")
            print(f"❌ Response body: {response.text}")
            return None

    except Exception as e:
        print(f"❌ OpenAI request exception: {e}")
        return None
    return None

# The retry_chatgpt function was removed - there is no longer a fallback

def test_gemini_api():
    """Test the Gemini API"""
    if not gemini_key:
        print("⚠️ Gemini API key not found, skipping test")
        return False

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={gemini_key}"
    data = {
        "contents": [{
            "parts": [{
                "text": "Hello! This is a test. Just write 'Test successful'."
            }]
        }],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 20
        }
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                print(f"✅ Gemini API is working! Response: {text.strip()}")
                return True
            else:
                print(f"⚠️ Gemini API responded but without content")
                return False
        else:
            print(f"❌ Gemini API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Gemini API exception: {e}")
        return False

def test_openai_api():
    """Test the OpenAI API (fallback)"""
    if not openai_key:
        print("⚠️ OpenAI API key not found, skipping test")
        return False

    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            print(f"✅ OpenAI API is working!")
            return True
        else:
            print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ OpenAI API exception: {e}")
        return False

def test_twitter_api():
    """Test the Twitter API"""
    url = "https://api.twitter.com/2/users/me"
    auth_header = create_oauth_header("GET", url)
    headers = {"Authorization": auth_header}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        username = data.get('data', {}).get('username', 'unknown')
        print(f"✅ Twitter API is working! User: @{username}")
        return True
    else:
        print(f"❌ Twitter API error: {response.text}")
        return False

def send_tweet(content, quoted_tweet_id=None):
    """Send a tweet - with rate limiting (Quote tweet support)"""
    global last_tweet_time

    # Rate limiting check
    current_time = time.time()
    if last_tweet_time:
        time_since_last = current_time - last_tweet_time
        if time_since_last < MINIMUM_INTERVAL:
            wait_time = MINIMUM_INTERVAL - time_since_last
            print(f"⏳ Rate limiting: Need to wait {wait_time/60:.1f} minutes (2.5-hour rule)...")
            return False

    url = "https://api.twitter.com/2/tweets"

    # Use OAuth 1.0a (POST body is not included in the signature)
    auth_header = create_oauth_header("POST", url)
    headers = {
        "Authorization": auth_header, 
        "Content-Type": "application/json"
    }

    # Prepare tweet data
    data = {"text": content}

    # Add quoted_tweet_id if it's a quote tweet
    if quoted_tweet_id:
        data["quote_tweet_id"] = quoted_tweet_id
        print(f"💬 Sending quote tweet: {quoted_tweet_id}")

    print(f"🔐 Auth Header: {auth_header[:50]}...")
    print(f"📤 Tweet Data: {data}")
    print(f"🌐 Request URL: {url}")
    print(f"📡 Request Headers: {headers}")

    response = requests.post(url, headers=headers, json=data)

    print(f"📊 Response Status: {response.status_code}")
    print(f"📄 Response Text: {response.text}")
    print(f"📋 Response Headers: {dict(response.headers)}")

    if response.status_code == 201:
        result = response.json()
        tweet_id = result['data']['id']
        last_tweet_time = current_time  # Update time after a successful tweet
        print(f"✅ Tweet sent!")
        print(f"📝 Content: {content}")
        print(f"🔗 Tweet ID: {tweet_id}")
        if quoted_tweet_id:
            print(f"💬 Quote Tweet ID: {quoted_tweet_id}")
        print(f"📊 Length: {len(content)} characters")
        return True
    elif response.status_code == 429:
        print(f"⚠️ Twitter API rate limit! Waiting for 2.5 hours...")
        print("🔄 The bot will wait automatically and try later")
        return False
    else:
        print(f"❌ Error sending tweet: {response.text}")
        return False

def get_recent_tweets():
    """Read own recent tweets - check which projects have been mentioned and from what angles"""
    try:
        url = "https://api.twitter.com/2/users/me/tweets"
        params = {
            'max_results': 20,  # Last 20 tweets
            'tweet.fields': 'created_at,text'
        }

        # For OAuth 1.0a, GET parameters must be included in the signature
        auth_header = create_oauth_header("GET", url, params)
        headers = {"Authorization": auth_header}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            tweets = data.get('data', [])

            project_tweet_types = {}  # {project_key: [tweet_types]}
            for tweet in tweets:
                text = tweet['text'].lower()

                # Check which projects were mentioned
                for project_key, project_data in projects.items():
                    mention = project_data['mention'].lower()
                    if mention in text:
                        # Guess the tweet type (simple keyword analysis)
                        detected_type = detect_tweet_type(text)

                        if project_key not in project_tweet_types:
                            project_tweet_types[project_key] = []
                        project_tweet_types[project_key].append(detected_type)

            print(f"📊 Analysis of last 20 tweets: {len(tweets)} tweets")
            for project, types in project_tweet_types.items():
                print(f"   🎯 {project}: {types}")

            return project_tweet_types
        else:
            print(f"⚠️ Could not read tweet history: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error reading tweet history: {e}")
        return {}

def detect_tweet_type(text):
    """Guess the type of the tweet from its content"""
    # Simple keyword analysis
    if any(word in text for word in ['how it works', 'technology', 'protocol', 'algorithm', 'architecture']):
        return 'tech_deep_dive'
    elif any(word in text for word in ['new', 'discovered', 'came across', 'first time']):
        return 'casual_discovery' 
    elif any(word in text for word in ['price', 'airdrop', 'token', 'investment', 'market']):
        return 'market_perspective'
    elif any(word in text for word in ['compare', 'versus', 'difference', 'similar']):
        return 'comparison'
    elif any(word in text for word in ['tried', 'used', 'testnet', 'experience']):
        return 'experience_share'
    elif any(word in text for word in ['i wonder', 'curious', 'how', 'why']):
        return 'question_wonder'
    elif any(word in text for word in ['future', '2025', 'will be huge', 'potential']):
        return 'future_prediction'
    else:
        return 'casual_discovery'  # Default

# NEWS MONITORING SYSTEM COMPLETELY REMOVED
# Removed functions:
# - get_crypto_news()
# - calculate_news_relevance()
# - get_trending_topics()
# - find_related_project()
# - create_news_based_tweet()
# - create_trend_based_tweet()
# - news/trends/newstweet/trendtweet commands

# News and trend functions have been removed

def get_tweet_performance(tweet_id):
    """Check tweet performance - for analytics"""
    try:
        url = f"https://api.twitter.com/2/tweets/{tweet_id}"
        params = {
            'tweet.fields': 'public_metrics,created_at',
            'expansions': 'author_id'
        }

        auth_header = create_oauth_header("GET", url, params)
        headers = {"Authorization": auth_header}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            tweet_data = data.get('data', {})
            metrics = tweet_data.get('public_metrics', {})

            performance = {
                'tweet_id': tweet_id,
                'likes': metrics.get('like_count', 0),
                'retweets': metrics.get('retweet_count', 0),
                'replies': metrics.get('reply_count', 0),
                'quotes': metrics.get('quote_count', 0),
                'impressions': metrics.get('impression_count', 0),  # If available
                'engagement_rate': 0,
                'check_time': datetime.now().isoformat()
            }

            # Calculate simple engagement rate
            total_engagement = performance['likes'] + performance['retweets'] + performance['replies'] + performance['quotes']
            if performance['impressions'] > 0:
                performance['engagement_rate'] = (total_engagement / performance['impressions']) * 100

            return performance
        else:
            print(f"⚠️ Could not read tweet performance: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Tweet analytics error: {e}")
        return None

def save_tweet_analytics(tweet_id, content, project_key, tweet_type):
    """Save tweet analytics to a file"""
    try:
        analytics_file = "tweet_analytics.json"

        # Read the existing file
        try:
            with open(analytics_file, 'r', encoding='utf-8') as f:
                analytics_data = json.load(f)
        except FileNotFoundError:
            analytics_data = {}

        # New tweet record
        analytics_data[tweet_id] = {
            'content': content,
            'project': project_key,
            'tweet_type': tweet_type,
            'sent_time': datetime.now().isoformat(),
            'initial_metrics': None,  # For the first check
            'day_1_metrics': None,    # After 24 hours
            'day_7_metrics': None     # After 7 days
        }

        # Save to file
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics_data, f, ensure_ascii=False, indent=2)

        print(f"📊 Tweet analytics saved: {tweet_id}")

    except Exception as e:
        print(f"❌ Error saving analytics: {e}")

def check_pending_analytics():
    """Check for pending tweet analytics"""
    try:
        analytics_file = "tweet_analytics.json"

        try:
            with open(analytics_file, 'r', encoding='utf-8') as f:
                analytics_data = json.load(f)
        except FileNotFoundError:
            return  # Nothing to do if the file doesn't exist

        current_time = datetime.now()
        updated = False

        for tweet_id, data in analytics_data.items():
            sent_time = datetime.fromisoformat(data['sent_time'])
            time_diff = current_time - sent_time

            # First check after 1 hour (initial metrics)
            if time_diff.total_seconds() > 3600 and data['initial_metrics'] is None:
                metrics = get_tweet_performance(tweet_id)
                if metrics:
                    data['initial_metrics'] = metrics
                    updated = True
                    print(f"📈 After 1 hour: {tweet_id} - {metrics['likes']} likes, {metrics['retweets']} RTs")

            # Check after 24 hours
            if time_diff.total_seconds() > 86400 and data['day_1_metrics'] is None:
                metrics = get_tweet_performance(tweet_id)
                if metrics:
                    data['day_1_metrics'] = metrics
                    updated = True
                    print(f"📈 After 24 hours: {tweet_id} - {metrics['likes']} likes, {metrics['retweets']} RTs")

            # Check after 7 days
            if time_diff.total_seconds() > 604800 and data['day_7_metrics'] is None:
                metrics = get_tweet_performance(tweet_id)
                if metrics:
                    data['day_7_metrics'] = metrics
                    updated = True
                    print(f"📈 After 7 days: {tweet_id} - {metrics['likes']} likes, {metrics['retweets']} RTs")

        # Save updated data
        if updated:
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics_data, f, ensure_ascii=False, indent=2)
            print("📊 Analytics updated!")

    except Exception as e:
        print(f"❌ Analytics check error: {e}")

def create_thread_content(project_key, sentiment_data):
    """Create thread content - for long-form analysis"""
    try:
        project = projects[project_key]

        # Custom prompt for the thread
        thread_prompt = f"""You are an expert who writes long-form crypto analyses. You will write a 2-3 tweet thread about {project['mention'].replace('@', '').replace('_', ' ').title()}.

Project Info:
- Focus: {project['focus']}
- Specialty: {project['specialty']}
- Tech Detail: {project.get('tech_detail', '')}
- Ecosystem: {project['ecosystem']}

Thread Structure:
Tweet 1 (Main): Introduce the project and grab attention (280-450 characters)
Tweet 2 (In-depth): Technical details and use cases (280-450 characters)
Tweet 3 (Conclusion): Vision and evaluation (280-450 characters)

Separate each tweet with tags [TWEET1], [TWEET2], [TWEET3].

Writing Rules:
- Do not use the @ symbol
- Avoid filler words like "well, so, like, I think"
- Do not use hashtags
- Each tweet should make sense on its own
- Use the project name naturally
- Fluent, conversational yet professional

Start with a thread title."""

        headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": thread_prompt}
            ],
            "max_tokens": 800,
            "temperature": 1.0
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            thread_content = result['choices'][0]['message']['content'].strip()

            # Parse the thread
            tweets = []
            for i in range(1, 4):
                tag = f"[TWEET{i}]"
                if tag in thread_content:
                    start = thread_content.find(tag) + len(tag)
                    if i < 3:
                        end = thread_content.find(f"[TWEET{i+1}]")
                        tweet_text = thread_content[start:end].strip()
                    else:
                        tweet_text = thread_content[start:].strip()

                    # Clean the tweet text
                    tweet_text = clean_tweet_text(tweet_text)
                    if tweet_text and len(tweet_text) > 50:  # If not too short
                        tweets.append(tweet_text)

            if len(tweets) >= 2:
                print(f"🧵 Thread created: {len(tweets)} tweets")
                return tweets
            else:
                print(f"⚠️ Could not parse thread, falling back to a normal tweet")
                return None

        else:
            print(f"❌ Thread AI error: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Thread creation error: {e}")
        return None

def clean_tweet_text(text):
    """Clean tweet text - remove @ mentions, hashtags, etc."""
    import re

    # Replace @ mentions with the project name
    for project_key, project_data in projects.items():
        mention = project_data['mention']
        project_name = mention.replace('@', '').replace('_', ' ').title()
        variations = [
            mention,  # @campnetworkxyz
            mention.replace('@', ''),  # campnetworkxyz
            mention.replace('@', '').lower(),  # campnetworkxyz
            mention.replace('@', '').capitalize(),  # Campnetworkxyz
            mention.replace('@', '').upper()  # CAMPNETWORKXYZ
        ]
        for var in variations:
            text = text.replace(var, project_name)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Unwanted phrases
    unwanted_phrases = [
        "this ", "well ", "I think how", "I think", 
        "how so", "so how", "this project", "that project"
    ]
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def send_thread(tweets):
    """Send a thread - each tweet in a chain"""
    try:
        if not tweets or len(tweets) < 2:
            return False

        thread_ids = []
        reply_to_id = None

        for i, tweet_content in enumerate(tweets):
            print(f"🧵 Thread {i+1}/{len(tweets)}: {tweet_content[:50]}...")

            # Prepare tweet data
            tweet_data = {"text": tweet_content}
            if reply_to_id:
                tweet_data["reply"] = {"in_reply_to_tweet_id": reply_to_id}

            # Send tweet
            url = "https://api.twitter.com/2/tweets"
            auth_header = create_oauth_header("POST", url)
            headers = {
                "Authorization": auth_header,
                "Content-Type": "application/json"
            }

            response = requests.post(url, headers=headers, json=tweet_data)

            if response.status_code == 201:
                result = response.json()
                tweet_id = result['data']['id']
                thread_ids.append(tweet_id)
                reply_to_id = tweet_id  # The next tweet will reply to this one

                print(f"✅ Thread {i+1} sent: {tweet_id}")

                # Wait 2 seconds between tweets in a thread
                if i < len(tweets) - 1:
                    time.sleep(2)
            else:
                print(f"❌ Thread {i+1} error: {response.text}")
                return False

        print(f"🎉 Thread completed! {len(thread_ids)} tweets")
        return thread_ids

    except Exception as e:
        print(f"❌ Thread sending error: {e}")
        return False

def check_mentions_and_reply():
    """Check mentions and auto-reply"""
    try:
        # Get recent mentions
        url = "https://api.twitter.com/2/users/me/mentions"
        params = {
            'max_results': 10,
            'tweet.fields': 'created_at,text,author_id,conversation_id',
            'expansions': 'author_id'
        }

        auth_header = create_oauth_header("GET", url, params)
        headers = {"Authorization": auth_header}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            mentions = data.get('data', [])
            users_data = data.get('includes', {}).get('users', [])

            # User ID to username mapping
            user_mapping = {user['id']: user['username'] for user in users_data}

            for mention in mentions:
                tweet_id = mention['id']
                text = mention['text'].lower()
                author_id = mention['author_id']
                username = user_mapping.get(author_id, 'unknown')

                # Check if this mention has already been replied to
                if check_if_already_replied(tweet_id):
                    continue

                # Simple reply logic - respond to crypto questions
                reply_content = generate_auto_reply(text, username)

                if reply_content:
                    success = send_reply(tweet_id, reply_content)
                    if success:
                        mark_as_replied(tweet_id)
                        print(f"💬 Auto-replied to user @{username}")

        else:
            print(f"⚠️ Could not read mentions: {response.status_code}")

    except Exception as e:
        print(f"❌ Mention check error: {e}")

def generate_auto_reply(mention_text, username):
    """Generate an automatic reply to a mention"""
    try:
        # Simple reply rules
        crypto_keywords = ['anoma', 'mitosis', 'union', 'virtuals', 'camp', 'somnia', 'pharos', 'zama', 'crypto', 'blockchain', 'defi']

        # Check if it's crypto-related
        if any(keyword in mention_text for keyword in crypto_keywords):

            # Project-specific questions
            for project_key, project_data in projects.items():
                if project_key in mention_text or project_data['mention'].replace('@', '') in mention_text:
                    replies = [
                        f"Hello! {project_data['mention'].replace('@', '').replace('_', ' ').title()} is a really interesting project. It stands out in the {project_data['focus']} space.",
                        f"I recommend doing more detailed research on this project. It's quite valuable in terms of its {project_data['specialty']}.",
                        f"Definitely a project worth following! It's a great contribution to the {project_data['ecosystem']} ecosystem."
                    ]
                    return random.choice(replies)

            # General crypto replies
            general_replies = [
                "Doing your own research in the crypto world is really important. Never forget the DYOR principle!",
                "I suggest you research this topic in more detail. It's good to check the projects' roadmaps.",
                "Interesting question! There are always new developments in the crypto space, you have to keep up.",
                "When researching such projects, don't forget to look at the tokenomics and the team!"
            ]
            return random.choice(general_replies)

        return None  # Don't reply if it's not crypto-related

    except Exception as e:
        print(f"❌ Error generating auto-reply: {e}")
        return None

def send_reply(tweet_id, content):
    """Send a reply to a tweet"""
    try:
        url = "https://api.twitter.com/2/tweets"
        tweet_data = {
            "text": content,
            "reply": {"in_reply_to_tweet_id": tweet_id}
        }

        auth_header = create_oauth_header("POST", url)
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=tweet_data)

        if response.status_code == 201:
            print(f"✅ Reply sent: {content[:50]}...")
            return True
        else:
            print(f"❌ Error sending reply: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error sending reply: {e}")
        return False

def check_if_already_replied(tweet_id):
    """Check if a reply has already been sent to this tweet"""
    try:
        replies_file = "replied_tweets.json"

        try:
            with open(replies_file, 'r', encoding='utf-8') as f:
                replied_data = json.load(f)
        except FileNotFoundError:
            replied_data = []

        return tweet_id in replied_data

    except Exception as e:
        print(f"❌ Reply check error: {e}")
        return False

def mark_as_replied(tweet_id):
    """Mark a tweet as replied to"""
    try:
        replies_file = "replied_tweets.json"

        try:
            with open(replies_file, 'r', encoding='utf-8') as f:
                replied_data = json.load(f)
        except FileNotFoundError:
            replied_data = []

        if tweet_id not in replied_data:
            replied_data.append(tweet_id)

            # Keep the last 100 tweets (so the file doesn't get too big)
            if len(replied_data) > 100:
                replied_data = replied_data[-100:]

            with open(replies_file, 'w', encoding='utf-8') as f:
                json.dump(replied_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"❌ Error marking as replied: {e}")

# ENHANCED send_tweet function - with analytics and quote tweet support
def send_tweet_with_analytics(content, project_key=None, tweet_type=None, quoted_tweet_id=None):
    """Send a tweet and save analytics (with quote tweet support)"""
    global last_tweet_time

    # Rate limiting check
    current_time = time.time()
    if last_tweet_time:
        time_since_last = current_time - last_tweet_time
        if time_since_last < MINIMUM_INTERVAL:
            wait_time = MINIMUM_INTERVAL - time_since_last
            print(f"⏳ Rate limiting: Need to wait {wait_time/60:.1f} minutes (2.5-hour rule)...")
            return False

    url = "https://api.twitter.com/2/tweets"

    auth_header = create_oauth_header("POST", url)
    headers = {
        "Authorization": auth_header, 
        "Content-Type": "application/json"
    }
    data = {"text": content}

    # Add quoted_tweet_id if it's a quote tweet
    if quoted_tweet_id:
        data["quote_tweet_id"] = quoted_tweet_id
        print(f"💬 Sending quote tweet: {quoted_tweet_id}")

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 201:
        result = response.json()
        tweet_id = result['data']['id']
        last_tweet_time = current_time

        print(f"✅ Tweet sent!")
        print(f"📝 Content: {content}")
        print(f"🔗 Tweet ID: {tweet_id}")
        if quoted_tweet_id:
            print(f"💬 Quote Tweet ID: {quoted_tweet_id}")
        print(f"📊 Length: {len(content)} characters")

        # Save analytics
        if project_key and tweet_type:
            save_tweet_analytics(tweet_id, content, project_key, tweet_type)

        return tweet_id  # Return the Tweet ID
    elif response.status_code == 429:
        print(f"⚠️ Twitter API rate limit! Waiting for 2.5 hours...")
        return False
    else:
        print(f"❌ Error sending tweet: {response.text}")
        return False

# ENHANCED create_enhanced_tweet - with thread and analytics support
def create_enhanced_tweet_v2():
    """Enhanced tweet v2 - with thread and analytics support"""
    try:
        # Analyze recent tweets
        project_tweet_types = get_recent_tweets()

        # Select project and tweet type
        project_keys = list(projects.keys())
        all_tweet_types = list(TWEET_TYPES.keys())

        import random

        # Project selection (previous logic)
        unused_projects = [p for p in project_keys if p not in project_tweet_types]

        if unused_projects:
            selected_project = random.choice(unused_projects)
            available_types = all_tweet_types
            print(f"🎯 New project selected: {selected_project} (never mentioned)")
        else:
            project_counts = {p: len(types) for p, types in project_tweet_types.items()}
            selected_project = min(project_counts.keys(), key=lambda x: project_counts[x])

            used_types = project_tweet_types.get(selected_project, [])
            available_types = [t for t in all_tweet_types if t not in used_types]

            if not available_types:
                from collections import Counter
                type_counts = Counter(used_types)
                available_types = [min(all_tweet_types, key=lambda x: type_counts.get(x, 0))]

            print(f"🎯 Existing project selected: {selected_project}")

        selected_type = random.choice(available_types)
        type_config = TWEET_TYPES[selected_type]

        sentiment_data = search_twitter_sentiment(selected_project)

        print(f"🎯 Selected project: {projects[selected_project]['mention']} - {projects[selected_project]['focus']}")
        print(f"🎭 Tweet type: {selected_type} - {type_config['style']}")

        # Direct long tweet mode - up to 4000 chars with Twitter Blue
        length_config = choose_tweet_length()

        if length_config['style'] == 'thread':
            print("📝 Long article tweet mode - direct long tweet with Twitter Blue...")

        tweet_content = get_enhanced_ai_tweet(selected_project, sentiment_data, length_config, selected_type, type_config)

        if tweet_content is None:
            print("❌ Could not create tweet with ChatGPT! Skipping this run.")
            return False

        print(f"💬 Tweet ready: {tweet_content}")
        print(f"📊 Length: {len(tweet_content)} characters")

        # Get tweet ID for quote tweet
        quoted_tweet_id = None
        if selected_type == "quote_commentary":
            quoted_tweet = find_recent_project_tweet(selected_project)
            if quoted_tweet:
                quoted_tweet_id = quoted_tweet['id']
                print(f"💬 Found quote tweet: {quoted_tweet['text'][:50]}...")

        # Send the tweet
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            print("🧪 TEST MODE: Not sending tweet, just creating it!")
            if quoted_tweet_id:
                print(f"💬 TEST: Would have been a quote tweet of: {quoted_tweet_id}")
            success = True
        else:
            tweet_id = send_tweet_with_analytics(tweet_content, selected_project, selected_type, quoted_tweet_id)
            success = bool(tweet_id)

        if success:
            print("🎉 Tweet successfully sent!")
        else:
            print("❌ Failed to send tweet!")

        return success

    except Exception as e:
        print(f"❌ Enhanced tweet v2 error: {e}")
        return False

# ENHANCED auto_tweet function
def auto_tweet_v2():
    """Enhanced automatic tweet v2 - with analytics and community features"""
    current_time = datetime.now()
    current_hour = current_time.hour

    print(f"⏰ {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Starting enhanced automatic tweet v2...")

    # Check analytics (on every run)
    print("📊 Checking tweet analytics...")
    check_pending_analytics()

    # Check for community interaction (30% probability)
    if random.randint(1, 100) <= 30:
        print("💬 Checking for mentions...")
        check_mentions_and_reply()

    # Time check - only tweet between 10:00 and 22:00
    if current_hour < TWEET_START_HOUR or current_hour >= TWEET_END_HOUR:
        print(f"🌙 Night time ({current_hour}:00) - Not tweeting (only between {TWEET_START_HOUR}:00-{TWEET_END_HOUR}:00)")
        return False

    # Rate limiting check
    global last_tweet_time
    if last_tweet_time:
        time_since_last = time.time() - last_tweet_time
        if time_since_last < MINIMUM_INTERVAL:
            wait_time = MINIMUM_INTERVAL - time_since_last
            print(f"⏳ Rate limiting active: should wait another {wait_time/60:.1f} minutes (2.5-hour rule)")
            return False

    success = create_enhanced_tweet_v2()
    if success:
        print("✅ Enhanced automatic tweet v2 completed!")
    else:
        print("❌ Enhanced automatic tweet v2 failed!")

    return success

def main():
    """Main function"""
    keep_alive() 
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("🧪 Test mode - Single tweet (Enhanced v2)")
        create_enhanced_tweet_v2()
    elif len(sys.argv) > 1 and sys.argv[1] == "quote":
        print("💬 Quote tweet test")
        # Specific quote tweet test
        project_key = list(projects.keys())[0]  # Select the first project
        quoted_tweet = find_recent_project_tweet(project_key)
        if quoted_tweet:
            print(f"✅ Found quote tweet: {quoted_tweet['text']}")
            # Test prompt
            sentiment_data = search_twitter_sentiment(project_key)
            length_config = choose_tweet_length()
            time_tone = get_time_based_tone()

            tweet_content = get_enhanced_ai_tweet(project_key, sentiment_data, length_config, "quote_commentary", TWEET_TYPES["quote_commentary"])
            if tweet_content:
                print(f"💬 Quote tweet content: {tweet_content}")
                print(f"📋 Original tweet: {quoted_tweet['text']}")
            else:
                print("❌ Could not create quote tweet")
        else:
            print("❌ Project tweet not found")

    elif len(sys.argv) > 1 and sys.argv[1] == "time":
        print("⏰ Time-based tone test")
        # Test different time tones manually
        for hour in [8, 14, 19, 23]:
            current_hour = hour

            if 6 <= current_hour < 12:
                time_tone = {
                    "tone": "energetic_morning",
                    "modifier": "With morning energy, positive and motivational"
                }
            elif 12 <= current_hour < 17:
                time_tone = {
                    "tone": "analytical_noon", 
                    "modifier": "Analytical approach for noon, more detailed"
                }
            elif 17 <= current_hour < 22:
                time_tone = {
                    "tone": "casual_evening",
                    "modifier": "Relaxed evening atmosphere, more friendly and sharing"
                }
            else:
                time_tone = {
                    "tone": "chill_night",
                    "modifier": "Calm of the night, thoughtful and in-depth"
                }

            print(f"Hour {hour}:00 - {time_tone['tone']}: {time_tone['modifier']}")
    elif len(sys.argv) > 1 and sys.argv[1] == "analytics":
        print("📊 Analytics report")
        check_pending_analytics()
    elif len(sys.argv) > 1 and sys.argv[1] == "mentions":
        print("💬 Check mentions")
        check_mentions_and_reply()
# News system commands removed
    else:
        print("🤖 Enhanced Bot v2 mode")
        # Enhanced auto_bot
        print("🤖 Starting Enhanced Kaito Twitter Bot v4.2...")

        # API tests
        if not test_twitter_api():
            print("❌ Twitter API connection failed! Stopping the bot.")
            return

        # AI API tests - Gemini priority
        ai_working = False
        if gemini_key:
            if test_gemini_api():
                print("✅ Gemini API is working! Will be used as Primary AI.")
                ai_working = True
            else:
                print("⚠️ Gemini API is not working, switching to OpenAI...")

        if not ai_working and openai_key:
            if test_openai_api():
                print("✅ OpenAI API is working! Will be used as Fallback AI.")
                ai_working = True
            else:
                print("❌ OpenAI API is also not working!")

        if not ai_working:
            print("❌ No AI API is working! Stopping the bot.")
            return
        else:
            print("✅ AI API is ready!")

        print("⏰ First tweet is waiting in the schedule (for rate limiting safety)")

        # Enhanced schedule
        def scheduled_tweet_v2():
            print("📅 Enhanced Schedule check - trying to tweet...")
            return auto_tweet_v2()

        # Check every 30 minutes
        schedule.every(30).minutes.do(scheduled_tweet_v2)

        print("⏰ Gemini Enhanced Bot schedule has been set:")
        print("   🧠 Gemini 2.5 Pro PRIMARY AI (most powerful + 2M tokens)")
        print("   📊 Every 30min: Analytics check, mention reply (30%)")
        print("   📈 Automatic: Tweet performance tracking")
        print("   🎯 Quote tweet, Detailed analysis focus, Time-based tone")
        print("   🤖 Gemini tweet mode (OpenAI fallback)")
        print("🔄 Gemini Enhanced Bot has started! Stop with Ctrl+C.")
        print("\nTest commands:")
        print("   python bot.py test    - Normal tweet test")
        print("   python bot.py quote   - Quote tweet test")
        print("   python bot.py time    - Time tone test")

        # Infinite loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                print("\n⏹️ Enhanced Bot stopped!")
                break
            except Exception as e:
                print(f"❌ Enhanced Bot error: {e}")
                time.sleep(300)

if __name__ == "__main__":
    main()
