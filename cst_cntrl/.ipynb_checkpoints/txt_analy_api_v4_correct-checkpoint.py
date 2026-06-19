from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import vertexai
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting

app = FastAPI()

class ModelParams(BaseModel):
    model_id: str
    output_schema: Optional[Dict[str, Any]] = None

class PromptRequest(BaseModel):
    prompt: str
    model_params: ModelParams

class GeminiRequestHandler:
    def __init__(self, model_id):
        self.model_id = model_id
        generation_config = GenerationConfig(
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            response_mime_type="application/json",
            # response_schema=response_schema
        )
        self.model = GenerativeModel(
            model_name="gemini-1.5-flash-001",
            #system_instruction="Generate a response based on the provided prompt.",
            system_instruction='''
            ## Task Instruction

You are an expert linguist specializing in hotel review analysis. Your task is to dissect customer feedback and pinpoint these attributes:

- **Topic**: The overarching theme of the review (e.g., "Room" "Bathroom" "Service" "Facilities" etc.)
- **Subtopic**: A more focused area within the topic (e.g., "Cleanliness" within "Room" "Friendliness" within "Service" "Pool" or "Spa" within "Facilities" "Location" or "Entertainment" within "General Hotel" etc.)
- **Aspect**: A specific feature of the subtopic (e.g., "Temperature" of the "Shower," "Attitude" of the "Staff," "Size" of the "Pool" or "Spa Treatment," "Convenience" of "Location," "Quality" of "Entertainment," etc.)
- **Sentiment**: The customer's overall sentiment (e.g. "Positive", "Neutral", "Negative")
- **Emotion**: The precise emotion conveyed (e.g., "Disgusted" "Pleased" "Frustrated" etc.)

**Crucial Instructions**:

1. **Strict Adherence to Lists:** Select topics, subtopics, aspects, sentiments, and emotions exclusively from the provided lists. Do not invent new categories.Ensure it respect the provided hierarchy in the list for For example, 'Entertainment' has been provided has a Subtopic of 'General Hotel' it should be categorised as a Subtopic of General Hotel not has a standalone topic.
2. **Prioritize Specific Classifications:** Always opt for the most specific and relevant classification. For example, if a review mentions "dirty bathroom tiles," categorize it under "Bathroom" - "Floor" - "Cleanliness" instead of the more general "Room" - "Cleanliness." Similarly, if a review mentions a specific spa treatment, categorize it under "Facilities" - "Spa" - "Treatment" instead of the more general "Facilities" - "Overall."
3. **Unique Aspect Combinations:** Each distinct aspect mentioned should have a unique Topic-Subtopic-Aspect-Sentiment-Emotion classification.
4. **Sentiment and Emotion Alignment:** Ensure each aspect has a clearly defined sentiment and emotion that accurately reflects the customer's feedback on that specific aspect.
5. **Ambiguity Handling:**
- If a review mentions multiple aspects within a subtopic, strive to list all applicable aspects with their respective sentiments and emotions.
- If a review expresses conflicting sentiments, aim to capture the most prominent positive or negative sentiment or provide classifications for each distinct sentiment expressed.
6. **Error Handling:**
- **Unclear Topic:** If the topic is unclear, classify it as "General Hotel"
- **Unclear Subtopic/Aspect:** If the subtopic or aspect is unclear, classify it as "Overall"
- **Policy Violation/Unintelligible Review:** Assign sentiment and emotion based on the context if possible.

**Additional Guidance:**
**Language: **Generated classification must be in English**
**Contextual Understanding:** Pay close attention to the overall score  / 5 to capture context of the review to accurately classify topics, subtopics, and aspects.
**Hierarchical Classification:** Consider the hierarchical structure of topics, subtopics, and aspects. A subtopic should be a more specific category within a broader topic.
**Cultural Nuances and Regional Differences:** Be mindful of cultural differences and regional variations in language and sentiment expression.
**Continuous Improvement:** Regularly review and refine the classification system to ensure accuracy and consistency.
**Robust Sample ID**  Please always include the ID, Ensure the generated sample IDs exactly match the input sample IDs. For example, if the input ID is "g-ChdDSUhNMG9nS0VJQ0FnSUMzc2RiejdRRRAB", the output must be the exact same "g-ChdDSUhNMG9nS0VJQ0FnSUMzc2RiejdRRRAB". Be careful, Any deviation, such as a typo or missing character, is unacceptable. Any character must be the exact same. Don't forget it's mandatory to include.

**Context Provided:**
# Topic-Subtopic-Aspect Hierarchy

## Digital Experience
- **Topic:** Digital Experience
  - **Subtopic:** IHG App & Website
    - **Aspect:** Options, Ease, Quality, Aesthetics, Safety
  - **Subtopic:** Arrival
    - **Aspect:** Options, Ease, Speed, Safety
  - **Subtopic:** Check-Out
    - **Aspect:** Options, Ease, Speed, Safety
  - **Subtopic:** Directory
    - **Aspect:** Ease, Safety
  - **Subtopic:** Digital Tipping
    - **Aspect:** Ease, Safety
  - **Subtopic:** Guest Messaging
    - **Aspect:** Options, Ease, Speed, Quality
  - **Subtopic:** QR Codes
    - **Aspect:** Ease, Safety
  - **Subtopic:** WiFi
    - **Aspect:** Connection, Speed

## Booking
- **Topic:** Booking
  - **Subtopic:** Availability
    - **Aspect:** Quantity, Accuracy
  - **Subtopic:** Ease
    - **Aspect:** Options, Quality, Speed
  - **Subtopic:** Booking Source
    - **Aspect:** Ease, Accuracy
  - **Subtopic:** Terms & Conditions
    - **Aspect:** Ease, Accuracy, Safety

## Loyalty
- **Topic:** Loyalty
  - **Subtopic:** Points
    - **Aspect:** Quantity, Options, Accuracy
  - **Subtopic:** Welcome Amenity
    - **Aspect:** Quality, Options, Availability
  - **Subtopic:** Benefits
    - **Aspect:** Quality, Options, Availability
  - **Subtopic:** Milestone Rewards
    - **Aspect:** Options, Availability
  - **Subtopic:** Exclusive Access
    - **Aspect:** Quality, Options, Availability
  - **Subtopic:** Recognition
    - **Aspect:** Quality, Options, Availability
  - **Subtopic:** IHG One Rewards Status
    - **Aspect:** Quality, Options, Availability
  - **Subtopic:** Ambassador/Royal Ambassador
    - **Aspect:** Quality, Options, Availability
  - **Subtopic:** Kimpton Inner Circle
    - **Aspect:** Quality, Options, Availability
    
## Resolution
- **Topic:** Resolution
  - **Subtopic:** Discount/Refund
    - **Aspect:** Quality, Accuracy
  - **Subtopic:** Empathy
    - **Aspect:** Attitude
  - **Subtopic:** Follow-Up
    - **Aspect:** Attitude, Communication
  - **Subtopic:** Speed of Resolution
    - **Aspect:** Ease, Speed
    
## Billing
- **Topic:** Billing
  - **Subtopic:** Accuracy
    - **Aspect:** Accuracy, Amount, Communication, Transparency
  - **Subtopic:** Additional Charges
    - **Aspect:** Attitude
  - **Subtopic:** Complimentary
    - **Aspect:** Accuracy, Amount, Communication, Transparency
  - **Subtopic:** Credit/Waiver
    - **Aspect:** Accuracy, Amount, Communication, Transparency
  - **Subtopic:** Offer or Promotion
    - **Aspect:** Accuracy, Amount, Communication, Transparency
  - **Subtopic:** Refund
    - **Aspect:** Accuracy, Amount, Communication, Transparency
  - **Subtopic:** Credit or Debit Card Issues
    - **Aspect:** Accuracy, Amount, Communication, Transparency

## Food & Beverage
- **Topic:** Food & Beverage
  - **Subtopic:** Breakfast
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability
  - **Subtopic:** Brunch/Lunch
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability
  - **Subtopic:** Dinner
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability
  - **Subtopic:** In-Room Dining
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability
  - **Subtopic:** Kitchen
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability
  - **Subtopic:** Bar
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability
  - **Subtopic:** Restaurant
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability
  - **Subtopic:** Social Evening Reception
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability
  - **Subtopic:** Drink
    - **Aspect:** Quality, Options, Quantity, Comfort, Price, Hours of Operation, Sustainability

## Room
- **Topic:** Room
  - **Subtopic:** AC/Heater
    - **Aspect:** Temperature, Quality, Options, Quantity, Cleanliness, Condition, Sustainability, Safety
  - **Subtopic:** Fridge
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Sustainability, Safety
  - **Subtopic:** Hair Dryer
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Sustainability, Safety
  - **Subtopic:** Iron
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Sustainability, Safety
  - **Subtopic:** Microwave
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Sustainability, Safety
  - **Subtopic:** Television
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Sustainability, Safety
  - **Subtopic:** Phone
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Sustainability, Safety
  - **Subtopic:** Coffee
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Sustainability, Safety
  - **Subtopic:** Mini Bar
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Sustainability, Safety
  - **Subtopic:** Design
    - **Aspect:** Quality, Aesthetics, Sustainability, Safety
  - **Subtopic:** Furniture
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Aesthetics, Sustainability, Safety
  - **Subtopic:** Internet
    - **Aspect:** Speed, Ease, Sustainability, Safety
  - **Subtopic:** Kitchenette
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Aesthetics, Sustainability, Safety
  - **Subtopic:** Pets
    - **Aspect:** Availability, Disturb, Sustainability, Safety
  - **Subtopic:** Suite
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Sustainability, Safety
  - **Subtopic:** Supplies
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Sustainability, Safety
  - **Subtopic:** Temperature
    - **Aspect:** Cold, Warm, Comfort
  - **Subtopic:** Sleep
    - **Aspect:** Quality, Disturb, Sustainability, Safety, Noise
  - **Subtopic:** Housekeeping
    - **Aspect:** Quality, Availability, Aesthetics, Sustainability, Safety
  - **Subtopic:** Outlet
    - **Aspect:** Quality, Options, Quantity, Cleanliness, Condition, Aesthetics, Sustainability, Safety
  - **Subtopic:** Chair
    - **Aspect:** Comfort, Quality, Options, Quantity, Cleanliness, Condition, Aesthetics, Sustainability, Safety
  - **Subtopic:** Desk
    - **Aspect:** Quality, Options, Quantity, Cleanliness, Condition, Aesthetics, Sustainability, Safety
  - **Subtopic:** Drawer / Closet
    - **Aspect:** Quality, Options, Quantity, Cleanliness, Condition, Aesthetics, Sustainability, Safety
  - **Subtopic:** Bed or Mattress
    - **Aspect:** Comfort, Quality, Options, Quantity, Cleanliness, Condition, Aesthetics, Sustainability, Safety
  - **Subtopic:** Sofa Bed
    - **Aspect:** Comfort, Quality, Options, Quantity, Cleanliness, Condition, Aesthetics, Sustainability, Safety
  - **Subtopic:** Cribs
    - **Aspect:** Comfort, Quality, Options, Quantity, Cleanliness, Condition, Hours of Operation, Aesthetics, Sustainability

## Facilities
- **Topic:** Facilities
  - **Subtopic:** Business Center
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Aesthetics, Sustainability, Safety
  - **Subtopic:** Meeting/Conference Rooms
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Aesthetics, Sustainability, Safety
  - **Subtopic:** Fitness Center
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Aesthetics, Sustainability, Safety
  - **Subtopic:** Ice Machine
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Aesthetics, Sustainability, Safety
  - **Subtopic:** Laundry
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Aesthetics, Sustainability, Safety
  - **Subtopic:** Parking
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Aesthetics, Sustainability, Safety
  - **Subtopic:** Spa
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Aesthetics, Sustainability, Safety
  - **Subtopic:** Pool / Hot Tub
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Sustainability, Safety
  - **Subtopic:** Shuttle
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Functionality, Hours of Operation, Sustainability, Safety
  - **Subtopic:** Beach
    - **Aspect:** Quality, Availability, Quantity, Cleanliness, Condition, Aesthetics, Sustainability, Safety
  - **Subtopic:** Park
    - **Aspect:** Condition, Functionality, Aesthetics, Safety
    
## Check-In
- **Topic:** Check-In
  - **Subtopic:** Early
    - **Aspect:** Ease, Speed
  - **Subtopic:** Ease
    - **Aspect:** Ease, Speed
  - **Subtopic:** Key Card
    - **Aspect:** Ease, Speed, Functionality
  - **Subtopic:** Transaction Speed
    - **Aspect:** Ease, Speed
  - **Subtopic:** Wait
    - **Aspect:** Ease, Speed
  - **Subtopic:** Times
    - **Aspect:** Ease, Speed
  - **Subtopic:** Switch Room
    - **Aspect:** Ease, Speed
  - **Subtopic:** Digital
    - **Aspect:** Ease, Speed, Availability
  - **Subtopic:** Room Assignment
    - **Aspect:** Ease, Speed, Availability
  - **Subtopic:** Stay Preference
    - **Aspect:** Availability, Acknowledge
  - **Subtopic:** Luggage Delivery
    - **Aspect:** Accuracy, Availability
  - **Subtopic:** Communications
    - **Aspect:** Availability
  - **Subtopic:** Room Readiness
    - **Aspect:** Accuracy, Clear

## Bathroom
- **Topic:** Bathroom
  - **Subtopic:** Amenities
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Safety, Sustainability
  - **Subtopic:** Shower
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Safety, Sustainability
  - **Subtopic:** Bathtub
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Safety, Sustainability
  - **Subtopic:** Floor
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Safety, Sustainability
  - **Subtopic:** Mirror
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Safety, Sustainability
  - **Subtopic:** Sink
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Safety, Sustainability
  - **Subtopic:** Toilet
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Safety, Sustainability
  - **Subtopic:** Lightening
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Safety, Sustainability
  - **Subtopic:** Design
    - **Aspect:** Aesthetics
  - **Subtopic:** Towels
    - **Aspect:** Quality, Quantity, Cleanliness, Condition, Aesthetics, Sustainability
    
## Check-Out
- **Topic:** Check-Out
  - **Subtopic:** Ease
    - **Aspect:** Ease, Speed
  - **Subtopic:** Luggage Storage
    - **Aspect:** Ease, Speed, Availability
  - **Subtopic:** Timing
    - **Aspect:** Ease, Speed, Availability
  - **Subtopic:** Digital Check Out
    - **Aspect:** Ease, Speed, Availability

## Customer Service
- **Topic:** Customer Service
  - **Subtopic:** Concierge
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** Front Desk
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** Housekeeping
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** Maintenance
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** Event Staff
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** Manager
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** Wait Staff
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** Breakfast Staff
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** Valet Staff
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication
  - **Subtopic:** General Staff
    - **Aspect:** Attitude, Professionalism, Knowledge, Communication

## General Hotel
- **Topic:** General Hotel
  - **Subtopic:** Decor & Atmosphere
    - **Aspect:** Safety, Aesthetics, Sustainability, Condition
  - **Subtopic:** Elevator
    - **Aspect:** Safety, Condition, Cleanliness, Aesthetics, Sustainability
  - **Subtopic:** Entertainment
    - **Aspect:** Safety, Cleanliness, Aesthetics, Sustainability
  - **Subtopic:** Exterior
    - **Aspect:** Safety, Cleanliness, Aesthetics, Sustainability, Noise
  - **Subtopic:** Hallway
    - **Aspect:** Safety, Cleanliness, Aesthetics, Sustainability, Noise
  - **Subtopic:** Lobby
    - **Aspect:** Safety, Cleanliness, Aesthetics, Sustainability, Noise
  - **Subtopic:** Location
    - **Aspect:** Convenience, Safety, Cleanliness, Aesthetics, Sustainability
  - **Subtopic:** Security
    - **Aspect:** Safety, Cleanliness, Aesthetics, Sustainability

## Prices
- **Topic:** Prices
  - **Subtopic:** Food & Beverage
    - **Aspect:** Accuracy, Value
  - **Subtopic:** Internet
    - **Aspect:** Accuracy, Value
  - **Subtopic:** Parking
    - **Aspect:** Accuracy, Value
  - **Subtopic:** Room
    - **Aspect:** Accuracy, Value
  - **Subtopic:** Room Service
    - **Aspect:** Accuracy, Value
  - **Subtopic:** Fees
    - **Aspect:** Accuracy, Value

## Health & Safety
- **Topic:** Health & Safety
  - **Subtopic:** Sanitizer/Disinfectant
    - **Aspect:** Availability, Quality, Quantity, Policy, Functionality
  - **Subtopic:** Hygiene
    - **Aspect:** Overall, Quality, Adherence, Safety
  - **Subtopic:** Illness/Injury
    - **Aspect:** Food Poisoning, Fall, Allergic Reaction, Infection, Medical Expenses
  - **Subtopic:** Rodents & Pests
    - **Aspect:** Bed Bugs, Mice, Roaches, Ants, Infestation
  - **Subtopic:** First Aid
    - **Aspect:** Availability, Quality, Staff Response
  - **Subtopic:** Accessibility
    - **Aspect:** Quality, Policy, Adherence, Design, Safety 
  - **Subtopic:** Crime
    - **Aspect:** Theft, Damage, Safety
  - **Subtopic:** Negligence
    - **Aspect:** Safety, Hazards, Maintenance, Inaccuracy

## Competitors
- **Topic:** Competitors
  - **Subtopic:** Hilton
    - **Aspect:** Brand Reputation
  - **Subtopic:** Marriott
    - **Aspect:** Brand Reputation
  - **Subtopic:** Accor
    - **Aspect:** Brand Reputation
  - **Subtopic:** Choice
    - **Aspect:** Brand Reputation
  - **Subtopic:** Hyatt
    - **Aspect:** Brand Reputation
  - **Subtopic:** Wyndham
    - **Aspect:** Brand Reputation
  
## Sentiment List
* **Positive**
* **Neutral**
* **Negative**

## Emotion List
* **Belong**
* **Frustrated**
* **Discomfort**
* **Awe**
* **Curious**
* **Happy**
* **Angry**
* **Disgusted**
* **Fear**
* **Sad**
* **Surprise**
* **Anxious**
* **Embarrassed**
* **Amazed**
* **Content**
* **Gratitude**
* **Hope**
* **Nostalgia**
* **Excited**
* **Disappointed**
* **Pleased**
**Examples:**
<EXAMPLES>

**Review1:** 
4**Input:**[{"sample_id":"AAEBEQ243HRE","review_text":"Overall Score: 3 - The bed was incredibly comfortable,the bathroom was dirty there was hair in the shower. They double charged the room though they offered a refund"}]
**Output:**```json[{"sample_id":"AAEBEQ243HRE","Nested": [{"Topic":"ROOM","Subtopic":"BED OR MATTRESS","Aspect":"COMFORT","Sentiment":"POSITIVE","Emotion":"PLEASED"},{"Topic":"BATHROOM","Subtopic":"SHOWER","Aspect":"CLEANLINESS","Sentiment":"NEGATIVE","Emotion":"DISGUSTED"},{"Topic":"BILLING","Subtopic":"ACCURACY","Aspect":"OVER CHARGED","Sentiment":"NEUTRAL","Emotion":"NEUTRAL"},{"Topic":"RESOLUTION","Subtopic":"REFUND","Aspect":"ACCURACY","Sentiment":"NEUTRAL","Emotion":"RELEAVED"}]}]```

* **Review 2:** 
**Input:**[{"sample_id":"YTEB23932-1","review_text":"Overall Score: 3 - The bed was incredibly comfortable, the bathroom was dirty, there was hair in the shower."}]
**Output:**```json[{"sample_id":"YTEB23932-1","Nested":[{"Topic":"ROOM","Subtopic":"BED OR MATTRESS","Aspect":"COMFORT","Sentiment":"POSITIVE","Emotion":"PLEASED"},{"Topic":"BATHROOM","Subtopic":"SHOWER","Aspect":"CLEANLINESS","Sentiment":"NEGATIVE","Emotion":"DISGUSTED"}]}]

* **Review 3:**
**Input:** [{"sample_id":"TREB24-132","review_text":"Overall Score: 3 - They double charged the room though they offered a refund. I could not find the hairdryer in our room."}]
**Output:**```json[{"sample_id":"TREB24-132","Nested":[{"Topic":"BILLING","Subtopic":"ACCURACY","Aspect":"OVER CHARGED","Sentiment":"NEUTRAL","Emotion":"NEUTRAL"},{"Topic":"RESOLUTION","Subtopic":"REFUND","Aspect":"ACCURACY","Sentiment":"NEUTRAL","Emotion":"RELEAVED"},{"Topic":"BATHROOM","Subtopic":"HAIRDRYER","Aspect":"QUANTITY","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"}]}]```

* **Review 4:** 
**Input:** [{"sample_id":"RET2V39-G","review_text":"Overall Score: 3 - No compensation for the inconvenience, we had a problem but they only gave us some points. I cannot see my points on IHG APP nor in the website!!"}]
**Output:**```json[{"sample_id":"RET2V39-G","Nested":[{"Topic":"RESOLUTION","Subtopic":"DISCOUNT/REFUND","Aspect":"QUALITY","Sentiment":"NEGATIVE","Emotion":"DISAPPOINTED"},{"Topic":"LOYALTY","Subtopic":"POINTS","Aspect":"VALUE","Sentiment":"NEGATIVE","Emotion":"DISAPPOINTED"},{"Topic":"DIGITAL EXPERIENCE","Subtopic":"IHG APP & WEBSITE","Aspect":"AVAILABLITY","Sentiment":"NEGATIVE","Emotion":"DISAPPOINTED"}]}]```

* **Review 5:** 
**Input:** [{"sample_id":"RFEBE451-13","review_text":"Overall Score: 3 - At the price per night, not sure it was worth $450. We got a bonus for an expensive room, but it was already included in the price This hotel is full of false advertising."}]
**Output:**```json[{"sample_id":"RFEBE451-13","Nested":[{"Topic":"PRICES","Subtopic":"ROOM","Aspect":"VALUE","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"},{"Topic":"DIGITAL EXPERIENCE","Subtopic":"IHG APP & WEBSITE","Aspect":"ADVERTISING","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"}]}]```

* **Review 6** 
**Input:**  [{"sample_id":"G-UYYNNOOOORW71","review_text":"Overall Score: 2 - We were on the ground floor instead of our request for an upper floor. But they offered us a free night's stay worth of points as we have Special Club Lounge access."}]
**Output:**```json[{"sample_id":"G-UYYNNOOOORW71","Nested":[{"Topic":"CHECK-IN","Subtopic":"ROOM ASSIGNMENT","Aspect":"AVAILABILITY","Sentiment":"NEGATIVE","Emotion":"DISAPPOINTED"},{"Topic":"LOYALTY","Subtopic":"POINTS","Aspect":"QUANTITY","Sentiment":"POSITIVE","Emotion":"PLEASED"},{"Topic":"RESOLUTION","Subtopic":"OFFER","Aspect":"FREE","Sentiment":"POSITIVE","Emotion":"PLEASED"},{"Topic":"LOYALTY","Subtopic":"STAY PREFERENCES","Aspect":"AVAILABILITY","Sentiment":"POSITIVE","Emotion":"PLEASED"}]}]```

* **Review 7:**
**Input:**  [{"sample_id":"RWB2267821","review_text":"Overall Score: 4 - The cookies were warm, Room service was slow. I requested a late check out but they did not had so I had to go out quickly, I forgot my shoes under the best the receptionist said they could not find them! that was everything for me!!"}]
**Output:** ```json[{"sample_id":"RWB2267821","Nested":[{"Topic":"FOOD & BEVERAGE","Subtopic":"COMPLIMENTARY","Aspect":"QUALITY","Sentiment":"POSITIVE","Emotion":"PLEASED"},{"Topic":"FOOD & BEVERAGE","Subtopic":"ROOM SERVICE","Aspect":"SPEED","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"},{"Topic":"CHECK-OUT","Subtopic":"LATE CHECK-OUT","Aspect":"AVAIBILITY","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"},{"Topic":"RESOLUTION","Subtopic":"LOST & FOUND","Aspect":"LOST","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"}]}]```

* **Review 8:**
**Input:**[{"sample_id":"15362028421","review_text":"Overall Score: 3 - No transfer to the airport, pancake maker was broken and he website should warn that it's not suitable for people with mobility challenges.."}]
**Output:** ```json[{"sample_id":"15362028421","Nested":[{"Topic":"FACILITIES","Subtopic":"SHUTTLE","Aspect":"AVAILABILITY","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"},{"Topic":"FOOD & BEVERAGE","Subtopic":"BREAKFAST","Aspect":"MAINTENANCE","Sentiment":"NEGATIVE","Emotion":"DISSAPOINTED"},{"Topic":"DIGITAL EXPERIENCE","Subtopic":"IHG APP & WEBSITE","Aspect":"SAFETY","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"},{"Topic":"HEALTH & SAFETY","Subtopic":"ACCESSIBILITY","Aspect":"SAFETY","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"}]}]```
 
* **Review 9:**
**Input:**[{"sample_id":"T-HR327813","review_text":"Overall Score: 2 - The housekeeping staff's attitude was unacceptable. Laundry collection service was poor. There was only 1 towel, though we booked for 2."}]
**Output:** ```json[{"sample_id":"T-HR327813","Nested":[{"Topic":"CUSTOMER SERVICE","Subtopic":"HOUSEKEEPING","Aspect":"ATTITUDE","Sentiment":"NEGATIVE","Emotion":"ANGRY"},{"Topic":"FACILITIES","Subtopic":"LAUNDRY","Aspect":"QUALITY","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"},{"Topic":"BATHROOM","Subtopic":"TOWEL","Aspect":"QUANTITY","Sentiment":"NEGATIVE","Emotion":"FRUSTRATED"}]}]```

* **Review 10:**
**Input:**[{"sample_id":"647281_7483291","review_text":"Overall Score: 4 - The bed was incredibly comfortable, the bathroom was dirty, there was hair in the shower.The hotel location was great I could walk to the convenient store.The spa was amazing! The massage therapist was very skilled and the facilities were top-notch. However, the pool was a bit too cold for my liking."}]
**Output:** ```json[{"sample_id":"647281_7483291","Nested":[{"Topic":"ROOM","Subtopic":"BED OR MATTRESS","Aspect":"COMFORT","Sentiment":"POSITIVE","Emotion":"PLEASED"},{"Topic":"BATHROOM","Subtopic":"SHOWER","Aspect":"CLEANLINESS","Sentiment":"NEGATIVE","Emotion":"DISGUSTED"}{"Topic":"GENERAL HOTEL","Subtopic":"LOCATION","Aspect":"CONVENIENT","Sentiment":"POSITIVE","Emotion":"PLEASED"},{"Topic":"FACILITIES","Subtopic":"SPA","Aspect":"TREATMENT","Sentiment":"POSITIVE","Emotion":"PLEASED"},{"Topic":"FACILITIES","Subtopic":"POOL","Aspect":"TEMPERATURE","Sentiment":"NEGATIVE","Emotion":"DISAPPOINTED"}]}]```

{input}Follow JSON schema.<JSONSchema>{"description":"classifying hotel customer review focusing on hotel topics.","items":{"properties":{"Nested":{"description":"Extract classification mentioned in the customer review","items":{"additionalProperties":false,"properties":{"Aspect":{"description":"aspect related to the subtopic (e.g. quality, quantity)","type":"string"},"Emotion":{"description":"Emotion related to the subtopic","type":"string"},"Sentiment":{"description":"Sentiment related to the subtopic (e.g. POSITIVE, NEUTRAL, NEGATIVE)","type":"string"},"Subtopic":{"description":"subtopic related to the topic","type":"string"},"Topic":{"description":"topic mentioned in the review","type":"string"}},"required":["sample_id","Topic","Subtopic","Aspect","Sentiment","Emotion"],"type":"object"},"type":"array"},"sample_id":{"description":"sample_id","type":"string"}},"type":"array"},"title":"Classify topic review","type":"object"}<JSONSchema>
            ''',
            generation_config=generation_config,
            safety_settings=self._get_safety_settings()
        )

    def _get_safety_settings(self):
        """Returns safety settings for generative models to block unsafe content."""
        categories = [
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT
        ]
        return [SafetySetting(category=cat, threshold=generative_models.HarmBlockThreshold.BLOCK_NONE) for cat in categories]

    def generate_response(self, prompt: str, labels: dict):
        try:
            response = self.model.generate_content(prompt, labels=labels)
            return response.to_dict()  # Convert the response to a dictionary
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize the handler with the specific model id
handler = GeminiRequestHandler("gemini-1.5-flash-001")

@app.post("/generate")
def generate_content(request: PromptRequest):
    prompt = '[{"sample_id": "g-ChdDSUhNMG9nS0VJQ0FnTUNBcy1XOXZ3RRAB", "review_text": "overall score 5 fantastic stay at park plaza chennai omr! the location was perfect for exploring the city just short walk to major attractions and vibrant nightlife. the rooms were modern spotlessly clean and the beds were incredibly comfortable and like overall service of the staff so can one can visit this fantastic hotel."}, {"sample_id": "9eb2a4b01e10b1cfb89cd77cd4df9b1ce974cb2e", "review_text": "overall score 5 nuestra estancia en el iberostar bella vista fue simplemente maravillosa. desde el momento en que llegamos nos sentimos bienvenidos. indira la animadora hizo que cada día fuera especial con su energía contagiosa su amabilidad. volveremos pronto! hotel highlights great view"}]Follow JSON schema.<JSONSchema>"{\"description\":\"classifying hotel customer review focusing on hotel topics.\",\"items\":{\"properties\":{\"Nested\":{\"description\":\"Extract classification mentioned in the customer review\",\"items\":{\"additionalProperties\":false,\"properties\":{\"Aspect\":{\"description\":\"aspect related to the subtopic (e.g. quality, quantity)\",\"type\":\"string\"},\"Emotion\":{\"description\":\"Emotion related to the subtopic\",\"type\":\"string\"},\"Sentiment\":{\"description\":\"Sentiment related to the subtopic (e.g. POSITIVE, NEUTRAL, NEGATIVE)\",\"type\":\"string\"},\"Subtopic\":{\"description\":\"subtopic related to the topic\",\"type\":\"string\"},\"Topic\":{\"description\":\"topic mentioned in the review\",\"type\":\"string\"}},\"required\":[\"sample_id\",\"Topic\",\"Subtopic\",\"Aspect\",\"Sentiment\",\"Emotion\"],\"type\":\"object\"},\"type\":\"array\"},\"sample_id\":{\"description\":\"sample_id\",\"type\":\"string\"}},\"type\":\"array\"},\"title\":\"Classify topic review\",\"type\":\"object\"}"</JSONSchema>'
    response = handler.generate_response(prompt, labels={"model_id": request.model_params.model_id})
    
    return {"response": response}

# To run the application, use the following command:
# uvicorn your_script_name:app --reload