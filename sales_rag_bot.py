import logging
from datetime import datetime
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import json
import pytz
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field, EmailStr
from enum import Enum
from datetime import datetime, timedelta
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LeadInfo(BaseModel):
    """Model for lead information."""
    Name: str = Field(description="Full name of the lead")
    Company: str = Field(description="Company name")
    Email: EmailStr = Field(description="Email address of the lead")
    Phone: str = Field(description="Phone number of the lead")

class LeadCaptureState(Enum):
    """Enum for tracking lead capture state."""
    NO_INTEREST = "no_interest"
    INTEREST_DETECTED = "interest_detected"
    COLLECTING_INFO = "collecting_info"
    INFO_COMPLETE = "info_complete"
    AWAITING_MEETING_CONFIRMATION = "awaiting_meeting_confirmation"
    WAITING_MEETING_SLOT_SELECTION = "waiting_meeting_slot_selection"

class SalesforceAPI:
    """Salesforce API client for lead management."""
    def __init__(self):
        """Initialize Salesforce API client."""
        self.auth_url = "https://iqb4-dev-ed.develop.my.salesforce.com/services/oauth2/token"
        self.client_id = "3MVG9pRzvMkjMb6kXIMaUGyXNzwSMewmrdMKrZmsdv8ZJ1dRg9cockiUAcWLre745UP.WoR.vWMe0Gh8Q4x35"
        self.client_secret = "67027AA5E4793A9FDCE0B13FA11E9FA2A41CA7C7270079D654B56EAC195DA91F"
        self.access_token = None
        self.instance_url = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Salesforce and get access token."""
        try:
            auth_data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }

            response = requests.post(self.auth_url, data=auth_data)
            response.raise_for_status()

            data = response.json()
            self.access_token = data.get("access_token")
            self.instance_url = data.get("instance_url")
            logger.info("Successfully authenticated with Salesforce")
            
        except Exception as e:
            logger.error(f"Salesforce authentication failed: {str(e)}")
            raise

    def create_lead(self, lead_info: Dict[str, str]) -> bool:
        """
        Create a new lead in Salesforce.
        
        Args:
            lead_info (Dict[str, str]): Lead information to create
            
        Returns:
            bool: True if lead was created successfully, False otherwise
        """
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()

            # Validate lead information
            if any(value == "N/A" for value in lead_info.values()):
                logger.error("Cannot create lead with N/A values")
                return False

            lead_url = f"{self.instance_url}/services/data/v60.0/sobjects/Lead/"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            sf_lead_payload = {
                "LastName": lead_info["Name"], 
                "Company": lead_info["Company"],
                "Email": lead_info["Email"],
                "Phone": lead_info["Phone"]
            }
            response = requests.post(
                lead_url,
                headers=headers,
                json=sf_lead_payload
            )
            # response.raise_for_status()
            if response.status_code == 201:
                lead_id = response.json().get("id")
                logger.info(f"lead_id: {lead_id}")
                logger.info(f"Lead created successfully: {response.json()}")
                return True, lead_id
                # return True, "Would you like to meet a sales advisor this week?"
            elif response.status_code == 400 and "DUPLICATES_DETECTED" in response.text:
                error_data = response.json()
                # Extract duplicate lead ID
                match_records = (
                    error_data[0]
                    .get("duplicateResult", {})
                    .get("matchResults", [])[0]
                    .get("matchRecords", [])
                )
                if match_records:
                    lead_id = match_records[0]["record"]["Id"]
                    logger.info(f"duplicate_lead_id: {lead_id}")
                    logger.info(f"Lead created successfully: {response.json()}")
                    return True, lead_id
                    # return True, "Would you like to meet a sales advisor this week?"
            else:
                logger.info('fail Created')
            # logger.info(f"Lead created successfully: {response.json()}")
            return "Would you like to meet a sales advisor this week?"
            
        except Exception as e:
            logger.error(f"Failed to create lead: {str(e)}")
            return False
    
    def create_meeting(self, lead_id: str, start_time_str: str) -> bool:
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()

            # Validate lead information
            event_url = f"{self.instance_url}/services/data/v60.0/sobjects/Event/"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            start_dt = datetime.strptime(start_time_str, "%H:%M")
            # Assuming meeting duration 30 minutes, and date as today
            ist = pytz.timezone('Asia/Kolkata')
            start_dt_local = datetime.strptime(start_time_str, "%H:%M")
            today_local = datetime.now(ist).date()
            start_local_dt = ist.localize(datetime.combine(today_local, start_dt_local.time()))
            start_utc_dt = start_local_dt.astimezone(pytz.utc) + timedelta(hours=5) + timedelta(minutes=30)
            end_utc_dt = start_utc_dt + timedelta(minutes=30)

            start_datetime_iso = start_utc_dt.isoformat()
            end_datetime_iso = end_utc_dt.isoformat()

            event_payload = {
                "Subject": "Call with Sales Advisor",
                "StartDateTime": start_datetime_iso,
                "EndDateTime": end_datetime_iso,
                "OwnerId": "0055j00000BYNIBAA5",
                "WhoId": lead_id,
                "Location": "Virtual Call",
                "Description": "Scheduled via Agentic Bot"
            }

            response = requests.post(event_url, headers=headers, json=event_payload)
            if response.status_code == 201:
                meeting_id = response.json().get("id")
                logger.info(f"Meeting created successfully: {meeting_id}")
                return True
            else:
                logger.error(f"Failed to create meeting: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Exception while creating meeting: {str(e)}")
            return False

    def show_availableMeeting(self) -> Optional[List[str]]:
        start_times = set()
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()
            event_url = f"{self.instance_url}/services/data/v60.0/query?q=SELECT+StartDateTime,+EndDateTime+FROM+Event+WHERE+StartDateTime+=+TODAY"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            response = requests.get(event_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])

                # if not records:
                #     logger.info("No meetings scheduled for today.")
                # else:
                fmt = "%H:%M"
                start_time = datetime.strptime("08:00", fmt)
                end_time = datetime.strptime("17:00", fmt)
                all_slots = set()
                current = start_time
                while current < end_time:
                    all_slots.add(current.strftime(fmt))
                    current += timedelta(minutes=30)               
                for event in records:
                    start = event.get("StartDateTime")
                    if start:
                        try:
                            dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f%z")
                            time_only = dt.strftime("%H:%M")
                            start_times.add(time_only)
                        except Exception as parse_err:
                            logger.warning(f"Failed to parse StartDateTime: {start} ({parse_err})")
                available_slots = sorted(all_slots - start_times)
                logger.info(f"Schedule  meeting slots: {start_times}")
                logger.info(f"Available meeting slots: {available_slots}")
                return available_slots
            else:
                logger.error(f"Failed to showing meeting Time: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Exception while showing meeting: {str(e)}")
            return False


class SalesRAGBot:
    def __init__(self, pdf_path: str, model_name: str = "gpt-3.5-turbo-0125"):
        """Initialize the Sales RAG Bot."""
        self.pdf_path = pdf_path
        self.model_name = model_name
        self._setup_environment()
        self._initialize_components()
        self.lead_state = LeadCaptureState.NO_INTEREST
        self.partial_lead_info = {}
        self.conversation_history = []
        self.salesforce = SalesforceAPI()
        self.awaiting_meeting_confirmation = False
        self.awaiting_meeting_slot_selection = False
        self.awaiting_meeting_response = False
        self.current_lead_id = None
        self.available_slots = []
        logger.info("SalesRAGBot initialized")

    def _setup_environment(self) -> None:
        """Set up environment variables and API keys."""
        load_dotenv()
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        logger.info("Environment setup completed")

    def _initialize_components(self) -> None:
        """Initialize all necessary components for the chatbot."""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(model=self.model_name)
            
            # Load and process PDF
            self._load_pdf()
            
            # Initialize vector store
            self._setup_vector_store()
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise

    def _load_pdf(self) -> None:
        """Load and split the PDF document."""
        try:
            loader = PyPDFLoader(self.pdf_path)
            raw_docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                length_function=len,
            )
            
            self.docs = text_splitter.split_documents(raw_docs)
            logger.info(f"PDF loaded and split into {len(self.docs)} chunks")
                
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise

    def _setup_vector_store(self) -> None:
        """Set up the FAISS vector store."""
        try:
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(
                documents=self.docs,
                embedding=embeddings
            )
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    def _get_relevant_context(self, query: str) -> str:
        """Get relevant context from the vector store."""
        try:
            docs = self.vector_store.similarity_search(query, k=5)
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return ""

    def _extract_lead_info(self, message: str) -> Optional[Dict[str, str]]:
        """Extract lead information from the message."""
        try:
            # Use LLM to extract structured information
            prompt = f"""Extract contact information from the following message. Return ONLY a JSON object with these exact fields if found:
            - Name (required; store full name here)
            - Company (required)
            - Email (required)
            - Phone (required)

            Rules:
            1. Extract information even if it's provided in different formats or variations
            2. For Name:
            - Look for patterns like "name is X", "I am X", "my name is X", "this is X", "I'm X"
            - If full name is provided, keep the full name in the Name field
            - Do not shorten or split the name
            - Also look for "I'm X" or just "X" if it appears to be a name
            3. For company:
            - Look for patterns like "I work at X", "my company is X", "I'm from X", "at X", "with X"
            - Also look for "company name is X", "organization is X"
            - Don't extract company names from general product mentions
            4. For email:
            - Look for patterns like "email is X", "my email is X", "contact me at X"
            - Also look for "reach me at X", "send to X"
            - Extract any valid email address format
            5. For phone:
            - Look for patterns like "number is X", "phone is X", "call me at X"
            - Also look for "reach me at X", "contact me at X"
            - Accept various phone formats (with/without country code, spaces, dashes)
            6. If multiple values are found for a field, use the most recent or most specific one
            7. If a field is not found, do not include it in the JSON
            8. Return null if no contact information is found
            9. Handle both single-line and multi-line information
            10. Look for information even if it's just the value without context (e.g., just an email address)          
            Message: {message}

            Return ONLY the JSON object or null, nothing else.
            """
                        
            response = self.llm.invoke(prompt)
            try:
                lead_data = json.loads(response.content)
                if lead_data:
                    # Normalize and validate each field individually
                    normalized_data = {}
                    for field in ['Name', 'Email', 'Phone']:
                        if field in lead_data and lead_data[field] and lead_data[field] != "N/A":
                            normalized_data[field] = lead_data[field].strip()
                    
                    if normalized_data:
                        normalized_data['Company'] = 'Iquestbee Technology'
                        return normalized_data
            except json.JSONDecodeError:
                pass
            return None

        except Exception as e:
            logger.error(f"Error extracting lead info: {str(e)}")
            return None

    def _update_lead_state(self, message: str) -> None:
        """Update the lead capture state based on the message."""
        interest_indicators = [
            "schedule", "meeting", "interested", "pricing", "cost","interest",
            "sign up", "enroll", "register", "buy", "purchase","want","Desire"
        ]
        
        if self.lead_state == LeadCaptureState.NO_INTEREST:
            if any(indicator in message.lower() for indicator in interest_indicators):
                self.lead_state = LeadCaptureState.INTEREST_DETECTED
                logger.info("Interest detected in conversation")
        
        if self.lead_state in [LeadCaptureState.INTEREST_DETECTED, LeadCaptureState.COLLECTING_INFO]:
            lead_info = self._extract_lead_info(message)
            if lead_info:
                # Update partial lead info with new information
                self.partial_lead_info.update(lead_info)
                self.lead_state = LeadCaptureState.COLLECTING_INFO
                
                # Check if we have all required information
                if all(key in self.partial_lead_info and self.partial_lead_info[key] not in [None, "N/A"]
                    for key in ['Name', 'Email', 'Phone']):
                    self.lead_state = LeadCaptureState.INFO_COMPLETE
                    logger.info("All lead information collected")

    def _get_missing_fields(self) -> List[str]:
        """Get list of missing required fields in lead information."""
        required_fields = ['Name', 'Email', 'Phone']
        return [field for field in required_fields 
                if field not in self.partial_lead_info or self.partial_lead_info[field] == "N/A"]

    def _generate_response(self, message: str) -> str:
        """Generate a response using the LLM with enhanced conversation history handling and topic tracking."""
        if self.awaiting_meeting_response:
            self.awaiting_meeting_response = False
            if message.strip().lower() in ["yes", "yeah", "yup", "sure", "please"]:
                available_slots = self.show_availableMeeting()
                if available_slots:
                    slots_text = ", ".join(available_slots)
                    return f"Here are the available meeting slots: {slots_text}. Please choose one."
                else:
                    return "Sorry, no available meeting slots were found at the moment."
            else:
                return "No problem. Let me know if you need anything else."
            
        try:
            # Get relevant context
            context = self._get_relevant_context(message)
            
            # Extract the last few messages for immediate context
            recent_messages = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
            
            # Extract topics from recent conversation
            topics_prompt = """Given these conversation messages, identify the main topic being discussed:
            {}
            Return ONLY the topic being discussed, nothing else.""".format("\n".join(recent_messages))
            
            topic_response = self.llm.invoke(topics_prompt)
            current_topic = topic_response.content
            
            # Prepare system context including current topic
            system_context = f"""Current topic of discussion: {current_topic}
            Previous conversation context: {recent_messages}
            Product information: {context}
            Lead information: {json.dumps(self.partial_lead_info, indent=2) if self.partial_lead_info else "No lead information yet"}
            Lead state: {self.lead_state.value}"""
            
            # Enhanced response generation prompt
            prompt = f"""You are a friendly and professional sales assistant. Your goal is to:
            1. Provide helpful information about our products/courses
            2. Engage in natural, human-like conversation
            3. Maintain continuity with the current topic: {current_topic}
            4. Understand and reference previous context when relevant
            5. If the user asks for more information about something you mentioned, expand on that specific topic
            6. If the query seems to reference previous information but is unclear (like 'Tell me more' or 'What about the price?'),
               use the current topic to understand what they're asking about
            7. Be concise but complete in your responses
            8. Use natural conversational language
            9. Only ask for contact information when there's genuine interest

            System Context:
            {system_context}

            Human: {message}

            Assistant: Be direct and natural in your response, maintaining the conversation flow about {current_topic} if relevant."""
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error. Let's try again."

    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message and handle lead capture if needed."""
        try:
            # Update lead capture state
            self._update_lead_state(message)
            
            # Add user message to conversation history
            self.conversation_history.append(f"Human: {message}")
            
            # Generate response
            response = self._generate_response(message)
            
            # Add assistant response to conversation history
            self.conversation_history.append(f"Assistant: {response}")
            self.conversation_history = self.conversation_history[-30:]
            logger.info(f"Conversation history: {self.conversation_history}")
            # Handle lead capture state
            if self.lead_state == LeadCaptureState.INTEREST_DETECTED:
                missing_fields = self._get_missing_fields()
                if missing_fields:
                    # Make the request for information more natural
                    if len(missing_fields) == 1:
                        response += f"\n\nCould you share your {missing_fields[0]}?"
                    else:
                        response += f"\n\nCould you share your {', '.join(missing_fields[:-1])} and {missing_fields[-1]}?"
            
            elif self.lead_state == LeadCaptureState.COLLECTING_INFO:
                missing_fields = self._get_missing_fields()
                if missing_fields:
                    # Only ask for missing fields
                    if len(missing_fields) == 1:
                        response += f"\n\nJust need your {missing_fields[0]} to get started."
                    else:
                        response += f"\n\nJust need your {', '.join(missing_fields[:-1])} and {missing_fields[-1]} to get started."
                else:
                    # If no missing fields but still in COLLECTING_INFO state, move to INFO_COMPLETE
                    self.lead_state = LeadCaptureState.INFO_COMPLETE
            
            elif self.lead_state == LeadCaptureState.INFO_COMPLETE:
                lead_created, lead_id = self.salesforce.create_lead(self.partial_lead_info)
                if lead_created:
                    self.current_lead_id = lead_id
                    logger.info("Lead information saved to Salesforce successfully")
                    self.lead_state = LeadCaptureState.AWAITING_MEETING_CONFIRMATION
                    response += "\n\nGreat! I've saved your information.\n\nDo you want to schedule meeting with FSTC Team Member? (Yes/No)"
                else:
                    logger.error("Failed to save lead information to Salesforce")
                    response += "\n\nSorry, I had trouble saving your information. Would you mind trying again?"

            elif self.lead_state == LeadCaptureState.AWAITING_MEETING_CONFIRMATION:
                if message.strip().lower() in ["yes", "yeah", "y", "sure", "please"]:
                    self.available_slots = self.salesforce.show_availableMeeting() or []
                    # self.lead_state = LeadCaptureState.NO_INTEREST  # Reset state
                    if self.available_slots:
                        self.lead_state = LeadCaptureState.WAITING_MEETING_SLOT_SELECTION
                        # slots_text = ", ".join(self.available_slots)
                        slots_text = self.format_slots_nicely(self.available_slots)
                        response += f"\n\nHere are the available meeting slots for today: {slots_text}"
                    else:
                        self.lead_state = LeadCaptureState.NO_INTEREST
                        response += "\n\nSorry, I couldn’t fetch available meeting slots right now."
                else:
                    self.lead_state = LeadCaptureState.NO_INTEREST
                    response += "\n\nNo problem! Let me know if you have any other questions."

            elif self.lead_state == LeadCaptureState.WAITING_MEETING_SLOT_SELECTION:
                # Clean and normalize user input
                parsed_time = message.strip().lower()
                parsed_time = parsed_time.replace("\"", "").replace("'", "").replace(" ", "").replace(".", "")

                # Normalize formats like "9", "930", "09", "0930"
                if parsed_time.isdigit():
                    if len(parsed_time) <= 2:
                        parsed_time = parsed_time.zfill(2) + ":00"        # "9" -> "09:00"
                    elif len(parsed_time) == 3:
                        parsed_time = "0" + parsed_time[0] + ":" + parsed_time[1:]  # "930" -> "09:30"
                    elif len(parsed_time) == 4:
                        parsed_time = parsed_time[:2] + ":" + parsed_time[2:]       # "0930" -> "09:30"
                elif ":" in parsed_time:
                    parts = parsed_time.split(":")
                    if len(parts) == 2 and all(p.isdigit() for p in parts):
                        parsed_time = parts[0].zfill(2) + ":" + parts[1].zfill(2)

                logger.info(f"✅ Final normalized time: {parsed_time}")
                logger.info(f"🔎 Comparing against available slots: {self.available_slots}")

                # Match the time against available slots exactly
                if parsed_time in self.available_slots and self.current_lead_id:
                    success = self.salesforce.create_meeting(self.current_lead_id, parsed_time)
                    self.lead_state = LeadCaptureState.NO_INTEREST
                    self.available_slots = []
                    self.current_lead_id = None
                    if success:
                        response = f"✅ Your meeting has been scheduled at {parsed_time}. Our team will contact you soon!"
                    else:
                        response = f"❌ Something went wrong while scheduling your meeting at {parsed_time}. Please try again."
                else:
                    response = f"⚠️ \"{message}\" is not a valid time. Please choose from: {', '.join(self.available_slots)}"


            return {
                "response": response,
                "lead_info": self.partial_lead_info if self.partial_lead_info else None,
                "lead_state": self.lead_state.value
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "Sorry, I encountered an error. Let's try again.",
                "lead_info": None,
                "lead_state": self.lead_state.value
            }


    def format_slots_nicely(self, slots: List[str], columns: int = 3) -> str:
        """Formats meeting slots in a professional table layout"""
        if not slots:
            return "No available time slots at the moment."
        
        # Calculate column width based on longest time slot
        max_length = max(len(slot) for slot in slots) 
        column_width = max_length + 5  # extra spacing between columns
        # Format slots into aligned columns
        rows = []
        for i in range(0, len(slots), columns):
            row = []
            for j in range(columns):
                idx = i + j
                if idx < len(slots):
                    row.append(f"{slots[idx]:>{column_width}}")
            rows.append("".join(row)) 
        
        return (
             "Available meeting times:\n\n" +
            "\n".join(rows) +  # only 1 newline between rows
            "\n\nPlease pick one."
        )


def main():
    """Main function to run the sales RAG chatbot."""
    try:
        # Initialize the chatbot
        # pdf_path = 'C:/Users/admin/Documents/Document/Bot/src/FSTC_Contact.pdf' 
        pdf_path = '/home/ubuntu/AgenticBotImplementation/FSTC_Contact.pdf'
        chatbot = SalesRAGBot(pdf_path)
        
        print("Welcome to the Sales Assistant!")
        print("I can help you learn more about our products and services.")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! Thank you for your interest.")
                break
                
            if not user_input:
                print("Please enter a message.")
                continue
                
            response = chatbot.process_message(user_input)
            print("\nBot:", response['response'])
            
            # Log if lead information was captured
            if response['lead_info']:
                print("\n[Lead information captured:", response['lead_info'], "]")
                print("[Current state:", response['lead_state'], "]")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
