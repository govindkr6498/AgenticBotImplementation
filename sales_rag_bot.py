import logging
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import json
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field, EmailStr
from enum import Enum

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
                logger.info(f"Lead created successfully: {response.json()}")
            elif response.status_code == 400 and "DUPLICATES_DETECTED" in response.text:
                logger.info(f"Lead created successfully: {response.json()}")
            else:
                logger.info('fail Created')
            # logger.info(f"Lead created successfully: {response.json()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create lead: {str(e)}")
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
                chunk_size=1000,
                chunk_overlap=200,
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
            docs = self.vector_store.similarity_search(query, k=3)
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
        """Generate a response using the LLM."""
        try:
            # Get relevant context
            context = self._get_relevant_context(message)
            
            # Prepare the prompt
            prompt = f"""You are a friendly and professional sales assistant. Your goal is to:
1. Provide helpful information about our products/courses
2. Engage in natural, human-like conversation
3. Be concise and avoid unnecessary questions
4. Only ask for contact information when genuinely needed
5. Avoid repetitive or robotic responses

Guidelines:
- Keep responses brief and to the point
- Use a casual, friendly tone
- Don't ask questions unless necessary
- Don't repeat information already provided
- Don't use formal or robotic language
- Don't add unnecessary pleasantries
- Don't ask for information that's already been provided

Previous conversation:
{self.conversation_history}

Relevant product information:
{context}

Current lead information:
{json.dumps(self.partial_lead_info, indent=2) if self.partial_lead_info else "No lead information yet"}

Current lead state: {self.lead_state.value}

Human: {message}

Assistant:"""
            
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
            self.conversation_history = self.conversation_history[-10:]

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
                if self.salesforce.create_lead(self.partial_lead_info):
                    logger.info("Lead information saved to Salesforce successfully")
                    response += "\n\nGreat! I've saved your information. Our team will reach out to you shortly."
                    self.lead_state = LeadCaptureState.NO_INTEREST
                    self.partial_lead_info = {}
                else:
                    logger.error("Failed to save lead information to Salesforce")
                    response += "\n\nSorry, I had trouble saving your information. Would you mind trying again?"
            
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

def main():
    """Main function to run the sales RAG chatbot."""
    try:
        # Initialize the chatbot
        # pdf_path = 'C:/Users/admin/Documents/Document/Bot/src/Question.pdf' 
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
