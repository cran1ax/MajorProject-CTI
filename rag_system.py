# ==================== RAG SYSTEM FOR CYBERSECURITY THREAT INTELLIGENCE ====================
"""
Retrieval-Augmented Generation (RAG) System for Cybersecurity
This module implements a RAG pipeline that:
1. Creates and manages a knowledge base of cybersecurity documents
2. Generates embeddings using sentence-transformers
3. Performs semantic search to retrieve relevant context
4. Augments LLM responses with retrieved knowledge
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers")

# For vector similarity
from sklearn.metrics.pairwise import cosine_similarity

# For text processing
import re
from collections import defaultdict


# ==================== CONFIGURATION ====================
@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model: str = "all-MiniLM-L6-v2"  # Lightweight, fast model
    chunk_size: int = 500  # Characters per chunk
    chunk_overlap: int = 50  # Overlap between chunks
    top_k: int = 5  # Number of documents to retrieve
    similarity_threshold: float = 0.3  # Minimum similarity score
    knowledge_base_path: str = "knowledge_base"
    embeddings_cache_path: str = "embeddings_cache.pkl"


# ==================== CYBERSECURITY KNOWLEDGE BASE ====================
CYBERSECURITY_KNOWLEDGE_BASE = [
    {
        "id": "phishing_001",
        "category": "Phishing",
        "title": "Phishing Attack Overview",
        "content": """Phishing is a type of social engineering attack often used to steal user data, including login credentials and credit card numbers. It occurs when an attacker, masquerading as a trusted entity, dupes a victim into opening an email, instant message, or text message. The recipient is then tricked into clicking a malicious link, which can lead to the installation of malware, the freezing of the system as part of a ransomware attack or the revealing of sensitive information.
        
Common phishing indicators include:
- Suspicious sender email addresses that mimic legitimate companies
- Urgent language demanding immediate action
- Generic greetings like "Dear Customer" instead of your name
- Mismatched or suspicious URLs (hover to check before clicking)
- Poor grammar and spelling mistakes
- Requests for sensitive information via email
- Unexpected attachments

Defense mechanisms:
- Email filtering and spam detection
- Multi-factor authentication (MFA)
- Security awareness training for employees
- Domain-based Message Authentication (DMARC)
- Regular phishing simulation exercises"""
    },
    {
        "id": "ransomware_001",
        "category": "Ransomware",
        "title": "Ransomware Attack Prevention and Response",
        "content": """Ransomware is malicious software that encrypts files on a device, rendering any files and systems that rely on them unusable. Malicious actors then demand ransom in exchange for decryption. Ransomware attacks can cause costly disruptions to operations and loss of critical information.

Types of Ransomware:
- Crypto ransomware: Encrypts valuable files
- Locker ransomware: Locks users out of their devices
- Double extortion: Encrypts and threatens to leak data
- RaaS (Ransomware-as-a-Service): Criminal business model

Prevention strategies:
1. Regular backups following 3-2-1 rule (3 copies, 2 media types, 1 offsite)
2. Keep systems and software updated with security patches
3. Implement network segmentation
4. Use endpoint detection and response (EDR) solutions
5. Restrict administrative privileges
6. Disable Remote Desktop Protocol (RDP) if not needed
7. Email filtering to block malicious attachments

Incident Response:
- Isolate affected systems immediately
- Do NOT pay the ransom (no guarantee of decryption)
- Report to law enforcement (FBI IC3, CISA)
- Restore from clean backups
- Conduct forensic analysis to understand attack vector"""
    },
    {
        "id": "ddos_001",
        "category": "DDoS",
        "title": "Distributed Denial of Service (DDoS) Attacks",
        "content": """A DDoS attack aims to overwhelm a website, server, or network with more traffic than it can handle, causing it to become slow or completely unavailable to legitimate users.

Types of DDoS attacks:
1. Volumetric attacks: Flood bandwidth (UDP floods, ICMP floods)
2. Protocol attacks: Exploit network protocol weaknesses (SYN floods, Ping of Death)
3. Application layer attacks: Target web applications (HTTP floods, Slowloris)

Attack metrics:
- Volume measured in Gbps (Gigabits per second)
- Packet rate measured in PPS (Packets per second)
- Request rate measured in RPS (Requests per second)

Mitigation strategies:
- Use Content Delivery Networks (CDNs) like Cloudflare, Akamai
- Implement rate limiting and traffic shaping
- Deploy Web Application Firewalls (WAF)
- Use anycast network distribution
- Enable DDoS protection services from cloud providers
- Have an incident response plan ready
- Monitor traffic patterns for anomalies

Early warning signs:
- Unusual traffic spikes from single IP or region
- Slow network performance
- Unavailability of specific services
- High number of requests to single endpoint"""
    },
    {
        "id": "malware_001",
        "category": "Malware",
        "title": "Malware Types and Protection",
        "content": """Malware (malicious software) is any program or code designed to harm computer systems. Understanding different types helps in implementing appropriate defenses.

Common malware types:
1. Viruses: Self-replicating code that attaches to clean files
2. Worms: Self-replicating malware that spreads across networks
3. Trojans: Disguised as legitimate software
4. Spyware: Secretly monitors user activity
5. Adware: Displays unwanted advertisements
6. Rootkits: Provides privileged access while hiding presence
7. Keyloggers: Records keystrokes to steal credentials
8. Fileless malware: Operates in memory, leaves no files

Infection vectors:
- Malicious email attachments
- Drive-by downloads from compromised websites
- Infected USB drives
- Software vulnerabilities
- Social engineering

Protection measures:
- Install reputable antivirus/anti-malware software
- Keep operating systems and applications updated
- Use application whitelisting
- Implement least privilege access
- Regular security scans
- Network traffic monitoring
- Sandbox suspicious files before execution
- User awareness training"""
    },
    {
        "id": "insider_threat_001",
        "category": "Insider Threat",
        "title": "Insider Threat Detection and Prevention",
        "content": """Insider threats come from people within the organization—employees, former employees, contractors, or business partners—who have inside information about security practices, data, and computer systems.

Types of insider threats:
1. Malicious insiders: Intentionally steal data or sabotage systems
2. Negligent insiders: Accidentally cause breaches through carelessness
3. Compromised insiders: Accounts taken over by external attackers

Warning indicators:
- Unusual access patterns or working hours
- Accessing data unrelated to job function
- Large data downloads or transfers
- Dissatisfaction or grievances with organization
- Financial difficulties
- Unexplained wealth
- Resignation followed by unusual activity

Prevention and detection:
- Implement User and Entity Behavior Analytics (UEBA)
- Enforce principle of least privilege
- Regular access reviews and audits
- Data Loss Prevention (DLP) solutions
- Monitor privileged user activities
- Comprehensive offboarding procedures
- Background checks for sensitive positions
- Create culture of security awareness
- Anonymous reporting channels"""
    },
    {
        "id": "sql_injection_001",
        "category": "SQL Injection",
        "title": "SQL Injection Attacks and Prevention",
        "content": """SQL injection is a code injection technique that exploits security vulnerabilities in an application's database layer. Attackers can insert malicious SQL statements into entry fields to manipulate databases.

Types of SQL injection:
1. In-band SQLi: Error-based and UNION-based attacks
2. Blind SQLi: Boolean-based and time-based attacks
3. Out-of-band SQLi: Uses different channels for attack and results

Attack examples:
- ' OR '1'='1' -- (bypasses authentication)
- '; DROP TABLE users; -- (destructive commands)
- UNION SELECT username, password FROM users -- (data extraction)

Prevention measures:
1. Parameterized queries (prepared statements)
2. Stored procedures with proper input validation
3. Input validation and sanitization
4. Escape special characters
5. Use ORM frameworks that handle SQL safely
6. Implement least privilege for database accounts
7. Regular security testing (SAST/DAST)
8. Web Application Firewalls (WAF)
9. Keep database software updated

Code example (Python):
# Vulnerable:
query = f"SELECT * FROM users WHERE id = {user_input}"
# Safe (parameterized):
cursor.execute("SELECT * FROM users WHERE id = ?", (user_input,))"""
    },
    {
        "id": "zero_day_001",
        "category": "Zero-Day",
        "title": "Zero-Day Vulnerabilities",
        "content": """A zero-day vulnerability is a software security flaw unknown to the vendor and for which no patch exists. Zero-day attacks exploit these vulnerabilities before developers have a chance to create a fix.

Characteristics:
- Unknown to software vendor
- No patch available
- High value in criminal and nation-state circles
- Often discovered through fuzzing or reverse engineering

Recent notable zero-days:
- Log4Shell (CVE-2021-44228)
- Spring4Shell (CVE-2022-22965)
- Microsoft Exchange ProxyLogon
- Chrome V8 engine vulnerabilities

Defense strategies:
1. Defense in depth approach
2. Network segmentation
3. Regular vulnerability scanning
4. Threat intelligence feeds
5. Endpoint Detection and Response (EDR)
6. Application sandboxing
7. Virtual patching through WAF
8. Behavior-based detection systems
9. Regular security assessments
10. Incident response planning"""
    },
    {
        "id": "incident_response_001",
        "category": "Incident Response",
        "title": "Cybersecurity Incident Response Framework",
        "content": """Incident response is a structured approach to handling security breaches. The NIST framework provides a widely-adopted methodology.

NIST Incident Response Phases:
1. Preparation
   - Develop incident response plan
   - Form incident response team
   - Conduct training and exercises
   - Prepare tools and resources
   
2. Detection and Analysis
   - Monitor security events
   - Analyze alerts and indicators
   - Determine incident scope
   - Document findings
   
3. Containment, Eradication, and Recovery
   - Short-term containment (isolate affected systems)
   - Long-term containment (apply patches)
   - Eradication (remove malware, close vulnerabilities)
   - Recovery (restore systems, verify functionality)
   
4. Post-Incident Activity
   - Lessons learned meeting
   - Update documentation
   - Improve security controls
   - Report to stakeholders

Key metrics:
- Mean Time to Detect (MTTD)
- Mean Time to Respond (MTTR)
- Mean Time to Contain (MTTC)
- Mean Time to Recover"""
    },
    {
        "id": "threat_intelligence_001",
        "category": "Threat Intelligence",
        "title": "Cyber Threat Intelligence (CTI)",
        "content": """Cyber Threat Intelligence is evidence-based knowledge about existing or emerging threats that helps organizations make informed decisions about their security posture.

Types of threat intelligence:
1. Strategic: High-level trends for executives
2. Tactical: TTPs (Tactics, Techniques, Procedures) for security teams
3. Operational: Specific attacks and campaigns
4. Technical: IOCs (Indicators of Compromise)

Sources of threat intelligence:
- Open Source Intelligence (OSINT)
- Commercial threat feeds
- Information Sharing and Analysis Centers (ISACs)
- Government agencies (CISA, FBI)
- Dark web monitoring
- Internal telemetry

MITRE ATT&CK Framework:
A globally-accessible knowledge base of adversary tactics and techniques based on real-world observations. Categories include:
- Initial Access
- Execution
- Persistence
- Privilege Escalation
- Defense Evasion
- Credential Access
- Discovery
- Lateral Movement
- Collection
- Exfiltration
- Impact

Threat Intelligence Platforms (TIP):
- MISP (Open source)
- ThreatConnect
- Recorded Future
- Anomali"""
    },
    {
        "id": "network_security_001",
        "category": "Network Security",
        "title": "Network Security Best Practices",
        "content": """Network security involves policies, practices, and tools designed to protect network infrastructure from unauthorized access, misuse, and attacks.

Defense layers:
1. Perimeter security
   - Firewalls (stateful, next-gen)
   - Intrusion Detection/Prevention Systems (IDS/IPS)
   - DMZ architecture
   
2. Network segmentation
   - VLANs
   - Micro-segmentation
   - Zero Trust Architecture
   
3. Access control
   - 802.1X authentication
   - Network Access Control (NAC)
   - VPN for remote access
   
4. Monitoring
   - SIEM (Security Information and Event Management)
   - Network traffic analysis
   - Flow data collection (NetFlow, sFlow)

Zero Trust principles:
- Never trust, always verify
- Assume breach
- Verify explicitly
- Use least privilege access
- Micro-segmentation

Common protocols to secure:
- DNS (use DNSSEC)
- HTTPS (enforce TLS 1.3)
- SSH (disable password auth, use keys)
- SMTP (implement SPF, DKIM, DMARC)"""
    },
    {
        "id": "compliance_001",
        "category": "Compliance",
        "title": "Cybersecurity Compliance Frameworks",
        "content": """Organizations must comply with various cybersecurity regulations and frameworks depending on their industry and geography.

Major frameworks:
1. NIST Cybersecurity Framework
   - Identify, Protect, Detect, Respond, Recover
   - Widely adopted across industries
   
2. ISO 27001
   - International standard for ISMS
   - Certification available
   
3. SOC 2
   - Trust Service Criteria
   - Security, Availability, Processing Integrity, Confidentiality, Privacy
   
4. PCI DSS
   - Required for payment card handling
   - 12 requirements covering network security, access control, etc.

Industry-specific regulations:
- HIPAA (Healthcare - USA)
- GDPR (Data Protection - EU)
- CCPA (Privacy - California)
- GLBA (Financial - USA)
- FERPA (Education - USA)

Compliance benefits:
- Reduced legal and financial risk
- Improved security posture
- Customer trust and confidence
- Competitive advantage
- Insurance premium reductions"""
    },
    {
        "id": "cloud_security_001",
        "category": "Cloud Security",
        "title": "Cloud Security Best Practices",
        "content": """Cloud security encompasses policies, controls, and technologies to protect cloud computing environments from threats.

Shared responsibility model:
- Cloud provider: Security OF the cloud (infrastructure)
- Customer: Security IN the cloud (data, access, applications)

Key security areas:
1. Identity and Access Management (IAM)
   - Implement least privilege
   - Use MFA for all accounts
   - Regular access reviews
   
2. Data protection
   - Encryption at rest and in transit
   - Key management (HSM, KMS)
   - Data classification
   
3. Network security
   - Virtual Private Clouds (VPC)
   - Security groups and NACLs
   - Private endpoints
   
4. Logging and monitoring
   - Enable CloudTrail/Cloud Audit Logs
   - Configure alerts for anomalies
   - Centralize log management

Cloud security tools:
- AWS: GuardDuty, Security Hub, WAF
- Azure: Defender for Cloud, Sentinel
- GCP: Security Command Center

Common misconfigurations:
- Publicly accessible storage buckets
- Overly permissive IAM policies
- Unencrypted databases
- Missing logging"""
    }
]


# ==================== TEXT PROCESSING ====================
class TextProcessor:
    """Handles text chunking and preprocessing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\'\"()/@]', '', text)
        return text.strip()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        text = TextProcessor.clean_text(text)
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '! ', '? ', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep != -1 and last_sep > chunk_size // 2:
                        end = start + last_sep + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks


# ==================== VECTOR STORE ====================
class VectorStore:
    """Simple vector store for document embeddings"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.model = None
        
        if EMBEDDINGS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            self.model = SentenceTransformer(self.config.embedding_model)
            print(f"✅ Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            self.model = None
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector store"""
        if self.model is None:
            raise ValueError("Embedding model not loaded")
        
        # Generate embeddings
        new_embeddings = self.model.encode(documents, show_progress_bar=True)
        
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        print(f"✅ Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents"""
        if self.model is None:
            raise ValueError("Embedding model not loaded")
        
        if not self.embeddings:
            return []
        
        top_k = top_k or self.config.top_k
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.config.similarity_threshold:
                results.append((
                    self.documents[idx],
                    float(similarities[idx]),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, path: str = None):
        """Save vector store to disk"""
        path = path or self.config.embeddings_cache_path
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Saved vector store to {path}")
    
    def load(self, path: str = None) -> bool:
        """Load vector store from disk"""
        path = path or self.config.embeddings_cache_path
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            print(f"✅ Loaded vector store from {path}")
            return True
        return False


# ==================== RAG ENGINE ====================
class RAGEngine:
    """Main RAG engine for retrieval-augmented generation"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.vector_store = VectorStore(self.config)
        self.text_processor = TextProcessor()
        self.is_initialized = False
    
    def initialize_knowledge_base(self, documents: List[Dict] = None, force_rebuild: bool = False):
        """Initialize the knowledge base with documents"""
        
        # Try to load from cache first
        if not force_rebuild and self.vector_store.load():
            self.is_initialized = True
            return
        
        # Use default knowledge base if none provided
        documents = documents or CYBERSECURITY_KNOWLEDGE_BASE
        
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            # Chunk the content
            chunks = self.text_processor.chunk_text(
                doc['content'],
                self.config.chunk_size,
                self.config.chunk_overlap
            )
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'doc_id': doc.get('id', ''),
                    'category': doc.get('category', ''),
                    'title': doc.get('title', ''),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
        
        # Add to vector store
        self.vector_store.add_documents(all_chunks, all_metadata)
        
        # Save to cache
        self.vector_store.save()
        self.is_initialized = True
        
        print(f"✅ Knowledge base initialized with {len(all_chunks)} chunks from {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if not self.is_initialized:
            raise ValueError("Knowledge base not initialized. Call initialize_knowledge_base() first.")
        
        results = self.vector_store.search(query, top_k)
        
        retrieved_docs = []
        for content, score, metadata in results:
            retrieved_docs.append({
                'content': content,
                'score': score,
                'category': metadata.get('category', ''),
                'title': metadata.get('title', ''),
                'doc_id': metadata.get('doc_id', '')
            })
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context string"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Source {i}: {doc['title']} (Category: {doc['category']}, Relevance: {doc['score']:.2f})]\n"
                f"{doc['content']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def augment_prompt(self, user_query: str, top_k: int = None) -> Tuple[str, List[Dict]]:
        """Augment user query with retrieved context"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(user_query, top_k)
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Create augmented prompt
        if context:
            augmented_prompt = f"""You are a cybersecurity expert assistant. Use the following retrieved knowledge to answer the user's question. If the retrieved information is relevant, incorporate it into your response. Always provide accurate, actionable advice.

RETRIEVED KNOWLEDGE:
{context}

USER QUESTION: {user_query}

Please provide a comprehensive answer based on the retrieved knowledge and your expertise. If the retrieved information doesn't fully address the question, supplement with your general knowledge while clearly indicating what comes from the retrieved sources vs. general knowledge."""
        else:
            augmented_prompt = f"""You are a cybersecurity expert assistant. Answer the following question based on your expertise.

USER QUESTION: {user_query}

Please provide a comprehensive and accurate answer."""
        
        return augmented_prompt, retrieved_docs
    
    def add_custom_documents(self, documents: List[Dict]):
        """Add custom documents to the knowledge base"""
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            chunks = self.text_processor.chunk_text(
                doc.get('content', ''),
                self.config.chunk_size,
                self.config.chunk_overlap
            )
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'doc_id': doc.get('id', hashlib.md5(chunk.encode()).hexdigest()[:8]),
                    'category': doc.get('category', 'Custom'),
                    'title': doc.get('title', 'Custom Document'),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
        
        self.vector_store.add_documents(all_chunks, all_metadata)
        self.vector_store.save()
        
        print(f"✅ Added {len(documents)} custom documents")
    
    def add_from_csv(self, csv_path: str, content_columns: List[str], category_column: str = None):
        """Add documents from a CSV file"""
        df = pd.read_csv(csv_path)
        documents = []
        
        for idx, row in df.iterrows():
            content_parts = [str(row[col]) for col in content_columns if col in df.columns]
            content = " | ".join(content_parts)
            
            doc = {
                'id': f"csv_{idx}",
                'content': content,
                'title': f"Record {idx}",
                'category': str(row[category_column]) if category_column and category_column in df.columns else 'CSV Data'
            }
            documents.append(doc)
        
        self.add_custom_documents(documents)
        print(f"✅ Added {len(documents)} documents from {csv_path}")


# ==================== RAG CHATBOT ====================
class RAGChatbot:
    """Chatbot with RAG capabilities"""
    
    def __init__(self, rag_engine: RAGEngine, api_key: str = None):
        self.rag_engine = rag_engine
        self.api_key = api_key
        self.conversation_history = []
    
    def get_response(self, user_message: str, use_api: bool = False) -> Dict:
        """Get response with RAG augmentation"""
        # Augment the prompt with retrieved context
        augmented_prompt, retrieved_docs = self.rag_engine.augment_prompt(user_message)
        
        response_data = {
            'user_message': user_message,
            'retrieved_docs': retrieved_docs,
            'num_sources': len(retrieved_docs),
            'augmented_prompt': augmented_prompt
        }
        
        if use_api and self.api_key:
            # Call external API (Gemini, OpenAI, etc.)
            response_data['response'] = self._call_api(augmented_prompt)
        else:
            # Generate a local response based on retrieved docs
            response_data['response'] = self._generate_local_response(user_message, retrieved_docs)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response_data['response']
        })
        
        return response_data
    
    def _generate_local_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate a response based on retrieved documents (no API)"""
        if not retrieved_docs:
            return "I couldn't find specific information about your query in my knowledge base. Please try rephrasing your question or ask about specific cybersecurity topics like phishing, ransomware, DDoS attacks, malware, or incident response."
        
        # Build response from retrieved docs
        response_parts = [
            f"Based on my cybersecurity knowledge base, here's what I found:\n"
        ]
        
        categories_covered = set()
        for doc in retrieved_docs[:3]:  # Use top 3 most relevant
            if doc['category'] not in categories_covered:
                response_parts.append(f"\n**{doc['title']}** (Relevance: {doc['score']:.0%})")
                response_parts.append(f"{doc['content'][:500]}...")
                categories_covered.add(doc['category'])
        
        response_parts.append(f"\n\n*Sources referenced: {', '.join(categories_covered)}*")
        
        return "\n".join(response_parts)
    
    def _call_api(self, prompt: str) -> str:
        """Call external LLM API"""
        import requests
        
        if not self.api_key:
            return "API key not configured"
        
        # Example with Gemini API
        API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"API Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# ==================== UTILITY FUNCTIONS ====================
def create_rag_system(api_key: str = None) -> Tuple[RAGEngine, RAGChatbot]:
    """Create and initialize a complete RAG system"""
    config = RAGConfig()
    rag_engine = RAGEngine(config)
    
    # Initialize with default knowledge base
    if EMBEDDINGS_AVAILABLE:
        rag_engine.initialize_knowledge_base()
    else:
        print("⚠️ Install sentence-transformers to enable RAG: pip install sentence-transformers")
    
    chatbot = RAGChatbot(rag_engine, api_key)
    
    return rag_engine, chatbot


def demo_rag_system():
    """Demonstrate RAG system capabilities"""
    print("=" * 60)
    print("🔍 RAG SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    if not EMBEDDINGS_AVAILABLE:
        print("\n❌ Cannot run demo: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return
    
    # Create RAG system
    rag_engine, chatbot = create_rag_system()
    
    # Test queries
    test_queries = [
        "How can I protect my organization from phishing attacks?",
        "What should I do if we're hit by ransomware?",
        "Explain the MITRE ATT&CK framework",
        "What are the best practices for cloud security?",
        "How do I detect insider threats?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"📝 Query: {query}")
        print("-" * 60)
        
        response_data = chatbot.get_response(query)
        
        print(f"📚 Retrieved {response_data['num_sources']} relevant sources:")
        for doc in response_data['retrieved_docs'][:3]:
            print(f"   - {doc['title']} ({doc['category']}) - Score: {doc['score']:.2f}")
        
        print(f"\n💬 Response:\n{response_data['response'][:500]}...")


# ==================== MAIN ====================
if __name__ == "__main__":
    demo_rag_system()
