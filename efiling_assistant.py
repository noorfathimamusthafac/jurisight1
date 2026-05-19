"""
E-Filing Assistant Module for JuriSight
Provides DCMS e-filing guidance and automation
"""

import json
import os
from typing import Dict, List, Any

class EFilingAssistant:
    """Handles e-filing wizard logic and document generation"""
    
    def __init__(self):
        self.dcms_data = self.load_dcms_requirements()
    
    def load_dcms_requirements(self) -> Dict:
        """Load DCMS requirements from JSON file"""
        try:
            with open('data/dcms_requirements.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading DCMS requirements: {e}")
            return {}
    
    def get_case_types(self) -> List[Dict]:
        """Get list of available case types"""
        case_types = []
        for key, value in self.dcms_data.get('case_types', {}).items():
            case_types.append({
                'id': key,
                'name': value['name'],
                'description': value['description']
            })
        return case_types
    
    def get_case_requirements(self, case_type: str) -> Dict:
        """Get requirements for a specific case type"""
        return self.dcms_data.get('case_types', {}).get(case_type, {})
    
    def get_filing_steps(self, case_type: str) -> List[str]:
        """Get filing steps for a case type"""
        case_data = self.get_case_requirements(case_type)
        return case_data.get('filing_steps', [])
    
    def calculate_court_fee(self, case_type: str, claim_amount: float = None) -> Dict:
        """Calculate court fee based on case type and claim amount"""
        fee_structure = self.dcms_data.get('court_fee_structure', {}).get(case_type, {})
        
        if case_type == 'civil' and claim_amount:
            if claim_amount <= 5000:
                return {'fee': 50, 'description': 'Fixed fee for claims up to ₹5,000'}
            elif claim_amount <= 10000:
                return {'fee': 100, 'description': 'Fixed fee for claims ₹5,001 to ₹10,000'}
            elif claim_amount <= 50000:
                fee = claim_amount * 0.02
                return {'fee': fee, 'description': f'2% of claim amount (₹{claim_amount})'}
            elif claim_amount <= 100000:
                fee = claim_amount * 0.03
                return {'fee': fee, 'description': f'3% of claim amount (₹{claim_amount})'}
            else:
                fee = min(claim_amount * 0.04, 10000)
                return {'fee': fee, 'description': f'4% of claim amount (max ₹10,000)'}
        
        # For non-civil cases or when claim amount is not provided
        if isinstance(fee_structure, dict):
            # Return first fee option
            first_key = list(fee_structure.keys())[0] if fee_structure else None
            if first_key:
                return {'fee': fee_structure[first_key], 'description': f'Fee for {first_key}'}
        
        return {'fee': 0, 'description': 'No court fee required'}
    
    def get_step_guidance(self, case_type: str, step: int) -> str:
        """Get AI-powered guidance for a specific filing step"""
        steps = self.get_filing_steps(case_type)
        
        if step < 1 or step > len(steps):
            return "Invalid step number"
        
        current_step = steps[step - 1]
        case_info = self.get_case_requirements(case_type)
        
        # Build guidance prompt
        guidance_prompt = f"""
        You are a legal assistant helping a lawyer file a {case_info['name']} in DCMS.
        
        Current Step ({step}/{len(steps)}): {current_step}
        
        Provide clear, concise guidance for this step including:
        1. What needs to be done
        2. What documents/information are required
        3. Common mistakes to avoid
        4. Tips for successful completion
        
        Keep it practical and actionable.
        """
        
        return guidance_prompt
    
    def validate_filing(self, case_type: str, documents: List[str]) -> Dict:
        """Validate if all required documents are present"""
        requirements = self.get_case_requirements(case_type)
        mandatory = requirements.get('mandatory_documents', [])
        
        missing = []
        for doc in mandatory:
            if doc not in documents:
                missing.append(doc)
        
        return {
            'valid': len(missing) == 0,
            'missing_documents': missing,
            'total_required': len(mandatory),
            'provided': len([d for d in documents if d in mandatory])
        }
    
    def get_checklist(self, case_type: str) -> Dict:
        """Get complete filing checklist"""
        requirements = self.get_case_requirements(case_type)
        
        return {
            'case_type': requirements.get('name', ''),
            'mandatory_documents': requirements.get('mandatory_documents', []),
            'optional_documents': requirements.get('optional_documents', []),
            'filing_steps': requirements.get('filing_steps', []),
            'court_fee': requirements.get('court_fee', '0'),
            'validation_rules': self.dcms_data.get('validation_rules', {})
        }

# Global instance
efiling_assistant = None

def get_efiling_assistant():
    """Get or create e-filing assistant instance"""
    global efiling_assistant
    if efiling_assistant is None:
        efiling_assistant = EFilingAssistant()
    return efiling_assistant
