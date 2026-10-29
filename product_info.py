"""
Product Information Retrieval Module
Fetches product details from Open Food Facts API and LLM fallback
"""

import requests
import logging
from typing import Optional, Dict, Any
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductInfo:
    """Retrieves product information from APIs and LLM"""

    def __init__(self, config):
        self.config = config
        self.open_food_facts_url = config.OPEN_FOOD_FACTS_API_URL
        self.llm_provider = config.LLM_PROVIDER
        self.gemini_key = getattr(config, 'GEMINI_API_KEY', '')
        self.openai_key = config.OPENAI_API_KEY
        self.anthropic_key = config.ANTHROPIC_API_KEY

    def get_product_from_barcode(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Get product information from Open Food Facts API using barcode

        Args:
            barcode: Barcode number (EAN, UPC, etc.)

        Returns:
            Product information dictionary or None
        """
        try:
            # Clean barcode - remove any whitespace
            barcode = barcode.strip()

            if not barcode or len(barcode) < 8:
                logger.warning(f"Invalid barcode format: {barcode}")
                return None

            url = f"{self.open_food_facts_url}/{barcode}.json"
            logger.info(f"Querying Open Food Facts API for barcode: {barcode}")

            # Make API request with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=15, headers={
                        'User-Agent': 'GroceryDetectionSystem/1.0'
                    })

                    if response.status_code == 200:
                        data = response.json()

                        # Check if product was found
                        if data.get('status') == 1 and 'product' in data:
                            product = data['product']

                            # Extract product name - try multiple fields
                            product_name = (
                                product.get('product_name') or
                                product.get('product_name_en') or
                                product.get('product_name_fr') or
                                product.get('abbreviated_product_name') or
                                ''
                            )

                            # Clean product name
                            if product_name:
                                product_name = product_name.strip()

                            # Extract other information
                            brand = product.get('brands', '') or product.get(
                                'brand', '') or ''
                            if brand:
                                # Take first brand
                                brand = brand.split(',')[0].strip()

                            quantity = (
                                product.get('quantity') or
                                product.get('product_quantity') or
                                product.get('quantity_en') or
                                ''
                            )

                            description = (
                                product.get('generic_name') or
                                product.get('generic_name_en') or
                                product.get('categories_tags', [None])[0] if product.get('categories_tags') else '' or
                                ''
                            )

                            # Build product info
                            product_info = {
                                'name': product_name,
                                'brand': brand,
                                'quantity': quantity,
                                'description': description,
                                'categories': product.get('categories', ''),
                                'ingredients': product.get('ingredients_text', '') or product.get('ingredients_text_en', ''),
                                'source': 'open_food_facts',
                                'barcode': barcode
                            }

                            # Validate that we have at least a product name
                            if product_info['name'] and product_info['name'] not in ['', 'Unknown Product']:
                                logger.info(
                                    f"Successfully retrieved product info: {product_info['name']}")
                                return product_info
                            else:
                                logger.warning(
                                    f"Product found but name is missing/empty for barcode: {barcode}")
                        else:
                            logger.warning(
                                f"Product not found in database for barcode: {barcode} (status: {data.get('status')})")

                    elif response.status_code == 404:
                        logger.warning(
                            f"Barcode not found in Open Food Facts database: {barcode}")
                        break
                    else:
                        logger.warning(
                            f"API returned status code {response.status_code} for barcode: {barcode}")

                except requests.exceptions.Timeout:
                    logger.warning(
                        f"Timeout fetching product info for barcode {barcode} (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        return None
                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"Request error for barcode {barcode}: {str(e)}")
                    if attempt == max_retries - 1:
                        return None

            return None

        except Exception as e:
            logger.error(
                f"Error fetching product info for barcode {barcode}: {str(e)}")
            return None

    def get_product_from_llm(self, product_name: str, context: str = "", detection_type: str = "object") -> Optional[Dict[str, Any]]:
        """
        Get product information from LLM when API data is insufficient or detection confidence is low

        Args:
            product_name: Name of the product or barcode
            context: Additional context (e.g., "detected as apple with 35% confidence")
            detection_type: Type of detection ("barcode" or "object")

        Returns:
            Product information dictionary or None
        """
        try:
            # Create a more detailed and structured prompt
            if detection_type == "barcode":
                prompt = f"""You are a grocery product information assistant. A barcode was scanned: {product_name}

{context}

Please provide accurate product information in JSON format with the following structure:
{{
    "name": "Full product name (e.g., 'Coca Cola Can 250ml')",
    "brand": "Brand name if known, else empty string",
    "quantity": "Quantity/size if mentioned, else empty string",
    "description": "Brief description of the product (1 sentence)"
}}

Important:
- Be specific and accurate
- If barcode is provided, try to identify the actual product
- If uncertain, provide a generic but helpful description
- Return ONLY valid JSON, no additional text"""
            else:
                prompt = f"""You are a grocery product information assistant. A product was detected: {product_name}

{context}

Please provide accurate product information in JSON format with the following structure:
{{
    "name": "Full product name",
    "brand": "Brand name if applicable, else empty string",
    "quantity": "Quantity/size if applicable, else empty string",
    "description": "Brief description of the product - what it is, common use (1-2 sentences)"
}}

Important:
- Be accurate and specific about the product
- Provide helpful information for a blind user shopping
- If the detected name seems incorrect, provide the most likely correct product information
- Return ONLY valid JSON, no additional text or markdown"""

            if self.llm_provider == "gemini" and self.gemini_key:
                return self._get_from_gemini(prompt, product_name)
            elif self.llm_provider == "openai" and self.openai_key:
                return self._get_from_openai(prompt, product_name)
            elif self.llm_provider == "anthropic" and self.anthropic_key:
                return self._get_from_anthropic(prompt, product_name)
            else:
                # Fallback to a simple description
                logger.warning(
                    "No LLM API key configured, using fallback description")
                return {
                    'name': product_name,
                    'brand': '',
                    'quantity': '',
                    'description': f"A grocery product: {product_name.lower()}",
                    'source': 'llm_fallback'
                }

        except Exception as e:
            logger.error(f"Error getting product info from LLM: {str(e)}")
            return None

    def _get_from_gemini(self, prompt: str, product_name: str = "Unknown") -> Optional[Dict[str, Any]]:
        """Get product info from Google Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)

            logger.info(f"Querying Gemini for product: {product_name}")

            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=300,
                )
            )

            content = response.text.strip()

            # Try to parse JSON from response
            try:
                # Remove markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                product_info = json.loads(content)

                if 'name' not in product_info:
                    product_info['name'] = product_name

                product_info['source'] = 'gemini_llm'
                logger.info(f"Successfully retrieved product info from Gemini: {product_info.get('name')}")
                return product_info

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from Gemini response: {e}")
                return {
                    'name': product_name,
                    'brand': '',
                    'quantity': '',
                    'description': content[:200] if content else f"A grocery product: {product_name}",
                    'source': 'gemini_llm'
                }

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return {
                'name': product_name,
                'brand': '',
                'quantity': '',
                'description': f"A grocery product: {product_name}",
                'source': 'gemini_llm_fallback'
            }

    def _get_from_openai(self, prompt: str, product_name: str = "Unknown") -> Optional[Dict[str, Any]]:
        """Get product info from OpenAI API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)

            logger.info(f"Querying OpenAI for product: {product_name}")

            # Try gpt-4o-mini first, fallback to gpt-3.5-turbo if not available
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Using more capable model for better accuracy
                    messages=[
                        {"role": "system", "content": "You are a helpful grocery product information assistant. Always respond with valid JSON only, no markdown formatting, no code blocks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # Lower temperature for more consistent, accurate responses
                    max_tokens=300,
                    # Force JSON response
                    response_format={"type": "json_object"}
                )
            except Exception as model_error:
                # Fallback to gpt-3.5-turbo if gpt-4o-mini is not available
                logger.warning(
                    f"gpt-4o-mini not available, using gpt-3.5-turbo: {model_error}")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful grocery product information assistant. Always respond with valid JSON only, no markdown formatting, no code blocks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON from response
            try:
                # Remove markdown code blocks if present (though shouldn't be with json_object format)
                if "```json" in content:
                    content = content.split("```json")[
                        1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                product_info = json.loads(content)

                # Validate required fields
                if 'name' not in product_info:
                    product_info['name'] = product_name

                product_info['source'] = 'openai_llm'
                logger.info(
                    f"Successfully retrieved product info from OpenAI: {product_info.get('name')}")
                return product_info

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON from OpenAI response: {e}")
                logger.warning(f"Response content: {content}")
                # If JSON parsing fails, try to extract information
                return {
                    'name': product_name,
                    'brand': '',
                    'quantity': '',
                    'description': content[:200] if content else f"A grocery product: {product_name}",
                    'source': 'openai_llm'
                }

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            # Fallback to simple description
            return {
                'name': product_name,
                'brand': '',
                'quantity': '',
                'description': f"A grocery product: {product_name}",
                'source': 'openai_llm_fallback'
            }

    def _get_from_anthropic(self, prompt: str, product_name: str = "Unknown") -> Optional[Dict[str, Any]]:
        """Get product info from Anthropic Claude API"""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.anthropic_key)

            logger.info(
                f"Querying Anthropic Claude for product: {product_name}")

            message = client.messages.create(
                # Using more capable model for better accuracy
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = message.content[0].text.strip()

            # Try to parse JSON
            try:
                # Remove markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[
                        1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                product_info = json.loads(content)

                # Validate required fields
                if 'name' not in product_info:
                    product_info['name'] = product_name

                product_info['source'] = 'anthropic_llm'
                logger.info(
                    f"Successfully retrieved product info from Claude: {product_info.get('name')}")
                return product_info

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON from Claude response: {e}")
                logger.warning(f"Response content: {content}")
                return {
                    'name': product_name,
                    'brand': '',
                    'quantity': '',
                    'description': content[:200] if content else f"A grocery product: {product_name}",
                    'source': 'anthropic_llm'
                }

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            # Fallback to simple description
            return {
                'name': product_name,
                'brand': '',
                'quantity': '',
                'description': f"A grocery product: {product_name}",
                'source': 'anthropic_llm_fallback'
            }

    def format_product_description(self, product_info: Dict[str, Any]) -> str:
        """
        Format product information into audio-friendly description

        Args:
            product_info: Product information dictionary

        Returns:
            Formatted string description
        """
        if not product_info:
            return "Product information not available"

        name = product_info.get('name', 'Unknown product')
        brand = product_info.get('brand', '')
        quantity = product_info.get('quantity', '')
        description = product_info.get('description', '')

        # Build description
        parts = []

        if brand and brand.lower() not in name.lower():
            parts.append(brand)

        parts.append(name)

        if quantity:
            parts.append(f", {quantity}")

        if description and description != name:
            parts.append(f", {description}")

        return "".join(parts) if parts else name
