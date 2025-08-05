"""Internationalization (i18n) support for carbon-aware trainer."""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class SupportedLanguage(str, Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    PORTUGUESE = "pt"
    KOREAN = "ko"
    ITALIAN = "it"
    DUTCH = "nl"


@dataclass
class LocaleInfo:
    """Information about a locale."""
    code: str
    name: str
    native_name: str
    rtl: bool = False  # Right-to-left text direction
    decimal_separator: str = "."
    thousands_separator: str = ","
    currency_symbol: str = "$"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"


class TranslationManager:
    """Manages translations and localization."""
    
    def __init__(self, default_language: str = "en", translations_dir: Optional[Path] = None):
        """Initialize translation manager.
        
        Args:
            default_language: Default language code
            translations_dir: Directory containing translation files
        """
        self.default_language = default_language
        self.current_language = default_language
        
        # Set translations directory
        if translations_dir:
            self.translations_dir = Path(translations_dir)
        else:
            self.translations_dir = Path(__file__).parent.parent / "translations"
        
        # Translation cache
        self._translations: Dict[str, Dict[str, str]] = {}
        
        # Locale information
        self._locales = self._initialize_locales()
        
        # Load translations
        self._load_translations()
    
    def _initialize_locales(self) -> Dict[str, LocaleInfo]:
        """Initialize locale information."""
        return {
            "en": LocaleInfo("en", "English", "English"),
            "es": LocaleInfo("es", "Spanish", "Español"),
            "fr": LocaleInfo("fr", "French", "Français"),
            "de": LocaleInfo("de", "German", "Deutsch"),
            "ja": LocaleInfo("ja", "Japanese", "日本語"),
            "zh": LocaleInfo("zh", "Chinese (Simplified)", "简体中文"),
            "pt": LocaleInfo("pt", "Portuguese", "Português"),
            "ko": LocaleInfo("ko", "Korean", "한국어"),
            "it": LocaleInfo("it", "Italian", "Italiano"),
            "nl": LocaleInfo("nl", "Dutch", "Nederlands")
        }
    
    def _load_translations(self) -> None:
        """Load translation files."""
        if not self.translations_dir.exists():
            logger.warning(f"Translations directory not found: {self.translations_dir}")
            return
        
        for lang_file in self.translations_dir.glob("*.json"):
            lang_code = lang_file.stem
            
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                    self._translations[lang_code] = translations
                    
                logger.debug(f"Loaded translations for {lang_code}: {len(translations)} keys")
                
            except Exception as e:
                logger.error(f"Failed to load translations for {lang_code}: {e}")
    
    def set_language(self, language_code: str) -> bool:
        """Set current language.
        
        Args:
            language_code: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if language was set successfully
        """
        if language_code not in self._locales:
            logger.warning(f"Unsupported language: {language_code}")
            return False
        
        self.current_language = language_code
        logger.info(f"Language set to: {language_code}")
        return True
    
    def get_available_languages(self) -> List[Dict[str, str]]:
        """Get list of available languages.
        
        Returns:
            List of language information dictionaries
        """
        available = []
        
        for code, locale in self._locales.items():
            has_translations = code in self._translations
            
            available.append({
                "code": code,
                "name": locale.name,
                "native_name": locale.native_name,
                "has_translations": has_translations,
                "is_current": code == self.current_language
            })
        
        return available
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a message key.
        
        Args:
            key: Translation key
            **kwargs: Variables for string formatting
            
        Returns:
            Translated message
        """
        # Try current language first
        translation = self._get_translation(self.current_language, key)
        
        # Fallback to default language
        if translation is None and self.current_language != self.default_language:
            translation = self._get_translation(self.default_language, key)
        
        # Fallback to key itself
        if translation is None:
            translation = key
            logger.debug(f"Missing translation for key: {key}")
        
        # Format with variables if provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation formatting failed for key {key}: {e}")
        
        return translation
    
    def _get_translation(self, language: str, key: str) -> Optional[str]:
        """Get translation for specific language and key.
        
        Args:
            language: Language code
            key: Translation key
            
        Returns:
            Translation string or None if not found
        """
        if language not in self._translations:
            return None
        
        # Support nested keys with dot notation (e.g., "error.network.timeout")
        translation_dict = self._translations[language]
        
        for part in key.split('.'):
            if isinstance(translation_dict, dict) and part in translation_dict:
                translation_dict = translation_dict[part]
            else:
                return None
        
        return translation_dict if isinstance(translation_dict, str) else None
    
    def get_locale_info(self, language_code: Optional[str] = None) -> Optional[LocaleInfo]:
        """Get locale information.
        
        Args:
            language_code: Language code (uses current if None)
            
        Returns:
            Locale information or None if not found
        """
        code = language_code or self.current_language
        return self._locales.get(code)
    
    def format_number(self, number: float, decimals: int = 2, language_code: Optional[str] = None) -> str:
        """Format number according to locale conventions.
        
        Args:
            number: Number to format
            decimals: Number of decimal places
            language_code: Language code (uses current if None)
            
        Returns:
            Formatted number string
        """
        locale = self.get_locale_info(language_code)
        if not locale:
            locale = self.get_locale_info(self.default_language)
        
        # Format number with specified decimals
        formatted = f"{number:.{decimals}f}"
        
        # Split integer and decimal parts
        if locale.decimal_separator in formatted:
            integer_part, decimal_part = formatted.split('.')
            decimal_formatted = locale.decimal_separator + decimal_part
        else:
            integer_part = formatted
            decimal_formatted = ""
        
        # Add thousands separators
        if len(integer_part) > 3:
            # Add separator every 3 digits from right
            parts = []
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    parts.append(locale.thousands_separator)
                parts.append(digit)
            integer_part = ''.join(reversed(parts))
        
        return integer_part + decimal_formatted
    
    def format_carbon_amount(self, amount_kg: float, language_code: Optional[str] = None) -> str:
        """Format carbon amount with appropriate units and locale.
        
        Args:
            amount_kg: Carbon amount in kilograms
            language_code: Language code
            
        Returns:
            Formatted carbon amount string
        """
        if amount_kg < 0.001:
            # Show in grams
            amount_g = amount_kg * 1000
            formatted_amount = self.format_number(amount_g, decimals=1, language_code=language_code)
            unit_key = "units.carbon.grams"
        elif amount_kg < 1000:
            # Show in kilograms
            formatted_amount = self.format_number(amount_kg, decimals=3, language_code=language_code)
            unit_key = "units.carbon.kilograms"
        else:
            # Show in metric tons
            amount_t = amount_kg / 1000
            formatted_amount = self.format_number(amount_t, decimals=2, language_code=language_code)
            unit_key = "units.carbon.tons"
        
        unit = self.translate(unit_key)
        return f"{formatted_amount} {unit}"
    
    def format_energy_amount(self, amount_kwh: float, language_code: Optional[str] = None) -> str:
        """Format energy amount with appropriate units.
        
        Args:
            amount_kwh: Energy amount in kWh
            language_code: Language code
            
        Returns:
            Formatted energy amount string
        """
        if amount_kwh < 1:
            # Show in Wh
            amount_wh = amount_kwh * 1000
            formatted_amount = self.format_number(amount_wh, decimals=0, language_code=language_code)
            unit = self.translate("units.energy.watt_hours")
        elif amount_kwh < 1000:
            # Show in kWh
            formatted_amount = self.format_number(amount_kwh, decimals=2, language_code=language_code)
            unit = self.translate("units.energy.kilowatt_hours")
        else:
            # Show in MWh
            amount_mwh = amount_kwh / 1000
            formatted_amount = self.format_number(amount_mwh, decimals=2, language_code=language_code)
            unit = self.translate("units.energy.megawatt_hours")
        
        return f"{formatted_amount} {unit}"
    
    def format_duration(self, duration_seconds: float, language_code: Optional[str] = None) -> str:
        """Format duration in human-readable form.
        
        Args:
            duration_seconds: Duration in seconds
            language_code: Language code
            
        Returns:
            Formatted duration string
        """
        if duration_seconds < 60:
            # Seconds only
            formatted = self.format_number(duration_seconds, decimals=1, language_code=language_code)
            unit = self.translate("units.time.seconds")
            return f"{formatted} {unit}"
        
        elif duration_seconds < 3600:
            # Minutes and seconds
            minutes = int(duration_seconds // 60)
            seconds = duration_seconds % 60
            
            if seconds < 1:
                return f"{minutes} {self.translate('units.time.minutes')}"
            else:
                return f"{minutes} {self.translate('units.time.minutes')} {seconds:.0f} {self.translate('units.time.seconds')}"
        
        elif duration_seconds < 86400:
            # Hours and minutes
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            
            if minutes < 1:
                return f"{hours} {self.translate('units.time.hours')}"
            else:
                return f"{hours} {self.translate('units.time.hours')} {minutes} {self.translate('units.time.minutes')}"
        
        else:
            # Days and hours
            days = int(duration_seconds // 86400)
            hours = int((duration_seconds % 86400) // 3600)
            
            if hours < 1:
                return f"{days} {self.translate('units.time.days')}"
            else:
                return f"{days} {self.translate('units.time.days')} {hours} {self.translate('units.time.hours')}"


# Global translation manager instance
_translation_manager: Optional[TranslationManager] = None


def get_translation_manager() -> TranslationManager:
    """Get global translation manager instance.
    
    Returns:
        Global translation manager
    """
    global _translation_manager
    
    if _translation_manager is None:
        # Initialize with environment language preference
        default_lang = os.getenv('CARBON_AWARE_LANGUAGE', 'en')
        _translation_manager = TranslationManager(default_language=default_lang)
    
    return _translation_manager


def set_language(language_code: str) -> bool:
    """Set global language.
    
    Args:
        language_code: Language code
        
    Returns:
        True if language was set successfully
    """
    manager = get_translation_manager()
    return manager.set_language(language_code)


def translate(key: str, **kwargs) -> str:
    """Translate message key using global manager.
    
    Args:
        key: Translation key
        **kwargs: Formatting variables
        
    Returns:
        Translated message
    """
    manager = get_translation_manager()
    return manager.translate(key, **kwargs)


def format_carbon(amount_kg: float) -> str:
    """Format carbon amount using global manager.
    
    Args:
        amount_kg: Carbon amount in kg
        
    Returns:
        Formatted carbon amount
    """
    manager = get_translation_manager() 
    return manager.format_carbon_amount(amount_kg)


def format_energy(amount_kwh: float) -> str:
    """Format energy amount using global manager.
    
    Args:
        amount_kwh: Energy amount in kWh
        
    Returns:
        Formatted energy amount
    """
    manager = get_translation_manager()
    return manager.format_energy_amount(amount_kwh)


def format_duration(duration_seconds: float) -> str:
    """Format duration using global manager.
    
    Args:
        duration_seconds: Duration in seconds
        
    Returns:
        Formatted duration
    """
    manager = get_translation_manager()
    return manager.format_duration(duration_seconds)