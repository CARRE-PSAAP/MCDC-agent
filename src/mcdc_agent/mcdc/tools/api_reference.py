from pathlib import Path
import re

class APIReference:
    """Helper to access MCDC API documentation."""
    
    def __init__(self):
        # Look for mcdc_api_reference.md in the same directory
        self.path = Path(__file__).parent / "mcdc_api_reference.md"
        self.content = self.path.read_text() if self.path.exists() else ""
        
    def get_full_text(self) -> str:
        return self.content
        
    def get_section(self, header: str) -> str:
        """Get content of a specific markdown section (e.g. '## Materials').
        
        Includes all subsections until the next header of same or higher level.
        """
        if not self.content:
            return ""
        
        # Determine the header level (number of #)
        header_stripped = header.lstrip()
        header_level = len(header_stripped) - len(header_stripped.lstrip('#'))
        if header_level == 0:
            header_level = 2  # Default to ## if no # provided
        
        clean_header = header.lstrip('#').strip()
        
        # Find the header line
        lines = self.content.split('\n')
        start_idx = None
        
        for i, line in enumerate(lines):
            line_stripped = line.lstrip()
            if line_stripped.startswith('#'):
                # Count the header level of this line
                line_level = len(line_stripped) - len(line_stripped.lstrip('#'))
                # Check if this matches our target header (case insensitive)
                line_title = line_stripped.lstrip('#').strip()
                if line_title.lower() == clean_header.lower():
                    start_idx = i + 1  # Start after the header line
                    break
        
        if start_idx is None:
            return ""
        
        # Find the end - next header of same or higher level (fewer or equal #)
        # Must track whether we're inside a code block to ignore # in code
        end_idx = len(lines)
        in_code_block = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            line_stripped = line.lstrip()
            
            # Track code block state
            if line_stripped.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # Only check for headers when not inside a code block
            if not in_code_block and line_stripped.startswith('#'):
                line_level = len(line_stripped) - len(line_stripped.lstrip('#'))
                # Stop at same level or higher (fewer #)
                if line_level <= header_level:
                    end_idx = i
                    break
        
        # Extract and return the section content
        section_lines = lines[start_idx:end_idx]
        return '\n'.join(section_lines).strip()

    def get_relevant_sections(self, query: str) -> str:
        """Get API sections relevant to a query (error message or step name)."""
        sections = []
        query_lower = query.lower()
        
        # Map keywords to sections in mcdc_api_reference.md
        keywords = {
            "surface": "Surfaces",
            "cell": "Cells and Regions",
            "material": "Materials",
            "source": "Source",
            "tally": "Tallies",
            "setting": "Settings",
            "lattice": "Universe & Lattice",
            "universe": "Universe & Lattice",
            "region": "Cells and Regions",
            "eigen": "Settings",
        }
        
        for kw, section_title in keywords.items():
            if kw in query_lower:
                content = self.get_section(section_title)
                if content:
                    sections.append(f"## {section_title}\n{content}")
        
        return "\n\n".join(sections)
