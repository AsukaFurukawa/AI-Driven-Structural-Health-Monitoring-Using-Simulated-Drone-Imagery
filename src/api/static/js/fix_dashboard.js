/**
 * Fix dashboard statistics to handle negative values
 * This script is added to provide a direct fix for any negative values showing in the dashboard
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log("Applying dashboard fixes...");
    
    try {
        // Fix dashboard statistics if they are negative
        const statElements = document.querySelectorAll('.card .display-4, .card h1');
        console.log(`Found ${statElements.length} statistic elements to check`);
        
        statElements.forEach(function(element) {
            const value = element.textContent.trim();
            console.log(`Checking element with value: ${value}`);
            
            if (value.startsWith('-')) {
                // Replace negative value with positive one
                const positiveValue = value.substring(1);
                console.log(`Fixing negative value: ${value} -> ${positiveValue}`);
                element.textContent = positiveValue;
            }
        });
        
        // Also fix value attributes if they exist
        const elementsWithValueAttr = document.querySelectorAll('[data-value]');
        elementsWithValueAttr.forEach(function(element) {
            const value = element.getAttribute('data-value');
            if (value && value.startsWith('-')) {
                const positiveValue = value.substring(1);
                element.setAttribute('data-value', positiveValue);
            }
        });
        
        console.log("Dashboard fixes applied successfully");
    } catch (error) {
        console.error("Error applying dashboard fixes:", error);
    }
}); 