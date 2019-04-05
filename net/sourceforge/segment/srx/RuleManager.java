package net.sourceforge.segment.srx;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import net.sourceforge.segment.util.Util;

/**
 * Represents segmentation rules manager.
 * Responsible for constructing and storing break and exception rules.
 * 
 * @author loomchild
 */
public class RuleManager {
	
	@SuppressWarnings("unused")
	private SrxDocument document;
	
	private int maxLookbehindConstructLength;

	private List<Rule> breakRuleList;
	
	private Map<Rule, Pattern> exceptionPatternMap;
	
	/**
	 * Constructor. Responsible for retrieving rules from SRX document for
	 * given language code, constructing patterns and storing them in 
	 * quick accessible format.
	 * Adds break rules to {@link #breakRuleList} and constructs
	 * corresponding exception patterns in {@link #exceptionPatternMap}.  
	 * Uses document cache to store rules and patterns. 
	 * @param document SRX document
	 * @param languageRuleList list of language rules
	 * @param maxLookbehindConstructLength Maximum length of regular expression in lookbehind (see {@link Util#finitize(String, int)}).
	 */
	public RuleManager(SrxDocument document, List<LanguageRule> languageRuleList, 
			int maxLookbehindConstructLength) {
		
		this.document = document;
		this.maxLookbehindConstructLength = maxLookbehindConstructLength;
		
		this.breakRuleList = new ArrayList<Rule>();
		this.exceptionPatternMap = new HashMap<Rule, Pattern>();

		StringBuilder exceptionPatternBuilder = new StringBuilder();
		
		for (LanguageRule languageRule : languageRuleList) {
			for (Rule rule : languageRule.getRuleList()) {

				if (rule.isBreak()) {
				
					breakRuleList.add(rule);
					
					Pattern exceptionPattern;
					
					if (exceptionPatternBuilder.length() > 0) {
						String exceptionPatternString = 
							exceptionPatternBuilder.toString();
						exceptionPattern = 
							Util.compile(document, exceptionPatternString);
					} else {
						exceptionPattern = null;
					}

					exceptionPatternMap.put(rule, exceptionPattern);
				
				} else {
				
					if (exceptionPatternBuilder.length() > 0) {
						exceptionPatternBuilder.append('|');
					}

					String patternString = createExceptionPatternString(rule);
					
					exceptionPatternBuilder.append(patternString);
			
				}
			
			}
		}
		
	}
	
	/**
	 * @return break rule list
	 */
	public List<Rule> getBreakRuleList() {
		return breakRuleList;
	}
	
	/**
	 * @param breakRule
	 * @return exception pattern corresponding to give break rule
	 */
	public Pattern getExceptionPattern(Rule breakRule) {
		return exceptionPatternMap.get(breakRule);
	}
	
	/**
	 * Creates exception pattern string that can be matched in the place 
	 * where break rule was matched. Both parts of the rule 
	 * (beforePattern and afterPattern) are incorporated
	 * into one pattern.
	 * beforePattern is used in lookbehind, therefore it needs to be 
	 * modified so it matches finite string (contains no *, + or {n,}). 
	 * @param rule exception rule
	 * @return string containing exception pattern
	 */
	private String createExceptionPatternString(Rule rule) {
		
		StringBuilder patternBuilder = new StringBuilder();
		
		// As Java does not allow infinite length patterns
		// in lookbehind, before pattern need to be shortened.
		String beforePattern = 
			Util.finitize(rule.getBeforePattern(), maxLookbehindConstructLength);
		String afterPattern = rule.getAfterPattern();
		
		patternBuilder.append("(?:");
		if (beforePattern.length() > 0) {
			patternBuilder.append("(?<=" + beforePattern + ")");
		}
		if (afterPattern.length() > 0) {
			patternBuilder.append("(?=" + afterPattern + ")");
		}
		patternBuilder.append(")");
		
		return patternBuilder.toString();
		
	}
	
}
