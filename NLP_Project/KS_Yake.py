import yake
from yake import KeywordExtractor


def YAKE_AL(text):
    kw_extractor = KeywordExtractor()
    print(kw_extractor)
    keywords = kw_extractor.extract_keywords(text)
    print(keywords)
    for kw in keywords:
        print(f"keyword: {kw[0]} - Score: {kw[1]}")



if __name__=="__main__":
    text="""Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning "\
"competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud "\
"Next conference in San Francisco this week, the official announcement could come as early as tomorrow. "\
"Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. "\
"Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, "\
"was founded by Goldbloom  and Ben Hamner in 2010. """
    YAKE_AL(text)	






