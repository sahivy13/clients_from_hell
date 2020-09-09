import requests, time
from bs4 import BeautifulSoup as bs
import re

class IronhackSpider:
    
    def __init__(self, url_pattern, pages_to_scrape=1, sleep_interval=-1, content_parser=None):
        self.url_pattern = url_pattern
        self.pages_to_scrape = pages_to_scrape
        self.sleep_interval = sleep_interval
        self.content_parser = content_parser
  
    def scrape_url(self, url):
        response = requests.get(url)
        result = self.content_parser(response.content)
        return result
            
    def kickstart(self):
        list_pages = []
        for i in range(1, self.pages_to_scrape+1):
            list_pages.append(self.scrape_url(self.url_pattern % i))            
        return list_pages

def get_categories(url):
    html = requests.get(url)
    response = bs(html.content, features="html.parser")
    get_items = [category for category in response.find_all('li', {'class':'flex items-center'})]
    categories = ['Dunces','Criminals','Deadbeats','Racists','Homophobes','Sexist','Frenemies','Cryptic','Ingrates','Chaotic Good']
    category_pair = []
    for item in get_items:
        href = item.find('a').get('href')
        item_name = re.sub('\\n','',item.text)
        pair = (item_name, href)
        if item_name in categories:
            category_pair.append(pair)
    return list(set(category_pair))

def url_categroy_creator(list_categories):

    list_url_patters = []

    for cat in list_categories:
        pattern = 'https://clientsfromhell.net'+cat[1]+'page/' # regex pattern for the urls to scrape
        list_url_patters.append((pattern,cat[0]))

    return list_url_patters
        
def page_num_creator(url_category_list : list):
    list_url_num =[]
    for url in url_category_list:
        html = requests.get(url[0]+'1')
        response = bs(html.content, "html.parser")
        list_items = response.find_all('a',{'class':'page-numbers'})

        len_=len(list_items)-2
        max_pag=list_items[len_].text
        list_url_num.append((url[0],max_pag,url[1]))
    return list_url_num

def case_parser(content):
    all_content = bs(content, "html.parser")
    pre_content = all_content.select('div [class="w-blog-post-content"] > p')
    
    case=[]
    
    for i, el in enumerate(pre_content):
        text = el.text
        case.append(text)

    return case

def initialize_scraping(url_pagenum_cat_list : list):
    
    html_cont_dict = {}

    for URL_PATTERN, PAGES_TO_SCRAPE, CAT in url_pagenum_cat_list:

        my_spider = IronhackSpider(URL_PATTERN+'%s/', int(PAGES_TO_SCRAPE), content_parser=case_parser)

        content = my_spider.kickstart()
        
        html_cont_dict.update({CAT: content})
        
    return html_cont_dict