from icrawler.builtin import BaiduImageCrawler
keyword=input("输入你想找图片的标签：")
google_storage = {'root_dir': r'./data'}
google_crawler = BaiduImageCrawler(parser_threads=4, 
                                   downloader_threads=4, 
                                   storage=google_storage)
google_crawler.crawl(keyword=keyword, 
                     max_num=1000)