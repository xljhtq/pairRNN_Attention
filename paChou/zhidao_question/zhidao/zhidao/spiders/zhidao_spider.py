# coding:utf-8
import os
import sys
import re
import scrapy
import pymysql
import logging
from urllib import quote

def load_properties(path):
    properties = []
    with open(path, "r") as f:
        for line in f:
            properties.append(line.strip())
    return properties

def load_bank_names(path):
    bank_names = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().strip("\n").split("\t")
            bank_names.append(line[1])
    return bank_names

class QuotesSpider(scrapy.Spider):
    name = "zhidao"

    def start_requests(self):
        # ===== 基础设定 ====
        seed_url = "https://zhidao.baidu.com/search?lm=0&rn=10&pn=%d&fr=search&ie=utf8&word=%s"
        path_this_dir = os.path.dirname(os.path.abspath(__file__))
        path_bank_names = os.path.join(path_this_dir, "nongHangFAQ.txt")
        bank_names = load_bank_names(path_bank_names)
        page_urls = []
        for bank_name in bank_names:
            for i in range(1, 4):
                url = seed_url % (i * 10, quote(bank_name))
                page_urls.append(url)

        # ==== 先爬取每个问题的url =====
        for url in page_urls:
            yield scrapy.Request(url=url, callback=self.parse_page_url)

    def parse_page_url(self, response):
        '''
            知道索引页面，爬取每个问题和问题的url
        '''
        # ==== 解析 ====
        # questions = response.xpath('//a[@class="ti"]').re('([\u4e00-\u9fa5].*[\u4e00-\u9fa5])')
        # questions = response.xpath('//a[@class="ti"]/text()').extract()
        urls = response.xpath('//a[@class="ti"]').re('href="(.*?)"')

        # ==== 存入数据库 ====
        db = pymysql.connect("localhost", "root", "zaqxsw", "nongHang", charset="utf8")
        cursor = db.cursor()

        for url in urls:
            page_id = re.search(r'([\w]+)\.html', url).group(1)
            sql = """ INSERT INTO zhidao_question (id, url) VALUES ('%s', '%s')""" % (page_id, url)

            try:
                cursor.execute(sql)
                db.commit()
            except Exception as ex:
                db.rollback()
                logging.exception("")
        db.close()


