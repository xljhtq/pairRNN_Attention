# coding:utf-8
import os
import sys
import re
import scrapy
import pymysql
import logging
import time
from urllib import quote


def load_properties(path):
    properties = []
    with open(path, "r") as f:
        for line in f:
            properties.append(line.strip())
    return properties


class QuotesSpider(scrapy.Spider):
    name = "zhidao"

    def parse_question_url(self, response):
        '''
            知道问题页面，爬取问题、答案和相似问题
        '''
        # ==== 解析 ====
        url = response.url
        page_id = re.search(r'([\w]+)\.html', url).group(1)

        # 相似问题
        question = "".join(response.xpath('//span[@class="ask-title "]/text()').extract())
        extensions = response.xpath('//span[@class="related-restrict-title grid"]/text()').extract()
        ext_dianzan = response.xpath('//span[@class="ml-5 ff-arial"]/text()').extract()
        ext_urls = response.xpath('//a[@class="related-link"]').re('href="(.*?)"')
        ext_ids = [re.search(r'([\w]+)\.html', url).group(1) for url in ext_urls]

        extensions = "\n".join(extensions)
        ext_ids = "\n".join(ext_ids)
        ext_dianzan = "\n".join(ext_dianzan)

        # 最佳答案
        best_answer = "".join(response.xpath('//pre[@class="best-text mb-10"]/text()').extract())

        # 点评
        if len(best_answer) > 0:
            eval_good, eval_bad = response.xpath('//div[@class="qb-zan-eva"]/span').re('data-evaluate="(.*?)"')[:2]
            evaluation = eval_good + '\n' + eval_bad
        else:
            evaluation = '0\n0'

        # ==== 存入数据库 ====
        db = pymysql.connect("localhost", "root", "zaqxsw", "nongHang", charset="utf8")
        cursor = db.cursor()
        sql = """ UPDATE zhidao_question SET question='%s',extensions = '%s', dianzan='%s',ext_ids = '%s', answer = '%s', evaluation='%s' WHERE id = '%s' """ % (
            question, extensions, ext_dianzan, ext_ids, best_answer, evaluation, page_id)
        try:
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()
            logging.exception("")
        db.close()

    def start_requests(self):
        # ==== 从数据库读取url =====
        dic_id = {}
        while True:
            question_urls = []
            db = pymysql.connect("localhost", "root", "zaqxsw", "nongHang", charset="utf8")
            cursor = db.cursor()

            sql = """ SELECT url,id FROM zhidao_question """
            try:
                cursor.execute(sql)
                results = cursor.fetchall()
                for row in results:
                    url = row[0]
                    id = row[1]
                    if id not in dic_id:
                        dic_id[id] = 1
                        question_urls.append(url)
            except:
                db.rollback()
                logging.exception("")
            db.close()
            if question_urls:
                for url in question_urls:
                    yield scrapy.Request(url=url, callback=self.parse_question_url)
            else:
                print("question_urls 无更新")
                time.sleep(5)
