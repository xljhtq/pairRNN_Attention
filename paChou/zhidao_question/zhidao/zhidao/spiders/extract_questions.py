# coding:utf-8
import pymysql
import logging


def extract():
    outOp = open("mysql_nongHang.txt", "w")
    db = pymysql.connect("localhost", "root", "zaqxsw", "nongHang", charset="utf8")
    cursor = db.cursor()
    sql = """ SELECT question, extensions,dianzan FROM zhidao_question """

    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            question = row[0]
            extensions = row[1]
            dianzan = row[2]
            if question:
                if extensions:
                    extensions = extensions.strip().split('\n')
                    dianzan = dianzan.strip().split("\n")
                    extensionStr = " ".join(extensions)
                    dianzan = " ".join(dianzan)
                    outOp.write("1" + "\t" + extensionStr.encode("utf8") + "\t" + question.encode("utf8")
                                + "\t" + dianzan.encode("utf8") + "\n")
            else:
                print("empty")
    except:
        db.rollback()
        logging.exception("")

    db.close()
    outOp.close()


def main():
    extract()


if __name__ == "__main__":
    main()
