USE nongHang;

DROP TABLE IF EXISTS `zhidao_question`;
CREATE TABLE `zhidao_question` (
    `id`            varchar(50)     NOT NULL,
     dianzan        VARCHAR(50),
    `question`      varchar(500),
    `url`           varchar(500)    NOT NULL,
    `extensions`    varchar(5000)   NULL,
    `ext_ids`       varchar(5000)   NULL,
    `answer`        varchar(5000)   NULL,
    `evaluation`    varchar(50)     NULL,
    `lastupdate`    timestamp       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;
