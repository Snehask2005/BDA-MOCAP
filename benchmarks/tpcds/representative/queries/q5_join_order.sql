SELECT
    d.d_year,
    s.s_state,
    i.i_category,
    cd.cd_gender,
    SUM(ss.ss_ext_sales_price) AS total_sales
FROM store_sales ss
JOIN date_dim d
    ON ss.ss_sold_date_sk = d.d_date_sk
JOIN store s
    ON ss.ss_store_sk = s.s_store_sk
JOIN item i
    ON ss.ss_item_sk = i.i_item_sk
JOIN customer_demographics cd
    ON ss.ss_cdemo_sk = cd.cd_demo_sk
WHERE d.d_year BETWEEN 2000 AND 2002
  AND s.s_state IS NOT NULL
  AND i.i_category IS NOT NULL
  AND cd.cd_gender IS NOT NULL
GROUP BY
    d.d_year,
    s.s_state,
    i.i_category,
    cd.cd_gender
