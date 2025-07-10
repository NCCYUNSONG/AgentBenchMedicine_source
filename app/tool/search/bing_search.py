from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import random
from app.logger import logger
from app.tool.search.base import SearchItem, WebSearchEngine
import time

ABSTRACT_MAX_LENGTH = 300

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/49.0.2623.108 Chrome/49.0.2623.108 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; pt-BR) AppleWebKit/533.3 (KHTML, like Gecko) QtWeb Internet Browser/3.7 http://www.QtWeb.net",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.2 (KHTML, like Gecko) ChromePlus/4.0.222.3 Chrome/4.0.222.3 Safari/532.2",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.4pre) Gecko/20070404 K-Ninja/2.1.3",
    "Mozilla/5.0 (Future Star Technologies Corp.; Star-Blade OS; x86_64; U; en-US) iNet Browser 4.7",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.13) Gecko/20080414 Firefox/2.0.0.13 Pogo/2.0.0.13.6866",
]

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": USER_AGENTS[0],
    "Referer": "https://www.bing.com/",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

BING_HOST_URL = "https://www.bing.com"
BING_SEARCH_URL = "https://www.bing.com/search?q="


class BingSearchEngine(WebSearchEngine):
    session: Optional[requests.Session] = None

    def __init__(self, **data):
        """Initialize the BingSearch tool with a requests session."""
        super().__init__(**data)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _search_sync(self, query: str, num_results: int = 10) -> List[SearchItem]:
        """
        Synchronous Bing search implementation to retrieve search results.

        Args:
            query (str): The search query to submit to Bing.
            num_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            List[SearchItem]: A list of search items with title, URL, and description.
        """
        if not query:
            return []

        list_result = []
        first = 1
        next_url = BING_SEARCH_URL + query

        while len(list_result) < num_results:
            data, next_url = self._parse_html(
                next_url, rank_start=len(list_result), first=first
            )
            if data:
                list_result.extend(data)
            if not next_url:
                break
            first += 10

        return list_result[:num_results]

    def _parse_html(
            self, url: str, rank_start: int = 0, first: int = 1
    ) -> Tuple[List[SearchItem], str]:
        """
        Parse Bing search result HTML with anti-bot measures.

        Returns:
            tuple: (List of SearchItem objects, next page URL or None)
        """
        try:
            # 1. 动态请求头设置
            new_headers = {
                                "User-Agent": random.choice(USER_AGENTS),
                               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                               "Accept-Language": "en-US,en;q=0.9",
                               "Referer": "https://www.bing.com/",
                             "Upgrade-Insecure-Requests": "1",
                              "Sec-Fetch-Site": "same-origin",
                               "Sec-Fetch-Mode": "navigate",
                               "Sec-Fetch-Dest": "document",
                               "Sec-Ch-Ua": '"Chromium";v="120", "Google Chrome";v="120", ";Not A Brand";v="99"',
                            "Sec-Ch-Ua-Mobile": "?0",
                              "Sec-Ch-Ua-Platform": '"Windows"',
            }

            # 2. 清除会话状态
            # self.session.cookies.clear()
            self.session.headers.update(new_headers)

            # 3. 随机延迟（关键！）
            time.sleep(random.uniform(1, 3))  # 2~5秒随机延迟

            # 4. 发送请求（增加超时和重试）
            res = self.session.get(
                url,
                timeout=10,
                allow_redirects=True  # 允许重定向
            )
            res.raise_for_status()  # 检查HTTP错误

            # 5. 解析HTML（增加容错）
            res.encoding = "utf-8"
            root = BeautifulSoup(res.text, "html.parser")  # 改用更兼容的html.parser

            list_data = []
            ol_results = root.find("ol", id="b_results") or root.find("ol", class_="search-results")  # 备用选择器

            if not ol_results:
                logger.warning("No search results found in HTML")
                return [], None

            for li in ol_results.find_all("li",
                                          class_=lambda x: x and ("b_algo" in x or "search-result" in x)):  # 更灵活的类名匹配
                try:
                    item = {
                        "title": (li.find("h2") or li.find("div", class_="title")).get_text().strip(),
                        "url": (li.find("a") or {}).get("href", "").strip(),
                        "abstract": (li.find("p") or li.find("div", class_="description")).get_text().strip()[
                                    :ABSTRACT_MAX_LENGTH]
                    }
                    list_data.append(
                        SearchItem(
                            title=item["title"] or f"Bing Result {rank_start + 1}",
                            url=item["url"],
                            description=item["abstract"]
                        )
                    )
                    rank_start += 1
                except Exception as e:
                    logger.debug(f"Skipping malformed result: {e}")
                    continue

            # 6. 更稳健的下一页检测
            next_btn = (
                    root.find("a", title="Next page") or
                    root.find("a", string=lambda t: t and "Next" in t)
            )
            next_url = BING_HOST_URL + next_btn["href"] if next_btn else None

            return list_data, next_url

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return [], None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return [], None

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Bing search engine.

        Returns results formatted according to SearchItem model.
        """
        return self._search_sync(query, num_results=num_results)