import random
import time
from typing import List
from duckduckgo_search import DDGS
from app.tool.search.base import SearchItem, WebSearchEngine

class DuckDuckGoSearchEngine(WebSearchEngine):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        DuckDuckGo search engine with random delay to avoid rate limiting.
        """
        # 随机延迟 1~3 秒（可调整范围）
        # delay = random.uniform(1, 3)  # 1.0 ~ 3.0 秒之间的随机值
        # time.sleep(delay)  # 暂停执行，模拟人类操作间隔

        # 正常执行搜索
        raw_results = DDGS().text(query, max_results=num_results)

        results = []
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                results.append(
                    SearchItem(
                        title=f"DuckDuckGo Result {i + 1}", url=item, description=None
                    )
                )
            elif isinstance(item, dict):
                results.append(
                    SearchItem(
                        title=item.get("title", f"DuckDuckGo Result {i + 1}"),
                        url=item.get("href", ""),
                        description=item.get("body", None),
                    )
                )
            else:
                try:
                    results.append(
                        SearchItem(
                            title=getattr(item, "title", f"DuckDuckGo Result {i + 1}"),
                            url=getattr(item, "href", ""),
                            description=getattr(item, "body", None),
                        )
                    )
                except Exception:
                    results.append(
                        SearchItem(
                            title=f"DuckDuckGo Result {i + 1}",
                            url=str(item),
                            description=None,
                        )
                    )
        return results