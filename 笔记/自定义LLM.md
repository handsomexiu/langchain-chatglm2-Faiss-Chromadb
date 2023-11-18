# 编写自定义LLM封装器

[如何编写自定义LLM包装器# – LangChain中文网](https://www.langchain.com.cn/modules/models/llms/examples/custom_llm)

[Langchain（六）进阶：创建自己的LangChain LLM封装器 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/628995493)

自定义LLM仅需要实现一件必需的事情：

- 一个 `_call` 方法，它接收一个字符串，一些可选的停用词，并返回一个字符串

它还可以实现第二个可选项：

- 一个 `_identifying_params` 属性，用于帮助打印该类。应该返回一个字典。

让我们实现一个非常简单的自定义LLM，它只返回输入的前N个字符。

```python
from typing import Any, List, Mapping, Optional
 
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
 
```

```python
class CustomLLM(LLM):
 
    n: int
 
    @property
    def _llm_type(self) -> str:
        return "custom"
 
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[:self.n]
 
    @property
    def _identifying_params(self) -> Mapping[str, Any]:#这是要求返回一个字典类型
		"""Get the identifying parameters."""
        return {"n": self.n}
 
```

```python                          
llm = CustomLLM(n=10)
llm("This is a foobar thing")
>>> 'This is a '
print(llm)
>>> CustomLLM
>>> Params: {'n': 10}
 
```

@property装饰器来创建**只读属性**，@property装饰器会将**方法**转换为相同名称的**只读属性**,可以与所定义的属性配合使用，这样可以防止属性被修改。

**@property 下方的函数只能是self参数,不能有其他的参数**

[python @property的介绍与使用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/64487092)
