# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Doc utilities: Utilities related to documentation
"""

# import re


# def replace_example_docstring(example_docstring):
#     def docstring_decorator(fn):
#         func_doc = fn.__doc__
#         lines = func_doc.split("\n")
#         i = 0
#         while i < len(lines) and re.search(r"^\s*Examples?:\s*$", lines[i]) is None:
#             i += 1
#         if i < len(lines):
#             lines[i] = example_docstring
#             func_doc = "\n".join(lines)
#         else:
#             raise ValueError(
#                 f"The function {fn} should have an empty 'Examples:' in its docstring as placeholder, "
#                 f"current docstring is:\n{func_doc}"
#             )
#         fn.__doc__ = func_doc
#         return fn

#     return docstring_decorator
import re

def replace_example_docstring(example_docstring):
    def docstring_decorator(fn):
        # --- 안전 가드: None이면 빈 문자열로 ---
        func_doc = fn.__doc__ or ""
        lines = func_doc.split("\n") if func_doc else []

        # 'Examples:' 혹은 'Example:' 라인을 찾는다
        placeholder_idx = None
        for i, line in enumerate(lines):
            if re.search(r"^\s*Examples?:\s*$", line) is not None:
                placeholder_idx = i
                break

        if placeholder_idx is not None:
            # 플레이스홀더 자리에 example 문단을 교체
            lines[placeholder_idx] = example_docstring
            fn.__doc__ = "\n".join(lines)
        else:
            # 플레이스홀더가 없으면: 에러 대신 부드럽게 처리
            # 1) 도큐스트링이 비어있으면 example만 넣어주기
            if not lines:
                fn.__doc__ = example_docstring
            else:
                # 2) 끝에 example을 덧붙이기(원하면 이 분기에서 그냥 그대로 fn 반환해도 됨)
                fn.__doc__ = func_doc.rstrip() + "\n\n" + example_docstring
        return fn

    return docstring_decorator