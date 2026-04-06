#!/usr/bin/env python3
"""
实际可运行的示例：展示 LCEL 与 LangGraph 实现 Agentic Loop 的区别
"""

from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


# ============================================================================
# 定义工具
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search(query: str) -> str:
    """搜索信息"""
    # 模拟搜索
    responses = {
        "weather": "The weather is sunny and 70°F",
        "population": "The population is 8 million",
    }
    for key, value in responses.items():
        if key in query.lower():
            return value
    return f"Found information about: {query}"


# ============================================================================
# 方法 1: LCEL 无法实现真正的循环
# ============================================================================

def lcel_attempt_failed():
    """
    演示：LCEL 无法实现动态循环
    """
    print("\n" + "=" * 70)
    print("方法 1: LCEL 尝试（失败 - 无法实现真正的循环）")
    print("=" * 70)

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(temperature=0)

    # 尝试 1: 单次链式调用 - 无法循环
    print("\n❌ 尝试 1: 单次链式调用")
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}")
        ])
        | model.bind_tools([calculator, search])
    )

    result = chain.invoke({"input": "What's 25 * 4?"})
    print(f"输出: {result}")
    print("问题: 只执行一次，如果 LLM 返回 tool_calls，无法自动执行工具并循环")

    # 尝试 2: 手动循环 - 不是真正的 LCEL
    print("\n⚠️  尝试 2: 手动循环（破坏了 LCEL 的优势）")

    def manual_loop(question: str, max_iter: int = 3):
        """必须在 Python 中手动实现循环"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]

        for i in range(max_iter):
            print(f"\n  --- 迭代 {i + 1} ---")

            # 每次都重新创建链
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | model.bind_tools([calculator, search])
            response = chain.invoke({})

            print(f"  LLM 响应: {response.content}")

            # 手动检查是否有 tool_calls
            if not response.tool_calls:
                print("  ✓ 无工具调用，结束")
                return response.content

            # 手动执行工具
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls
            })

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"  → 调用工具: {tool_name}({tool_args})")

                # 手动执行
                if tool_name == "calculator":
                    result = calculator.invoke(tool_args)
                elif tool_name == "search":
                    result = search.invoke(tool_args)
                else:
                    result = "Unknown tool"

                print(f"  ← 工具结果: {result}")

                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call["id"]
                })

        return "Max iterations reached"

    result = manual_loop("What's 25 * 4?")
    print(f"\n最终答案: {result}")
    print("\n问题:")
    print("  - 循环逻辑在 Python 代码中，不在链内部")
    print("  - 无法使用 LCEL 的流式、追踪等特性")
    print("  - 每次迭代都重新创建链，效率低")


# ============================================================================
# 方法 2: LangGraph 原生支持循环
# ============================================================================

def langgraph_native_loop():
    """
    演示：LangGraph 原生支持 Agentic Loop
    """
    print("\n" + "=" * 70)
    print("方法 2: LangGraph 原生循环（正确方法）")
    print("=" * 70)

    from langgraph.graph import StateGraph, END, MessagesState
    from langchain_openai import ChatOpenAI

    # 1. 定义状态（自动追踪所有消息）
    class AgentState(MessagesState):
        """继承 MessagesState，自动管理消息列表"""
        pass

    # 2. 定义节点
    def model_node(state: AgentState):
        """调用 LLM 节点"""
        model = ChatOpenAI(temperature=0)
        bound_model = model.bind_tools([calculator, search])

        response = bound_model.invoke(state["messages"])
        print(f"\n  [Model] 输出: {response.content or '(调用工具)'}")
        if response.tool_calls:
            print(f"  [Model] Tool calls: {[tc['name'] for tc in response.tool_calls]}")

        return {"messages": [response]}

    def tools_node(state: AgentState):
        """执行工具节点"""
        last_message = state["messages"][-1]
        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  [Tools] 执行: {tool_name}({tool_args})")

            # 执行工具
            if tool_name == "calculator":
                result = calculator.invoke(tool_args)
            elif tool_name == "search":
                result = search.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            print(f"  [Tools] 结果: {result}")

            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": tool_messages}

    # 3. 定义条件函数（决定是否循环）
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """决定下一步：继续到 tools 还是结束"""
        last_message = state["messages"][-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"  [Router] → 路由到 tools 节点")
            return "tools"
        else:
            print(f"  [Router] → 结束循环")
            return END

    # 4. 构建图（关键：支持循环！）
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)

    # 设置入口
    graph.set_entry_point("model")

    # 添加条件边（从 model 出发）
    graph.add_conditional_edges(
        "model",
        should_continue,
        {
            "tools": "tools",  # 如果有 tool_calls → 去 tools
            END: END           # 否则结束
        }
    )

    # 关键：循环边！tools 执行后回到 model
    graph.add_edge("tools", "model")

    # 5. 编译
    app = graph.compile()

    # 6. 执行
    print("\n执行图...")
    result = app.invoke({
        "messages": [HumanMessage(content="What's 25 * 4, and what's the weather?")]
    })

    print("\n" + "=" * 70)
    print("完整消息历史:")
    print("=" * 70)
    for i, msg in enumerate(result["messages"], 1):
        role = msg.__class__.__name__
        content = msg.content if hasattr(msg, 'content') else str(msg)
        print(f"{i}. [{role}] {content[:100]}")

    final_answer = result["messages"][-1].content
    print(f"\n最终答案: {final_answer}")

    print("\n优势:")
    print("  ✓ 循环逻辑在图结构中，不需要手动 for 循环")
    print("  ✓ 自动管理消息历史")
    print("  ✓ 支持流式输出、检查点、追踪")
    print("  ✓ 可视化图结构")


# ============================================================================
# 方法 3: 混合方式 - LCEL 组件 + LangGraph 控制
# ============================================================================

def hybrid_approach():
    """
    演示：在 LangGraph 节点中使用 LCEL 链
    """
    print("\n" + "=" * 70)
    print("方法 3: 混合方式（推荐 - 结合两者优势）")
    print("=" * 70)

    from langgraph.graph import StateGraph, END, MessagesState
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # LCEL 链 - 用于单次处理
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when needed."),
        ("placeholder", "{messages}")
    ])

    model = ChatOpenAI(temperature=0)
    model_with_tools = system_prompt | model.bind_tools([calculator, search])

    # LangGraph 图 - 用于控制循环
    def model_node(state: MessagesState):
        """节点内部使用 LCEL 链"""
        print(f"  [Model Node] 调用 LCEL 链...")
        response = model_with_tools.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    def tools_node(state: MessagesState):
        """工具执行"""
        last_message = state["messages"][-1]
        results = []

        for tc in last_message.tool_calls:
            if tc["name"] == "calculator":
                result = calculator.invoke(tc["args"])
            else:
                result = search.invoke(tc["args"])
            results.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        return {"messages": results}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, 'tool_calls') and last.tool_calls) else END

    # 构建图
    graph = StateGraph(MessagesState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    app = graph.compile()

    # 执行
    print("\n执行混合方式...")
    result = app.invoke({
        "messages": [HumanMessage(content="Calculate 15 * 8 and tell me the weather")]
    })

    print(f"\n最终答案: {result['messages'][-1].content}")
    print("\n这种方式:")
    print("  ✓ 节点内用 LCEL（组合 prompt | model | parser）")
    print("  ✓ 图结构用 LangGraph（控制循环逻辑）")
    print("  ✓ 结合两者优势")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "#" * 70)
    print("#  LangChain 链式调用 vs LangGraph Agentic Loop 对比")
    print("#" * 70)

    try:
        # 方法 1: LCEL 失败的尝试
        lcel_attempt_failed()

        # 方法 2: LangGraph 正确的方式
        langgraph_native_loop()

        # 方法 3: 混合方式
        hybrid_approach()

    except Exception as e:
        print(f"\n错误: {e}")
        print("(可能需要设置 OPENAI_API_KEY 环境变量)")

    print("\n" + "#" * 70)
    print("#  总结")
    print("#" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                  关键结论                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LCEL 是 DAG（有向无环图）                                   │
│     → 无法实现真正的循环                                        │
│     → 必须在外部用 Python for 循环                              │
│                                                                 │
│  2. Agentic Loop 需要循环图（Cyclic Graph）                    │
│     → LangGraph 天然支持 add_edge(tools, model)                │
│     → 状态自动管理，无需手动维护                                 │
│                                                                 │
│  3. 最佳实践：混合使用                                          │
│     → 节点内：LCEL 链（prompt | model | parser）               │
│     → 图结构：LangGraph（循环控制）                             │
│                                                                 │
│  4. 何时用 LCEL vs LangGraph?                                  │
│     LCEL:  单向数据处理、RAG 管道、简单链式调用                  │
│     LangGraph: Agents、多步推理、需要循环的场景                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
