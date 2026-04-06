"""
对比 LangChain 链式调用 vs LangGraph 实现 Agentic Loop
"""

# ============================================================================
# 方法 1: LCEL 链式调用 - 无法实现真正的循环
# ============================================================================

def lcel_pseudo_loop():
    """使用 LCEL 的伪循环实现（有局限性）"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI()

    # 预定义固定步骤的链
    chain = (
        {"question": lambda x: x["question"]}
        | ChatPromptTemplate.from_template("Answer: {question}")
        | model
        | StrOutputParser()
    )

    # 问题：无法动态循环，必须在外部手动实现循环逻辑
    def manual_loop(question: str, max_iter: int = 5):
        messages = []
        for i in range(max_iter):
            response = chain.invoke({"question": question})

            # 手动判断逻辑
            if should_stop(response):
                break

            # 手动更新状态
            messages.append(response)

        return response

    return manual_loop


# ============================================================================
# 方法 2: LangGraph - 原生支持循环
# ============================================================================

def langgraph_native_loop():
    """使用 LangGraph 实现原生的 Agentic Loop"""
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated
    from langchain_core.messages import HumanMessage, AIMessage

    # 1. 定义状态
    class AgentState(TypedDict):
        messages: Annotated[list, "append"]
        iteration: int

    # 2. 定义节点
    def call_model(state: AgentState):
        """调用 LLM"""
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI()
        response = model.invoke(state["messages"])
        return {
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1
        }

    def call_tools(state: AgentState):
        """执行工具"""
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls

        tool_messages = []
        for tool_call in tool_calls:
            # 执行工具
            result = execute_tool(tool_call)
            tool_messages.append(result)

        return {"messages": tool_messages}

    # 3. 定义条件边（循环的关键）
    def should_continue(state: AgentState):
        """决定是否继续循环"""
        last_message = state["messages"][-1]

        # 如果有 tool_calls，继续到 tools 节点
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"

        # 否则结束
        return END

    # 4. 构建图（支持循环！）
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)

    # 添加边 - 关键：tools 循环回 model
    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model",
        should_continue,
        {
            "tools": "tools",  # 继续到工具节点
            END: END           # 或者结束
        }
    )
    graph.add_edge("tools", "model")  # ← 循环回 model！

    # 5. 编译
    return graph.compile()


# ============================================================================
# 方法 3: 混合方式 - LCEL 组件 + LangGraph 循环
# ============================================================================

def hybrid_approach():
    """在 LangGraph 节点中使用 LCEL 链"""
    from langgraph.graph import StateGraph, END
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from typing import TypedDict

    # LCEL 链：处理单次 model 调用
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])
    model = ChatOpenAI()
    chain = prompt | model  # LCEL 链

    # LangGraph 状态
    class State(TypedDict):
        input: str
        messages: list
        iterations: int

    # 在节点中使用 LCEL 链
    def model_node(state: State):
        """节点内部使用 LCEL"""
        response = chain.invoke({"input": state["input"]})
        return {
            "messages": state["messages"] + [response],
            "iterations": state["iterations"] + 1
        }

    # LangGraph 控制循环逻辑
    def should_continue(state: State):
        if state["iterations"] >= 5:
            return END
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools"
        return END

    graph = StateGraph(State)
    graph.add_node("model", model_node)
    graph.add_node("tools", lambda s: {"messages": s["messages"]})
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")  # 循环

    return graph.compile()


# ============================================================================
# 完整的 ReAct Agent 实现对比
# ============================================================================

def compare_react_implementations():
    """对比两种 ReAct 实现"""

    # ─────────────────────────────────────────────────────────────────────
    # LCEL 方式：无法实现真正的循环
    # ─────────────────────────────────────────────────────────────────────
    print("=== LCEL 方式（伪循环）===")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    def lcel_react_agent(question: str):
        """手动实现 ReAct 循环"""
        model = ChatOpenAI()
        messages = [{"role": "user", "content": question}]

        # 必须手动写循环
        for i in range(5):
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | model
            response = chain.invoke({})

            # 手动判断
            if not response.tool_calls:
                return response.content

            # 手动执行工具
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)
                messages.append({"role": "tool", "content": result})

        return "Max iterations reached"

    # ─────────────────────────────────────────────────────────────────────
    # LangGraph 方式：原生循环支持
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== LangGraph 方式（原生循环）===")

    from langgraph.graph import StateGraph, END, MessagesState
    from langchain_core.messages import HumanMessage

    def langgraph_react_agent():
        """使用 LangGraph 的 ReAct Agent"""

        def model_node(state: MessagesState):
            model = ChatOpenAI()
            return {"messages": [model.invoke(state["messages"])]}

        def tools_node(state: MessagesState):
            # 自动执行所有 tool_calls
            from langgraph.prebuilt import ToolNode
            tools = [search_tool, calculator_tool]
            tool_node = ToolNode(tools)
            return tool_node.invoke(state)

        def should_continue(state: MessagesState):
            last_msg = state["messages"][-1]
            if last_msg.tool_calls:
                return "tools"
            return END

        # 构建图 - 循环在图结构中定义
        graph = StateGraph(MessagesState)
        graph.add_node("model", model_node)
        graph.add_node("tools", tools_node)
        graph.set_entry_point("model")
        graph.add_conditional_edges("model", should_continue)
        graph.add_edge("tools", "model")  # ← 关键：循环边

        return graph.compile()

    agent = langgraph_react_agent()
    result = agent.invoke({"messages": [HumanMessage("What's 2+2?")]})
    print(result)


# ============================================================================
# 核心对比总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────┐
│                    LCEL vs LangGraph 对比                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  特性              │  LCEL 链式调用        │  LangGraph            │
│  ─────────────────│──────────────────────│──────────────────────  │
│  循环支持          │  ❌ 手动外部循环      │  ✅ 原生图循环        │
│  条件分支          │  ✅ RunnableBranch   │  ✅ conditional_edges │
│  状态管理          │  ❌ 手动管理          │  ✅ StateGraph        │
│  中间状态持久化     │  ❌ 不支持           │  ✅ Checkpointer      │
│  动态停止条件      │  ❌ 需手动实现        │  ✅ 条件函数          │
│  并发执行          │  ✅ RunnableParallel │  ✅ Send              │
│  适用场景          │  简单链式处理        │  复杂 Agent 循环       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

核心结论：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LCEL 本质是 DAG（有向无环图）
   - 优势：简单、组合性好
   - 局限：无法实现真正的循环

2. Agentic Loop 需要循环图（Cyclic Graph）
   - 必须能够"从 tools 回到 model"
   - LCEL 在设计上就不支持这种模式

3. LangGraph 是 Agentic Loop 的正确工具
   - 图结构天然支持循环（add_edge 可以形成环）
   - 状态管理、持久化、条件路由都是内置功能

4. 最佳实践：混合使用
   - 节点内部：使用 LCEL 链（prompt | model | parser）
   - 图结构：使用 LangGraph 控制循环逻辑
"""


def execute_tool(tool_call):
    """模拟工具执行"""
    return "Tool result"


def should_stop(response):
    """判断是否停止"""
    return "FINAL" in response


def search_tool():
    pass


def calculator_tool():
    pass
