#!/usr/bin/env python3
"""Capability taxonomy and scenario trigger constants."""
from __future__ import annotations


REQUIRED_CAPABILITY_IDS = [
    "local_execution",
    "project_governance",
    "workflow_packets",
    "task_scoping",
    "cognitive_search",
    "evolution_loop",
    "independent_review",
    "user_cognition",
    "tool_boundary",
    "evaluation_runner",
    "prompt_optimizer",
    "durable_execution",
    "observability",
]

REQUIRED_EXTERNAL_SOURCES = [
    "Standalone CLI",
    "AGENTS.md",
    "Skills",
    "Trellis",
    "MCP",
    "CheckModel",
    "Tree of Thoughts",
    "Graph of Thoughts",
    "LangGraph",
    "DSPy",
    "GEPA",
    "Promptfoo",
    "SkyDiscover",
]

CAPABILITY_KEYWORDS = {
    "local_execution": ["standalone", "cli", "run", "doctor", "shell", "driver", "api", "api接入", "驱动", "模型调用", "命令", "执行", "校验"],
    "project_governance": ["agents.md", "governance", "definition of done", "complexity", "治理", "复杂度", "完成标准", "框架"],
    "workflow_packets": ["skill", "skills", "workflow packet", "workflow", "技能", "工作流"],
    "task_scoping": ["scope", "scoping", "intake", "prd", "requirement", "requirements", "需求", "范围", "入口", "认知增强", "任务开头", "问题定义", "问题", "spec"],
    "cognitive_search": ["thought", "tree", "graph", "search", "candidate", "候选", "搜索", "分叉", "方案", "tot", "got"],
    "evolution_loop": ["evolve", "evolution", "mutation", "archive", "selector", "演化", "进化", "评估", "选择", "调优", "优化"],
    "independent_review": ["checkmodel", "review", "critic", "risk", "审查", "评审", "复核", "风险"],
    "user_cognition": ["cognition", "feedback", "assumption", "validation", "认知", "假设", "反馈", "验证", "用户"],
    "tool_boundary": ["mcp", "tool", "browser", "external", "工具", "外部", "文档", "检索"],
    "evaluation_runner": ["eval", "promptfoo", "test", "regression", "测试", "回归", "红队", "评测"],
    "prompt_optimizer": ["dspy", "gepa", "optimizer", "prompt optimizer", "prompt optimization", "优化器", "提示词优化"],
    "durable_execution": ["langgraph", "durable", "resume", "interrupt", "human-in-loop", "多天", "恢复", "审批"],
    "observability": ["trace", "observability", "langfuse", "phoenix", "telemetry", "追踪", "可观测", "诊断"],
}

FRAMEWORK_EVOLUTION_CAPABILITY_IDS = [
    "project_governance",
    "task_scoping",
    "cognitive_search",
    "independent_review",
    "user_cognition",
    "observability",
]

NATIVE_RUNTIME_CAPABILITY_IDS = [
    "tool_boundary",
    "durable_execution",
    "cognitive_search",
    "evolution_loop",
    "independent_review",
    "user_cognition",
    "observability",
]

PROJECTED_RUNTIME_CAPABILITY_IDS = {
    "workflow_packets",
    "evaluation_runner",
    "prompt_optimizer",
}

SCENARIO_CAPABILITY_TERMS = {
    "workflow_packets": ["skill", "skills", "workflow packet", "技能", "工作流包"],
    "tool_boundary": ["tool", "mcp", "browser", "工具", "外部工具", "权限边界"],
    "independent_review": ["review", "checkmodel", "risk", "审查", "评审", "复核", "风险"],
    "evaluation_runner": ["eval", "evaluation", "promptfoo", "测试", "评测", "回归"],
    "prompt_optimizer": ["prompt", "dspy", "gepa", "提示词", "优化器"],
    "durable_execution": ["langgraph", "durable", "resume", "state", "持久", "恢复", "状态"],
    "observability": ["trace", "observability", "telemetry", "追踪", "可观测", "诊断"],
}

MANDATORY_AI_BACKEND_CAPABILITIES = [
    "durable_execution",
    "cognitive_search",
    "evolution_loop",
    "independent_review",
    "observability",
]

